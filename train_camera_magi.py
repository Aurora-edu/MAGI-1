import argparse
import os
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import lightning as pl
from tqdm import tqdm
import math
import re
import json
import wandb

from inference.pipeline import MagiPipeline
from inference.model.dit.dit_model import get_dit
from inference.common import MagiConfig
from inference.pipeline.video_process import process_camera_trajectory
from inference.infra.distributed import dist_init, get_device, get_world_size, is_last_rank, is_last_tp_cp_rank

# 在训练脚本开头添加
class GradientLogger(pl.Callback):
    def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx=0):
        """优化器步进前被调用 - 这个比 on_after_backward 更可靠，特别是对 DeepSpeed"""
        if is_last_rank():
            print(f"\n[Step {trainer.global_step}] Gradient Summary:")
            
            # 统计每个模块的梯度状态
            cam_modules_with_grad = 0
            cam_modules_total = 0
            
            for name, module in pl_module.magi_model.named_modules():
                if "cam_encoder" in name or "projector" in name:
                    cam_modules_total += 1
                    has_grad = False
                    for param_name, param in module.named_parameters():
                        if param.grad is not None and param.grad.norm().item() > 0:
                            has_grad = True
                            break
                    
                    if has_grad:
                        cam_modules_with_grad += 1
            
            print(f"  Camera modules with gradients: {cam_modules_with_grad}/{cam_modules_total}")
            
            # 随机选择一个模块详细打印
            import random
            modules = [m for n, m in pl_module.magi_model.named_modules() 
                       if "cam_encoder" in n or "projector" in n]
            
            if modules:
                sample_module = random.choice(modules)
                print(f"  Sample module params:")
                for param_name, param in sample_module.named_parameters():
                    if param.grad is not None:
                        print(f"    {param_name}: grad={param.grad.norm().item():.6f}, param={param.norm().item():.6f}")

class CameraDataset(Dataset):
    """Dataset for camera-conditioned video generation training."""
    
    def __init__(self, base_path, metadata_path, steps_per_epoch=500):
        """
        Args:
            base_path: Base directory containing the data
            metadata_path: Path to the metadata CSV file
            steps_per_epoch: Number of steps per epoch
        """
        import pandas as pd
        metadata = pd.read_csv(metadata_path)
        self.paths = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        self.tensor_paths = [p + ".tensors.pth" for p in self.paths if os.path.exists(p + ".tensors.pth")]
        print(f"Found {len(self.tensor_paths)} tensor files")
        self.steps_per_epoch = steps_per_epoch
    
    def parse_camera_matrix(self, matrix_str):
        """Parse camera matrix string into numpy array."""
        rows = matrix_str.strip().split('] [')
        matrix = []
        for row in rows:
            row = row.replace('[', '').replace(']', '')
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)
    
    def get_relative_pose(self, cam_params):
        """Calculate relative poses between cameras."""
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        return np.array(ret_poses, dtype=np.float32)
    
    def __getitem__(self, index):
        """Return a data sample."""
        data = {}
        data_id = (index + torch.randint(0, len(self.tensor_paths), (1,))[0]) % len(self.tensor_paths)
        
        try:
            # Load latent tensors
            path_tgt = self.tensor_paths[data_id]
            data_tgt = torch.load(path_tgt, weights_only=True, map_location="cpu")
            
            # Load target and condition latents
            match = re.search(r'cam(\d+)', path_tgt)
            tgt_idx = int(match.group(1))
            cond_idx = random.randint(1, 10)
            while cond_idx == tgt_idx:
                cond_idx = random.randint(1, 10)
            path_cond = re.sub(r'cam(\d+)', f'cam{cond_idx:02}', path_tgt)
            data_cond = torch.load(path_cond, weights_only=True, map_location="cpu")
            
            # Combine latents
            data['latents'] = torch.cat((data_tgt['latents'], data_cond['latents']), dim=1)
            data['prompt_emb'] = data_tgt['prompt_emb']
            
            # Load camera trajectories
            base_path = path_tgt.rsplit('/', 2)[0]
            camera_path = os.path.join(base_path, "cameras", "camera_extrinsics.json")
            with open(camera_path, 'r') as file:
                cam_data = json.load(file)
            
            # Process camera data for source and target views
            multiview_c2ws = []
            cam_idx = list(range(81))[::4]  # Sample frames at interval of 4
            
            # Process camera matrices for each view
            for view_idx in [cond_idx, tgt_idx]:
                traj = [self.parse_camera_matrix(cam_data[f"frame{idx}"][f"cam{view_idx:02d}"]) for idx in cam_idx]
                traj = np.stack(traj).transpose(0, 2, 1)
                c2ws = []
                for c2w in traj:
                    c2w = c2w[:, [1, 2, 0, 3]]
                    c2w[:3, 1] *= -1.
                    c2w[:3, 3] /= 100
                    c2ws.append(c2w)
                multiview_c2ws.append(c2ws)
            
            # Calculate relative camera poses
            from collections import namedtuple
            Camera = namedtuple('Camera', ['c2w_mat', 'w2c_mat'])
            cond_cam_params = [Camera(c2w, np.linalg.inv(c2w)) for c2w in multiview_c2ws[0]]
            tgt_cam_params = [Camera(c2w, np.linalg.inv(c2w)) for c2w in multiview_c2ws[1]]
            
            relative_poses = []
            for i in range(len(tgt_cam_params)):
                relative_pose = self.get_relative_pose([cond_cam_params[0], tgt_cam_params[i]])
                relative_poses.append(torch.as_tensor(relative_pose)[:,:3,:][1])
                
            camera_embedding = torch.stack(relative_poses, dim=0)  # [num_frames, 3, 4]
            camera_embedding = camera_embedding.reshape(camera_embedding.shape[0], -1)  # [num_frames, 12]
            data['camera'] = camera_embedding.to(torch.float32)
            return data
        
        except Exception as e:
            print(f"Error loading sample: {e}")
            # Return a different sample
            return self.__getitem__((index + 1) % len(self.tensor_paths))
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.steps_per_epoch


class MAGICameraTrainer(pl.LightningModule):
    """Lightning module for training camera conditioning in MAGI."""
    
    def __init__(
        self,
        config_path,
        learning_rate=1e-5,
        use_gradient_checkpointing=True,
        resume_ckpt_path=None,
        wandb_project="camera-magi",  
        wandb_run_name=None           
    ):
        super().__init__()
        # Initialize MAGI pipeline and model
        self.config = MagiConfig.from_json(config_path)
        self.magi_model = get_dit(self.config)
        
        # Add camera encoder modules to transformer layers
        self.add_camera_modules()
        
        # Load checkpoint if provided
        if resume_ckpt_path is not None:
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            self.magi_model.load_state_dict(state_dict, strict=False)
        
        # Freeze most parameters
        self.freeze_parameters()
        
        # Unfreeze camera-related modules
        self.unfreeze_camera_modules()
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        
    def add_camera_modules(self):
        """Add camera encoder modules to all transformer layers."""
        from inference.model.dit.dit_module import TransformerLayer
        
        # Find the appropriate dimension
        dim = self.magi_model.model_config.hidden_size
        
        # Add camera encoder to each transformer layer
        for module in self.magi_model.modules():
            if isinstance(module, TransformerLayer):
                # Initial camera encoder from camera embedding (12) to intermediate size
                module.cam_encoder = nn.Linear(12, dim)
                
                # Projector to map to final hidden size ensuring compatibility with model dimensions
                module.projector = nn.Linear(dim, dim)
                
                # Initialize with zeros
                module.cam_encoder.weight.data.zero_()
                module.cam_encoder.bias.data.zero_()
                
                # Initialize projector as identity
                module.projector.weight = nn.Parameter(torch.eye(dim))
                module.projector.bias = nn.Parameter(torch.zeros(dim))    
    
    def freeze_parameters(self):
        """Freeze all parameters of the model."""
        for param in self.magi_model.parameters():
            param.requires_grad = False
        self.magi_model.eval()
    
    def unfreeze_camera_modules(self):
        """Unfreeze camera-related and self-attention modules."""
        for name, module in self.magi_model.named_modules():
            if any(keyword in name for keyword in ["cam_encoder", "projector", "self_attn"]):
                print(f"Trainable: {name}")
                # 设置为训练模式
                module.train()
                # 确保参数可训练
                for param in module.parameters():
                    param.requires_grad = True
    
    def on_fit_start(self):
        """训练开始时初始化 wandb"""
        if is_last_rank():  # 只在最后一个 rank 上初始化
            import os
            
            # 更安全地获取 batch_size
            batch_size = 1  # 默认值
            if hasattr(self.trainer, 'train_dataloader'):
                train_dataloader = self.trainer.train_dataloader
                if train_dataloader is not None and hasattr(train_dataloader, 'batch_size'):
                    batch_size = train_dataloader.batch_size
                # 如果上面方法不行，尝试从 dataloader_iter 获取
                elif hasattr(self.trainer, 'train_dataloader_iter') and self.trainer.train_dataloader_iter is not None:
                    dataloader = self.trainer.train_dataloader_iter.loaders
                    if hasattr(dataloader, 'batch_size'):
                        batch_size = dataloader.batch_size
            
            # 更安全地获取 max_epochs
            max_epochs = self.trainer.max_epochs if hasattr(self.trainer, 'max_epochs') else 10
            
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name or f"camera_training_{os.path.basename(self.trainer.default_root_dir)}",
                config={
                    "learning_rate": self.learning_rate,
                    "batch_size": batch_size,
                    "max_epochs": max_epochs,
                    "gradient_checkpointing": self.use_gradient_checkpointing,
                }
            )
            
            # 计算可训练参数数量
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            wandb.run.summary["trainable_parameters"] = trainable_params

    def training_step(self, batch, batch_idx):
        """Training step that better simulates inference autoregressive generation."""
        # Extract data
        latents = batch["latents"].to(self.device)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        camera_emb = batch["camera"].to(self.device).requires_grad_(True)
            
        # Generate noise
        noise = torch.randn_like(latents)
        
        # Choose random timestep
        timestep_id = torch.randint(0, self.config.runtime_config.num_steps, (1,), device=self.device)
        
        # Create noisy latents
        origin_latents = latents.clone()
        
        # Target part is the first half for source-to-target generation
        tgt_latent_len = latents.shape[2] // 2
        
        # Simulate chunk-wise generation
        chunk_width = self.config.runtime_config.chunk_width
        num_chunks = tgt_latent_len // chunk_width
        
        # Randomly select a window size as in inference (min 1, max 4, but not more than num_chunks)
        window_size = min(random.randint(1, 4), num_chunks)
        
        if num_chunks > 1 and window_size > 0:
            # Choose a random starting chunk (simulating a point in the generation process)
            max_start = max(0, num_chunks - window_size)
            start_chunk = random.randint(0, max_start)
            end_chunk = start_chunk + window_size
            
            # Create timesteps with varying noise levels for chunks in the window
            # This simulates different denoising stages for different chunks
            chunk_timesteps = []
            for i in range(window_size):
                # Simulate decreasing noise levels for later chunks (more denoised)
                # Ensures proper autoregressive conditioning
                chunk_ts = max(1, timestep_id.item() - i * (timestep_id.item() // window_size))
                chunk_timesteps.append(chunk_ts)
            
            chunk_timesteps = torch.tensor(chunk_timesteps, device=self.device)
            
            # Apply noise with varying levels to the chunks in the window
            noisy_latents = origin_latents.clone()
            
            # 1. Set everything before the window to clean (already generated)
            for i in range(start_chunk):
                chunk_start = i * chunk_width
                chunk_end = (i + 1) * chunk_width
                # Clean chunks are exactly like the original
                noisy_latents[:, :, chunk_start:chunk_end, ...] = origin_latents[:, :, chunk_start:chunk_end, ...]
            
            # 2. Add varying noise to the chunks in the window
            for i in range(window_size):
                chunk_idx = start_chunk + i
                chunk_start = chunk_idx * chunk_width
                chunk_end = (chunk_idx + 1) * chunk_width
                
                # Apply noise according to this chunk's timestep
                chunk_noise_scale = chunk_timesteps[i] / self.config.runtime_config.num_steps
                noisy_latents[:, :, chunk_start:chunk_end, ...] = (
                    origin_latents[:, :, chunk_start:chunk_end, ...] + 
                    noise[:, :, chunk_start:chunk_end, ...] * chunk_noise_scale
                )
            
            # 3. Keep source data clean for conditioning (second half of the latents)
            noisy_latents[:, :, tgt_latent_len:, ...] = origin_latents[:, :, tgt_latent_len:, ...]
            
            # Setup KV range similar to inference
            start_tokens = start_chunk * chunk_width
            end_tokens = end_chunk * chunk_width
            tokens_in_window = end_tokens - start_tokens
            
            window_token_nums = (
                tokens_in_window
                * (self.config.runtime_config.video_size_h // self.config.model_config.patch_size) 
                * (self.config.runtime_config.video_size_w // self.config.model_config.patch_size)
            )
            
            # Create KV range that includes all previous chunks (similar to inference)
            kv_range = torch.tensor([[0, window_token_nums]], dtype=torch.int32, device=self.device)
            
            # Create a timestep tensor matching inference shape: [batch_size, denoising_range_num]
            timestep = chunk_timesteps.unsqueeze(0)  # Shape: [1, window_size]
            
            # Select camera embeddings for current window - matching the exact expected shape
            #window_camera_emb = camera_emb[:, start_chunk:end_chunk, :] if camera_emb.dim() > 2 else camera_emb[start_chunk:end_chunk]
            window_camera_emb = camera_emb
            #暂时给每个chunk都注入所有的camera embedding
            # Parameters for model call
            window_params = {
                "fwd_extra_1st_chunk": False,
                "range_num": end_chunk,
                "denoising_range_num": window_size,
                "slice_point": start_chunk,
                "chunk_width": chunk_width,
                "num_steps": self.config.runtime_config.num_steps,
            }
            
            # Setup attention mask
            emb_length = prompt_emb["context"].shape[1]
            xattn_mask = torch.ones((1, 1, min(self.config.model_config.caption_max_length, emb_length)), device=self.device)
            xattn_mask = xattn_mask.repeat(window_size, 1, 1)
            
            # Setup dropout mask
            caption_dropout_mask = torch.tensor([False] * timestep.shape[0], dtype=torch.bool, device=self.device)
            
            # Setup inference params
            from inference.common import InferenceParams
            max_sequence_length = (
                noisy_latents.shape[2] 
                * (noisy_latents.shape[3] // self.config.model_config.patch_size) 
                * (noisy_latents.shape[4] // self.config.model_config.patch_size)
            )
            inference_params = InferenceParams(max_batch_size=1, max_sequence_length=max_sequence_length)
            
            # Model forward pass
            noise_pred = self.magi_model(
                x=noisy_latents,
                t=timestep,
                y=prompt_emb["context"].repeat(window_size, 1, 1, 1),
                caption_dropout_mask=caption_dropout_mask,
                xattn_mask=xattn_mask,
                kv_range=kv_range,
                inference_params=inference_params,
                cam_emb=window_camera_emb,
                **window_params
            )
            
            # Compute loss only on window chunks
            target_slice = noise[:, :, start_tokens:end_tokens, ...]
            pred_slice = noise_pred[:, :, start_tokens:end_tokens, ...]
            
            # Weight loss by timestep to focus more on lower-noise steps
            weighted_loss = 0
            for i in range(window_size):
                chunk_idx = i
                chunk_start = chunk_idx * chunk_width
                chunk_end = min((chunk_idx + 1) * chunk_width, target_slice.shape[2])
                
                # Calculate chunk-specific loss
                chunk_target = target_slice[:, :, chunk_start:chunk_end, ...]
                chunk_pred = pred_slice[:, :, chunk_start:chunk_end, ...]
                chunk_loss = F.mse_loss(chunk_pred.float(), chunk_target.float())
                
                # Weight by inverse timestep (lower noise = higher weight)
                timestep_weight = 1.0 - (chunk_timesteps[i] / self.config.runtime_config.num_steps)
                weighted_loss += chunk_loss * (1.0 + timestep_weight)
                
            loss = weighted_loss / window_size
                
        else:
            # Fallback for single-chunk case
            noisy_latents = latents + noise * timestep_id.view(-1, 1, 1, 1, 1) / self.config.runtime_config.num_steps
            noisy_latents[:, :, tgt_latent_len:, ...] = origin_latents[:, :, tgt_latent_len:, ...]
            
            # Setup for single chunk
            kv_range = torch.tensor([[0, tgt_latent_len * (self.config.runtime_config.video_size_h // self.config.model_config.patch_size) * 
                                    (self.config.runtime_config.video_size_w // self.config.model_config.patch_size)]], 
                                    dtype=torch.int32, device=self.device)
            
            timestep = torch.tensor([[timestep_id.item()]], device=self.device)
            
            # Setup attention mask
            emb_length = prompt_emb["context"].shape[1]
            xattn_mask = torch.ones((1, 1, min(self.config.model_config.caption_max_length, emb_length)), device=self.device)
            
            # Setup dropout mask
            caption_dropout_mask = torch.tensor([False], dtype=torch.bool, device=self.device)
            
            # Setup inference params
            from inference.common import InferenceParams
            max_sequence_length = (
                noisy_latents.shape[2] 
                * (noisy_latents.shape[3] // self.config.model_config.patch_size) 
                * (noisy_latents.shape[4] // self.config.model_config.patch_size)
            )
            inference_params = InferenceParams(max_batch_size=1, max_sequence_length=max_sequence_length)
            
            # Model forward pass
            noise_pred = self.magi_model(
                x=noisy_latents,
                t=timestep,
                y=prompt_emb["context"],
                caption_dropout_mask=caption_dropout_mask,
                xattn_mask=xattn_mask,
                kv_range=kv_range,
                inference_params=inference_params,
                cam_emb=camera_emb,
                fwd_extra_1st_chunk=False,
                range_num=1,
                denoising_range_num=1,
                slice_point=0,
                chunk_width=tgt_latent_len,
                num_steps=self.config.runtime_config.num_steps,
            )
            
            loss = F.mse_loss(
                noise_pred[:, :, :tgt_latent_len, ...].float(), 
                noise[:, :, :tgt_latent_len, ...].float()
            )

        # Log metrics
        if is_last_rank():
            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": self.learning_rate,
                "global_step": self.global_step,
                "camera_embedding_norm": camera_emb.norm().item(),
                "camera_grad_norm": camera_emb.grad.norm().item() if camera_emb.grad is not None else 0,
                "window_size": window_size if num_chunks > 1 and window_size > 0 else 1,
            })

        # Log loss
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def on_train_end(self):
        """训练结束时关闭 wandb"""
        if is_last_rank():
            wandb.finish()

    def configure_optimizers(self):
        """Configure optimizers for training."""
        trainable_modules = filter(lambda p: p.requires_grad, self.magi_model.parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    
    def on_save_checkpoint(self, checkpoint):
        """Custom checkpoint saving to only save model state dict."""
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        current_step = self.global_step
        
        # Clear checkpoint to save space
        checkpoint.clear()
        
        # Get state dict of trainable parameters
        trainable_param_names = list(filter(
            lambda named_param: named_param[1].requires_grad, 
            self.magi_model.named_parameters()
        ))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        
        # Save model state dict
        state_dict = self.magi_model.state_dict()
        torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Camera-conditioned MAGI")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the MAGI config file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Number of epochs",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
        help="Path to resume training from checkpoint",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="metadata.csv",
        help="Name of the metadata file",
    )
    return parser.parse_args()


def main():
    """Main function to train camera conditioning for MAGI."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok=True)

    # Create dataset
    dataset = CameraDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, args.metadata_file),
        steps_per_epoch=args.steps_per_epoch,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    # Initialize distributed environment first
    config = MagiConfig.from_json(args.config_path)
    dist_init(config)
    
    # Create trainer model
    model = MAGICameraTrainer(
        config_path=args.config_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        resume_ckpt_path=args.resume_ckpt_path,
    )
    
    from lightning.pytorch.strategies import DeepSpeedStrategy
    # Create Lightning trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        # precision="32",
        # strategy=DeepSpeedStrategy(
        #     stage=1,
        #     offload_optimizer=False,
        #     offload_parameters=False
        # ),
        precision="bf16",
        strategy="deepspeed_stage_1",
        default_root_dir=args.output_path,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1),
                   #GradientLogger()
                   ],
    )
    
    # Train model
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
