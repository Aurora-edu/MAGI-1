import argparse
import os
import torch
import numpy as np
import pandas as pd
import json
import imageio
from tqdm import tqdm
from einops import rearrange
from PIL import Image
from torchvision.transforms import v2
import lightning as pl

from inference.common import MagiConfig
from inference.infra.distributed import dist_init
from inference.model.vae import AutoModel
from inference.pipeline.prompt_process import get_txt_embeddings
from inference.pipeline.video_process import VaeHelper

class CameraVideoDataset(torch.utils.data.Dataset):
    """Dataset for processing multi-camera video data."""
    
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, 
                 num_frames=81, height=480, width=832, is_i2v=False):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = v2.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=v2.InterpolationMode.BILINEAR
        )
        return image
    
    def load_frames_using_imageio(self, file_path):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < self.max_num_frames:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(self.num_frames):
            frame = reader.get_data(frame_id * self.frame_interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = self.frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        
        if self.is_i2v:
            return frames, first_frame
        else:
            return frames
    
    def load_camera_trajectory(self, video_path):
        """Load camera trajectory data for a video."""
        # Extract base directory and camera ID
        base_dir = os.path.dirname(os.path.dirname(video_path))
        match = os.path.basename(video_path).split('.')[0]
        video_id = match.split('_')[0] if '_' in match else match
        
        # Fix the camera ID extraction
        # Old code that's causing the error:
        # cam_id = int(os.path.basename(video_path).split('cam')[1].split('_')[0])
        
        # New extraction logic
        filename = os.path.basename(video_path)
        if 'cam' in filename:
            cam_part = filename.split('cam')[1]
            # Extract just the digits before the next non-digit character
            cam_id = ""
            for char in cam_part:
                if char.isdigit():
                    cam_id += char
                else:
                    break
            cam_id = int(cam_id) if cam_id else 1  # Default to 1 if extraction fails
        else:
            cam_id = 1  # Default camera ID if not found
        
        # Path to camera trajectory file
        camera_file = os.path.join(base_dir, "cameras", "camera_extrinsics.json")
        
        if not os.path.exists(camera_file):
            return None
        
        # Rest of the method remains unchanged
        try:
            with open(camera_file, 'r') as f:
                camera_data = json.load(f)
                
            # Extract camera matrices for all frames
            camera_matrices = []
            for frame_idx in range(self.num_frames):
                frame_key = f"frame{frame_idx}"
                cam_key = f"cam{cam_id:02d}"
                
                if frame_key in camera_data and cam_key in camera_data[frame_key]:
                    # Parse the camera matrix string
                    matrix_str = camera_data[frame_key][cam_key]
                    rows = matrix_str.strip().split('] [')
                    matrix = []
                    for row in rows:
                        row = row.replace('[', '').replace(']', '')
                        matrix.append(list(map(float, row.split())))
                    
                    # Convert to 3x4 matrix
                    camera_matrix = np.array(matrix).reshape(4, 4)[:3]
                    camera_matrices.append(camera_matrix)
                else:
                    # Use identity matrix if data is missing
                    camera_matrices.append(np.eye(3, 4))
            
            return torch.tensor(np.array(camera_matrices), dtype=torch.float32)
        except Exception as e:
            print(f"Error loading camera data: {e}")
            return None
        
    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        
        try:
            video = self.load_frames_using_imageio(path)
            camera_trajectory = self.load_camera_trajectory(path)
            
            if video is None:
                raise ValueError(f"Could not load video: {path}")
                
            if self.is_i2v:
                video, first_frame = video
                data = {
                    "text": text, 
                    "video": video, 
                    "path": path, 
                    "first_frame": first_frame,
                    "camera_trajectory": camera_trajectory
                }
            else:
                data = {
                    "text": text, 
                    "video": video, 
                    "path": path,
                    "camera_trajectory": camera_trajectory
                }
            return data
        except Exception as e:
            print(f"Error processing {path}: {e}")
            # Return a different sample if this one fails
            if data_id + 1 < len(self.path):
                return self.__getitem__(data_id + 1)
            else:
                return self.__getitem__(0)  # Fallback to first item
    
    def __len__(self):
        return len(self.path)

class MAGIDataProcessor(pl.LightningModule):
    """Lightning module for processing and saving MAGI data."""
    
    def __init__(self, config_path, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        self.config = MagiConfig.from_json(config_path)
        self.tiled = tiled
        self.tile_size = tile_size
        self.tile_stride = tile_stride
    
    def test_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]
        camera_trajectory = batch.get("camera_trajectory", None)
        
        # Output file path
        pth_path = path + ".tensors.pth"
        if os.path.exists(pth_path):
            print(f"File {pth_path} already exists, skipping.")
            return
        
        # Process text embedding
        caption_embs, emb_masks = get_txt_embeddings(text, self.config)
        
        # Process video
        video = video.to(dtype=torch.bfloat16, device=self.device)
        vae = VaeHelper.get_vae(self.config.runtime_config.vae_pretrained)
        
        # Encode video with tiling if enabled
        if self.tiled:
            latents = VaeHelper.encode(
                video,
                vae,
                tile_sample_min_height=self.tile_size[0],
                tile_sample_min_width=self.tile_size[1],
                spatial_tile_overlap_factor=0.25,
                temporal_tile_overlap_factor=0,
                tile_sample_min_length=self.config.runtime_config.fps // 2,
                allow_spatial_tiling=True
            )
        else:
            latents = VaeHelper.patch_vae_encode(vae, video)
        
        # Scale latents
        latents = latents * self.config.runtime_config.scale_factor
        
        # Process camera trajectory if available
        camera_emb = None
        if camera_trajectory is not None and camera_trajectory.shape[0] > 0:
            from inference.pipeline.video_process import process_camera_trajectory
            camera_emb = process_camera_trajectory(camera_trajectory[0], self.config)
        
        # Save processed data
        data = {
            "latents": latents,
            "caption_embs": caption_embs,
            "emb_masks": emb_masks,
            "camera_emb": camera_emb
        }
        
        torch.save(data, pth_path)
        print(f"Processed and saved: {pth_path}")

def data_process_magi(args):
    """Process dataset for camera-conditioned MAGI training."""
    # Create dataset
    dataset = CameraVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, args.metadata_file),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_mode
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    
    # Initialize MAGI processor
    processor = MAGIDataProcessor(
        config_path=args.config_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width)
    )
    
    # Initialize distributed environment first
    config = MagiConfig.from_json(args.config_path)
    dist_init(config)
    
    # Create trainer and process data
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path
    )
    
    trainer.test(processor, dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data for camera-conditioned MAGI training")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_path", type=str, default="./processed_data", help="Output path for processed data")
    parser.add_argument("--config_path", type=str, required=True, help="Path to MAGI config file")
    parser.add_argument("--tiled", action="store_true", help="Enable tiled encoding in VAE to reduce VRAM usage")
    parser.add_argument("--tile_size_height", type=int, default=34, help="Tile height for VAE encoding")
    parser.add_argument("--tile_size_width", type=int, default=34, help="Tile width for VAE encoding")
    parser.add_argument("--tile_stride_height", type=int, default=18, help="Tile stride height for VAE encoding")
    parser.add_argument("--tile_stride_width", type=int, default=16, help="Tile stride width for VAE encoding")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to process")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--width", type=int, default=832, help="Frame width")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--metadata_file", type=str, default="metadata.csv", help="Metadata filename")
    parser.add_argument("--image_mode", action="store_true", help="Process as image-to-video (first frame only)")
    
    args = parser.parse_args()
    data_process_magi(args)