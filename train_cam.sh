#!/bin/bash
# train_camera.sh

export MASTER_ADDR=localhost
export MASTER_PORT=6009
export GPUS_PER_NODE=1
export NNODES=1
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OFFLOAD_T5_CACHE=true
export OFFLOAD_VAE_CACHE=true
export SKIP_LOAD_MODEL=true

MAGI_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH="$MAGI_ROOT:$PYTHONPATH"

python train_camera_magi.py \
    --dataset_path /home/zhuyixuan05/ReCamMaster/example_test_data \
    --config_path example/4.5B/4.5B_config.json \
    --output_path ./outputs/camera_model \
    --learning_rate 1e-5 \
    --batch_size 1 \
    --max_epochs 10 \
    --steps_per_epoch 500 \
    --num_workers 4 \
    --use_gradient_checkpointing