#!/bin/bash

export MASTER_ADDR=localhost
export MASTER_PORT=6010
export GPUS_PER_NODE=1
export NNODES=1
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=0
export RANK=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OFFLOAD_T5_CACHE=true
export OFFLOAD_VAE_CACHE=true

export TORCH_DISTRIBUTED_DEBUG=INFO

MAGI_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH="$MAGI_ROOT:$PYTHONPATH"

# Increased tile sizes from 34 to 40 to avoid VAE kernel size error
python data_process.py \
    --dataset_path ./example_test_data \
    --output_path ./processed_data \
    --config_path example/4.5B/4.5B_config.json \
    --tiled \
    --tile_size_height 40 \
    --tile_size_width 40 \
    --tile_stride_height 20 \
    --tile_stride_width 20 \
    --num_frames 81 \
    --height 480 \
    --width 832 \
    --dataloader_num_workers 2