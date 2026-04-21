#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS=4 # number of GPUs, aligned with CUDA_DEVICES, can be 1 for single GPU training
CUDA_DEVICES=0,1,2,3 # GPU IDs

cd "$ROOT_DIR"
source env.sh # link to mmcv, mmdet and mmseg
micromamba activate torch29 # change to your environment name

mkdir -p work_dirs/logs

# dinov3 examples
# dinov3 with maskrcnn
mkdir -p work_dirs/coco2017_maskrcnn_dinov3
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" bash detection/dist_train.sh \
    my-configs/coco2017_maskrcnn_dinov3.py \
    "$GPUS" \
    --work-dir work_dirs/coco2017_maskrcnn_dinov3 \
    --cfg-options model.backbone.adapter_mode=official_adapter \
    | tee work_dirs/logs/coco2017_maskrcnn_dinov3.log

# dinov3 with fasterrcnn
mkdir -p work_dirs/coco2017_fasterrcnn_dinov3
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" bash detection/dist_train.sh \
    my-configs/coco2017_fasterrcnn_dinov3.py \
    "$GPUS" \
    --work-dir work_dirs/coco2017_fasterrcnn_dinov3 \
    --cfg-options model.backbone.adapter_mode=official_adapter \
    | tee work_dirs/logs/coco2017_fasterrcnn_dinov3.log

