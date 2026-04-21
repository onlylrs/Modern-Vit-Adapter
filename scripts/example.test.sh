#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_DEVICE=0 # single GPU id for testing

cd "$REPO_ROOT"
source "$REPO_ROOT/env.sh" # link to mmcv, mmdet and mmseg
micromamba activate torch29 # change to your environment name

mkdir -p work_dirs/logs

# dinov3 examples
# dinov3 with maskrcnn
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python detection/test.py \
    my-configs/coco2017_maskrcnn_dinov3.py \
    work_dirs/coco2017_maskrcnn_dinov3/latest.pth \
    --work-dir work_dirs/coco2017_maskrcnn_dinov3/eval \
    --eval bbox segm \
    --cfg-options model.backbone.adapter_mode=official_adapter \
    | tee work_dirs/logs/coco2017_maskrcnn_dinov3_eval.log

# dinov3 with fasterrcnn
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python detection/test.py \
    my-configs/coco2017_fasterrcnn_dinov3.py \
    work_dirs/coco2017_fasterrcnn_dinov3/latest.pth \
    --work-dir work_dirs/coco2017_fasterrcnn_dinov3/eval \
    --eval bbox \
    --cfg-options model.backbone.adapter_mode=official_adapter \
    | tee work_dirs/logs/coco2017_fasterrcnn_dinov3_eval.log

# run_experiment_eval examples
# Compared with plain detection/test.py, this path saves richer summary json files.
# Detection summary includes mAP, AP30, AP50 and mAR with bootstrap confidence intervals.
# Segmentation summary includes mAP, AP50, AJI and Dice with bootstrap confidence intervals.
# Bootstrap can be expensive. Use a smaller resample count for a quick sanity check,
# then increase it (for example to 1000) when you need more stable final numbers.
# Results will be written under <experiment_dir>/eval/.

# detection metrics + bootstrap
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python detection/run_experiment_eval.py \
    --task detection \
    --detection-exp work_dirs/coco2017_fasterrcnn_dinov3 \
    --detection-config my-configs/coco2017_fasterrcnn_dinov3.py \
    --checkpoint-select latest \
    --bootstrap-resamples 200 \
    --bootstrap-seed 42 \
    --python-executable python \
    --cuda-visible-devices "$CUDA_DEVICE" \
    | tee work_dirs/logs/coco2017_fasterrcnn_dinov3_bootstrap_eval.log

# segmentation metrics + bootstrap
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python detection/run_experiment_eval.py \
    --task segmentation \
    --segmentation-exp work_dirs/coco2017_maskrcnn_dinov3 \
    --segmentation-config my-configs/coco2017_maskrcnn_dinov3.py \
    --checkpoint-select latest \
    --bootstrap-resamples 200 \
    --bootstrap-seed 42 \
    --python-executable python \
    --cuda-visible-devices "$CUDA_DEVICE" \
    | tee work_dirs/logs/coco2017_maskrcnn_dinov3_bootstrap_eval.log
