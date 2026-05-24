#!/bin/bash
#SBATCH --job-name=ade20k-kaa-m2f
#SBATCH --account=smartlab2021
#SBATCH --partition=gpu-rtx4090d
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --time=72:00:00
#SBATCH --output=work_dirs/logs/slurm_logs/%x-%j.out
#SBATCH --error=work_dirs/logs/slurm_logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-/home/rliuar/bin/micromamba}"
MAMBA_ENV_NAME="${MAMBA_ENV_NAME:-torch29}"
GPUS="${GPUS:-4}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PORT="${PORT:-29501}"
SEED="${SEED:-500272652}"
VARIANT="${VARIANT:-product}"

case "$VARIANT" in
  head)
    CONFIG="segmentation/configs/ade20k/mask2former_dinov3_head_base_640_160k_ade20k_frozen.py" ;;
  adapter)
    CONFIG="segmentation/configs/ade20k/mask2former_dinov3_adapter_base_640_160k_ade20k_frozen.py" ;;
  fusion)
    CONFIG="segmentation/configs/ade20k/kaa_mask2former_dinov3_base_640_160k_ade20k_frozen_fusion.py" ;;
  additive)
    CONFIG="segmentation/configs/ade20k/kaa_mask2former_dinov3_base_640_160k_ade20k_frozen_additive.py" ;;
  product)
    CONFIG="segmentation/configs/ade20k/kaa_mask2former_dinov3_base_640_160k_ade20k_frozen_product.py" ;;
  align)
    CONFIG="segmentation/configs/ade20k/kaa_mask2former_dinov3_base_640_160k_ade20k_frozen_align.py" ;;
  *)
    echo "Unknown VARIANT=$VARIANT. Use head|adapter|fusion|additive|product|align." >&2
    exit 2 ;;
esac

WORK_DIR="work_dirs/ade20k/$(basename "$CONFIG" .py)"
TRAIN_LOG="work_dirs/logs/$(basename "$CONFIG" .py)-${SLURM_JOB_ID:-local}.log"

cd "$ROOT_DIR"
source "$ROOT_DIR/env.sh"
mkdir -p "$WORK_DIR" work_dirs/logs work_dirs/logs/slurm_logs

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" PORT="$PORT" \
"$MICROMAMBA_BIN" run -n "$MAMBA_ENV_NAME" \
  bash segmentation/dist_train.sh \
    "$CONFIG" \
    "$GPUS" \
    --work-dir "$WORK_DIR" \
    --seed "$SEED" \
    --deterministic \
  | tee "$TRAIN_LOG"
