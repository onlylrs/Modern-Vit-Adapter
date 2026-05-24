#!/bin/bash
#SBATCH --job-name=ade20k-kaa-table1
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
PARTITION_HINT="${SLURM_JOB_PARTITION:-gpu-rtx4090d}"
VARIANTS="${VARIANTS:-head adapter fusion additive product align}"

cd "$ROOT_DIR"
for VARIANT in $VARIANTS; do
  echo "Submitting ADE20K KAA Mask2Former variant: $VARIANT"
  sbatch --partition="$PARTITION_HINT" --export=ALL,VARIANT="$VARIANT" segmentation/slurm_ade20k_kaa_mask2former.sh
done
