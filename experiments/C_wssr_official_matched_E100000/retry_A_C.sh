#!/bin/bash
#SBATCH --job-name=C-wssr100k-retry-AC
#SBATCH --account=def-ortner
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

case "${SLURM_ARRAY_TASK_ID}" in
  0) ORIGINAL_TASK=0 ;;
  1) ORIGINAL_TASK=2 ;;
  *) exit 2 ;;
esac

export SLURM_ARRAY_TASK_ID="${ORIGINAL_TASK}"
export RUN_SUFFIX=_retry1
export UPDATE_DIAGNOSTICS=False
exec bash \
  /scratch/dexuan1/vmcnet/experiments/C_wssr_official_matched_E100000/train_array.sh
