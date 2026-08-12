#!/bin/bash
#SBATCH --job-name=C-wssr100k-C-ssi40
#SBATCH --account=def-ortner
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

export SLURM_ARRAY_TASK_ID=2
export RUN_SUFFIX=_ssi40
export UPDATE_DIAGNOSTICS=False
export EXACT_FIRST=False
export SVD_INITIAL=40
export VARIANT_OVERRIDE=C_rank800_hard

exec bash \
  /scratch/dexuan1/vmcnet/experiments/C_wssr_official_matched_E100000/train_array.sh
