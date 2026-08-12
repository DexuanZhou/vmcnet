#!/bin/bash
#SBATCH --job-name=C-wssr-SSI40-next
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

ORIGINAL_TASK_ID="${SLURM_ARRAY_TASK_ID}"
case "${ORIGINAL_TASK_ID}" in
  0)
    # Reuse the adaptive-complement mapping, changing only rank and initialization.
    export SLURM_ARRAY_TASK_ID=1
    export RANK_OVERRIDE=800
    export VARIANT_OVERRIDE=D_rank800_adaptive_beta02
    ;;
  1)
    # Reuse the hard-complement mapping, changing rank and initialization.
    export SLURM_ARRAY_TASK_ID=2
    export RANK_OVERRIDE=1600
    export VARIANT_OVERRIDE=E_rank1600_hard
    ;;
  *)
    echo "invalid task ${ORIGINAL_TASK_ID}" >&2
    exit 2
    ;;
esac

export RUN_SUFFIX=_ssi40
export UPDATE_DIAGNOSTICS=False
export EXACT_FIRST=False
export SVD_INITIAL=40

exec bash \
  /scratch/dexuan1/vmcnet/experiments/C_wssr_official_matched_E100000/train_array.sh
