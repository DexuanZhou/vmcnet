#!/bin/bash
#SBATCH --job-name=C-early-memory
#SBATCH --account=def-ortner
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-6%4
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

ROOT=${ROOT:-/scratch/dexuan1/runs/C_rank1600_early_memory_scaling_20260802}
REPLICATES=${REPLICATES:-6}

case "${SLURM_ARRAY_TASK_ID}" in
  0) ARM=single_current ;;
  1) ARM=wssr_eta03 ;;
  2) ARM=wssr_eta08 ;;
  3) ARM=wssr_eta095 ;;
  4) ARM=wssr_eta099 ;;
  5) ARM=wssr_eta099_bias_corrected ;;
  6) ARM=spring_mu099 ;;
  *) echo "unexpected array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

OUT=${ROOT}/${ARM}
[[ ! -e "${OUT}" ]] || {
  echo "refusing to overwrite ${OUT}" >&2
  exit 2
}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

mkdir -p "${OUT}" /scratch/dexuan1/runs/logs
cd /scratch/dexuan1/vmcnet
git rev-parse HEAD > "${OUT}/git_commit.txt"
git status --short > "${OUT}/git_status_short.txt"
scontrol show job "${SLURM_JOB_ID}" > "${OUT}/slurm_job.txt"
cp experiments/C_rank1600_early_memory_scaling_20260802/README.md \
  "${OUT}/preregistered_protocol.md"

python experiments/C_rank1600_early_memory_scaling_20260802/audit.py \
  --run /scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1 \
  --epoch 1000 \
  --arm "${ARM}" \
  --replicates "${REPLICATES}" \
  --output "${OUT}/audit.json"

test -s "${OUT}/audit.json"
