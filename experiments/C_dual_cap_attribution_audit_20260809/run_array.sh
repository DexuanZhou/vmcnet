#!/bin/bash
#SBATCH --job-name=C-cap-attr
#SBATCH --account=def-ortner
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-1%2
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

SEED=${SLURM_ARRAY_TASK_ID}
REPO=/scratch/dexuan1/vmcnet
SOURCE=/scratch/dexuan1/runs/C_function_space_matched_E5000_20260809/train/wssr_rank1600/seed${SEED}
ROOT=${ROOT:-/scratch/dexuan1/runs/C_dual_cap_attribution_audit_20260809}
TRANSITIONS=${TRANSITIONS:-20}
PROBES=${PROBES:-16}
OUTPUT=${ROOT}/seed${SEED}
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=true

mkdir -p "${OUTPUT}" /scratch/dexuan1/runs/logs
cd "${REPO}"
git rev-parse HEAD > "${OUTPUT}/git_commit.txt"
git status --short > "${OUTPUT}/git_status_short.txt"
scontrol show job "${SLURM_JOB_ID}" > "${OUTPUT}/slurm_job.txt"

python experiments/C_dual_cap_attribution_audit_20260809/audit.py \
  --run "${SOURCE}" \
  --epoch 5000 \
  --transitions "${TRANSITIONS}" \
  --probes "${PROBES}" \
  --beta-e 0.15 \
  --beta-f 0.15 \
  --output "${OUTPUT}/audit"

test -s "${OUTPUT}/audit/metrics.csv"
test -s "${OUTPUT}/audit/summary.json"
