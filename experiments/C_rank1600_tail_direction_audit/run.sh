#!/bin/bash
#SBATCH --job-name=C-tail-audit
#SBATCH --account=def-ortner
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

ROOT=/scratch/dexuan1/runs/C_wssr_rank1600_tail_direction_audit_20b
RESULT=${ROOT}/audit.json
REPORT=${ROOT}/report.md
[[ ! -e "${ROOT}" ]] || { echo "refusing to overwrite ${ROOT}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${ROOT}" /scratch/dexuan1/runs/logs
cd /scratch/dexuan1/vmcnet
git rev-parse HEAD > "${ROOT}/git_commit.txt"
git status --short > "${ROOT}/git_status_short.txt"
scontrol show job "${SLURM_JOB_ID}" > "${ROOT}/slurm_job.txt"

python experiments/C_rank1600_tail_direction_audit/audit.py \
  --run /scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage2_E2000 \
  --epoch 102000 --batches 20 --output "${RESULT}"
python experiments/C_rank1600_tail_direction_audit/collect.py \
  --input "${RESULT}" --output "${REPORT}"
test -s "${RESULT}"
test -s "${REPORT}"
