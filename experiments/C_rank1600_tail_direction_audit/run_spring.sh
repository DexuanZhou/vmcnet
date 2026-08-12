#!/bin/bash
#SBATCH --job-name=C-spring-tail
#SBATCH --account=def-ortner
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail
ROOT=/scratch/dexuan1/runs/C_spring_tail_response_audit_20b
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

python experiments/C_rank1600_tail_direction_audit/audit_spring.py \
  --run /scratch/dexuan1/runs/C_spring_official_kfac1000_E100000 \
  --epoch 100000 --batches 20 --output "${ROOT}/audit.json"
python experiments/C_rank1600_tail_direction_audit/compare.py \
  --wssr /scratch/dexuan1/runs/C_wssr_rank1600_tail_direction_audit_20b/audit.json \
  --spring "${ROOT}/audit.json" --output "${ROOT}/comparison.json"
test -s "${ROOT}/audit.json"
test -s "${ROOT}/comparison.json"
