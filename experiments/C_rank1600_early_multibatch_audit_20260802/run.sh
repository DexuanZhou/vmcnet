#!/bin/bash
#SBATCH --job-name=C-early-multibatch
#SBATCH --account=def-ortner
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

: "${SOURCE_RUN:?set SOURCE_RUN}"
: "${SOURCE_EPOCH:?set SOURCE_EPOCH}"
: "${OUT:?set OUT}"
REPLICATES=${REPLICATES:-6}

[[ ! -e "${OUT}" ]] || { echo "refusing to overwrite ${OUT}" >&2; exit 2; }
mkdir -p "${OUT}" /scratch/dexuan1/runs/logs

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

cd /scratch/dexuan1/vmcnet
git rev-parse HEAD > "${OUT}/git_commit.txt"
git status --short > "${OUT}/git_status_short.txt"
scontrol show job "${SLURM_JOB_ID}" > "${OUT}/slurm_job.txt"
python experiments/C_rank1600_early_multibatch_audit_20260802/audit.py \
  --run "${SOURCE_RUN}" --epoch "${SOURCE_EPOCH}" \
  --replicates "${REPLICATES}" --output "${OUT}/audit.json"
python experiments/C_rank1600_early_multibatch_audit_20260802/collect.py \
  --root "${OUT}"
test -s "${OUT}/summary.json"
