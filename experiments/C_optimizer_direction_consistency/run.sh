#!/bin/bash
#SBATCH --job-name=C-dir-consistency
#SBATCH --account=def-ortner
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail
: "${METHOD:?set METHOD}"
: "${SOURCE_RUN:?set SOURCE_RUN}"
: "${SOURCE_EPOCH:?set SOURCE_EPOCH}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
BATCHES=${BATCHES:-30}
REPLICATE_KEYS=${REPLICATE_KEYS:-1}
[[ ! -e "${OUTPUT_ROOT}" ]] || {
  echo "refusing to overwrite ${OUTPUT_ROOT}" >&2
  exit 2
}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${OUTPUT_ROOT}" /scratch/dexuan1/runs/logs
cd /scratch/dexuan1/vmcnet
git rev-parse HEAD > "${OUTPUT_ROOT}/git_commit.txt"
git status --short > "${OUTPUT_ROOT}/git_status_short.txt"
scontrol show job "${SLURM_JOB_ID}" > "${OUTPUT_ROOT}/slurm_job.txt"
python experiments/C_optimizer_direction_consistency/audit.py \
  --method "${METHOD}" --run "${SOURCE_RUN}" --epoch "${SOURCE_EPOCH}" \
  --batches "${BATCHES}" --replicate-keys "${REPLICATE_KEYS}" \
  --output "${OUTPUT_ROOT}/audit.json"
test -s "${OUTPUT_ROOT}/audit.json"
