#!/bin/bash
#SBATCH --job-name=C-wssr-native-prox
#SBATCH --account=def-ortner
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gpus-per-node=h100_3g.40gb:1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
SOURCE_RUN=${SOURCE_RUN:-/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage2_E2000}
SOURCE_EPOCH=${SOURCE_EPOCH:-102000}
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
EXTRA_ARGS=()
if [[ -n "${KEY_FOLD_IN:-}" ]]; then
  EXTRA_ARGS+=(--key-fold-in "${KEY_FOLD_IN}")
fi
if [[ -n "${CONFIRMATION_GAMMA:-}" ]]; then
  EXTRA_ARGS+=(--confirmation-gamma "${CONFIRMATION_GAMMA}")
fi
python experiments/C_wssr_subspace_causal_audit/native_proximal_replay.py \
  --run "${SOURCE_RUN}" --epoch "${SOURCE_EPOCH}" \
  --output "${OUTPUT_ROOT}/audit.json" "${EXTRA_ARGS[@]}"
test -s "${OUTPUT_ROOT}/audit.json"
test -s "${OUTPUT_ROOT}/summary.md"
