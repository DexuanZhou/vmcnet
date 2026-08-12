#!/bin/bash
#SBATCH --job-name=C-euclid-confound
#SBATCH --account=def-ortner
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407,fc10405,fc10515
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err
#SBATCH --array=0-1

set -euo pipefail

ROOT=/scratch/dexuan1/runs/C_euclidean_constraint_confound_audit_20260809
REPO=/scratch/dexuan1/vmcnet
BATCHES=5
DIRECTION_SOURCE=${DIRECTION_SOURCE:-stored}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${ROOT}" /scratch/dexuan1/runs/logs
cd "${REPO}"

if [[ "${SLURM_ARRAY_TASK_ID}" == 0 ]]; then
  METHOD=spring
  CONFIG_RUN=/scratch/dexuan1/runs/C_spring_official_kfac1000_E100000_replicate2
  CHECKPOINTS=(
    "${CONFIG_RUN}/checkpoints/5000.npz"
    "${CONFIG_RUN}/checkpoints/10000.npz"
    "${CONFIG_RUN}/checkpoints/20000.npz"
    "${CONFIG_RUN}/checkpoints/50000.npz"
    "${CONFIG_RUN}/checkpoints/100000.npz"
  )
else
  METHOD=wssr
  CONFIG_RUN=/scratch/dexuan1/runs/C_wssr_rank1600_eta03_lr004_E100000/lr0p04_eta0p3
  FIRST=${CONFIG_RUN}/checkpoints
  RESUME=/scratch/dexuan1/runs/C_wssr_rank1600_eta03_lr004_E100000_resume1/checkpoints
  CHECKPOINTS=(
    "${FIRST}/5000.npz"
    "${FIRST}/10000.npz"
    "${FIRST}/20000.npz"
    "${RESUME}/50000.npz"
    "${RESUME}/100000.npz"
  )
fi
if [[ -n "${OUTPUT_PREFIX}" ]]; then
  OUT=${ROOT}/${OUTPUT_PREFIX}_${METHOD}
else
  OUT=${ROOT}/${METHOD}
fi

[[ ! -e "${OUT}" ]] || { echo "refusing to overwrite ${OUT}" >&2; exit 2; }
mkdir -p "${OUT}"
git rev-parse HEAD > "${OUT}/git_commit.txt"
git status --short > "${OUT}/git_status_short.txt"
scontrol show job "${SLURM_JOB_ID}" > "${OUT}/slurm_job.txt"

ARGS=()
for checkpoint in "${CHECKPOINTS[@]}"; do
  [[ -f "${checkpoint}" ]] || { echo "missing ${checkpoint}" >&2; exit 3; }
  ARGS+=(--checkpoint "${checkpoint}")
done
python experiments/C_euclidean_constraint_confound_audit_20260809/audit.py \
  --method "${METHOD}" --config-run "${CONFIG_RUN}" --batches "${BATCHES}" \
  --direction-source "${DIRECTION_SOURCE}" "${ARGS[@]}" \
  --output "${OUT}/audit.json"
test -s "${OUT}/audit.json"
