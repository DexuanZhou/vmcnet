#!/bin/bash
#SBATCH --job-name=C-ELrhs-x64isolated
#SBATCH --account=def-ortner
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-3%2
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/scratch/dexuan1/runs/C_local_energy_rhs_transport_x64isolated_20260809
SOURCE=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1
SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_rhs_transport_x64isolated_20260809

case "${SLURM_ARRAY_TASK_ID}" in
  0) METHOD=wssr; REPLICATE=0 ;;
  1) METHOD=wssr; REPLICATE=1 ;;
  2) METHOD=spring; REPLICATE=0 ;;
  3) METHOD=spring; REPLICATE=1 ;;
  *) exit 2 ;;
esac

OUTPUT=${ROOT}/results/${METHOD}_rep${REPLICATE}.json
[[ -f "${SOURCE}/checkpoints/1000.npz" ]]
[[ -d "${SNAPSHOT}/vmcnet" ]]
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export PYTHONPATH="${SNAPSHOT}"
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true

mkdir -p "$(dirname "${OUTPUT}")" "${ROOT}/metadata" /scratch/dexuan1/runs/logs
cd "${SNAPSHOT}"
{
  hostname
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  git rev-parse HEAD
  python - <<'PY'
import jax
print("jax", jax.__version__)
print("backend", jax.default_backend())
print("x64", jax.config.x64_enabled)
print("devices", jax.devices())
assert jax.default_backend() == "gpu"
assert jax.config.x64_enabled
PY
} > "${ROOT}/metadata/${METHOD}_rep${REPLICATE}.txt"

python experiments/C_local_energy_rhs_transport_20260809/audit.py \
  --method "${METHOD}" --replicate "${REPLICATE}" \
  --source "${SOURCE}" --microbatch 10 --output "${OUTPUT}"

test -s "${OUTPUT}"
