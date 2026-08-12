#!/bin/bash
#SBATCH --job-name=C-gtransport-audit
#SBATCH --account=def-ortner
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-1%2
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

SOURCE=/scratch/dexuan1/runs/C_wssr_adaptive_etaS_E5000_20260803/train/exponential_tau1000/seed0
ROOT=/scratch/dexuan1/runs/C_wssr_gradient_transport_etaG02_20260803/audit
case "${SLURM_ARRAY_TASK_ID}" in
  0) EPOCH=1000 ;;
  1) EPOCH=3000 ;;
  *) echo "unexpected task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac
OUTPUT=${ROOT}/epoch${EPOCH}/audit.json
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "$(dirname "${OUTPUT}")" /scratch/dexuan1/runs/logs
cd /scratch/dexuan1/vmcnet
python experiments/C_wssr_gradient_transport_etaG02_20260803/audit.py \
  --run "${SOURCE}" --epoch "${EPOCH}" --transitions 8 --output "${OUTPUT}"
test -s "${OUTPUT}"
