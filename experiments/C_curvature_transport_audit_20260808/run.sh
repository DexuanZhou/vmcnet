#!/bin/bash
#SBATCH --job-name=C-Stransport
#SBATCH --account=def-ortner
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407,fc10404
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

SOURCE_RUN=/scratch/dexuan1/runs/C_wssr_split_eta_E5000_20260803/train/D_etaS0_etaG0/seed0
ROOT=/scratch/dexuan1/runs/C_curvature_transport_audit_20260808_retry1
OUT=${ROOT}/early_epoch1000

[[ ! -e "${OUT}" ]] || {
  echo "refusing to overwrite ${OUT}" >&2
  exit 2
}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${OUT}" /scratch/dexuan1/runs/logs
cd /scratch/dexuan1/vmcnet_snapshot_C_lazy_ssi_20260808
git -C /scratch/dexuan1/vmcnet rev-parse HEAD > "${OUT}/git_commit.txt"
git -C /scratch/dexuan1/vmcnet status --short > "${OUT}/git_status_short.txt"
scontrol show job "${SLURM_JOB_ID}" > "${OUT}/slurm_job.txt"

python /scratch/dexuan1/vmcnet/experiments/C_curvature_transport_audit_20260808/audit.py \
  --run "${SOURCE_RUN}" \
  --epoch 1000 \
  --replicates 8 \
  --output "${OUT}/audit.json"

test -s "${OUT}/audit.json"
