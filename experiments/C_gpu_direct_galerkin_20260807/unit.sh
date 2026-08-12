#!/bin/bash
#SBATCH --job-name=C-gpu-galerkin-unit
#SBATCH --account=def-ortner
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

REPO=/scratch/dexuan1/vmcnet
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "${REPO}"
pytest -q tests/units/updates/test_wssr.py \
  -k 'device_cholesky_galerkin or galerkin_rejects_unknown or cached_and_explicit_galerkin or galerkin_large_lambda or full_current' \
  --disable-warnings --maxfail=1
