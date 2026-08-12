#!/bin/bash
#SBATCH --job-name=C-lazySSI-unit-cpu
#SBATCH --account=def-ortner
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_C_lazy_ssi_20260808
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export PYTHONPATH="${SNAPSHOT}"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=8
cd "${SNAPSHOT}"
python -m pytest -q tests/units/updates/test_wssr.py \
  -k 'lazy_fixed_basis or delayed_refresh_reuses or full_current_batch_galerkin or device_galerkin'
