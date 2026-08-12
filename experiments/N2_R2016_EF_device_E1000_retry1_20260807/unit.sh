#!/bin/bash
#SBATCH --job-name=N2-EF-unit
#SBATCH --account=def-ortner
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /scratch/dexuan1/vmcnet
pytest -q tests/units/updates/test_wssr.py \
  -k 'error_feedback or device_cholesky_galerkin or applied_residual_capture_metrics or default_config_contains_wssr_warm_svd_right' \
  --disable-warnings --maxfail=1
