#!/bin/bash
#SBATCH --job-name=C-gtransport-unit
#SBATCH --account=def-ortner
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /scratch/dexuan1/vmcnet
pytest -q tests/units/updates/test_wssr.py \
  -k 'adaptive_gradient_eta_resolution or adaptive_gradient_average_integrated_schedule or adaptive_eta_resolution or adaptive_s_averaging_eta_schedules'
