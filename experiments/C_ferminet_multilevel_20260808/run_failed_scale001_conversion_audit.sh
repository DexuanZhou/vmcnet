#!/bin/bash
#SBATCH --job-name=C-ML001-audit
#SBATCH --account=def-ortner
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export PYTHONPATH=/scratch/dexuan1/vmcnet_snapshot_C_arch_multilevel_fivelevel_20260808
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /scratch/dexuan1/vmcnet_snapshot_C_arch_multilevel_fivelevel_20260808
python /scratch/dexuan1/vmcnet/experiments/C_ferminet_multilevel_20260808/audit_failed_scale001_conversion.py
