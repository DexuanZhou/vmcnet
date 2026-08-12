#!/bin/bash
#SBATCH --job-name=N2-krylov-offline
#SBATCH --account=def-ortner
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# This audit materializes the full parameter-by-walker score operator and its
# centering temporary; the 20-GiB pilot OOMed, so retain a 40-GiB slice.
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail
REPO=/scratch/dexuan1/vmcnet
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311
OUTPUT=/scratch/dexuan1/runs/N2_krylov_offline_20260809
[[ ! -e "${OUTPUT}" ]] || exit 2

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export PYTHONPATH="${REPO}"
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p /scratch/dexuan1/runs/logs
cd "${REPO}"
python experiments/N2_krylov_offline_20260809/audit.py
