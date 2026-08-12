#!/bin/bash
#SBATCH --job-name=C-gpu-galerkin-micro
#SBATCH --account=def-ortner
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MICROBENCH_OUT=/scratch/dexuan1/runs/C_gpu_direct_galerkin_20260807/microbench
mkdir -p "${MICROBENCH_OUT}" /scratch/dexuan1/runs/logs
cd /scratch/dexuan1/vmcnet
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits \
  > "${MICROBENCH_OUT}/gpu_identity.csv"
python experiments/C_gpu_direct_galerkin_20260807/microbench.py
