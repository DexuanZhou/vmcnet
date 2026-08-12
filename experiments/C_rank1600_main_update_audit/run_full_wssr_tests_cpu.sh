#!/bin/bash
#SBATCH --job-name=C-wssr-split-full
#SBATCH --account=def-ortner
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2025a
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=8"
cd /scratch/dexuan1/vmcnet

pytest -q tests/units/updates/test_wssr.py
