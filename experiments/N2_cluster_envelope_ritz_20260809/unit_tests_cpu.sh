#!/bin/bash
#SBATCH --job-name=env-ritz-unit-cpu
#SBATCH --account=def-ortner
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export PYTHONPATH=/scratch/dexuan1/vmcnet
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
cd /scratch/dexuan1/vmcnet
pytest -q \
  tests/units/updates/test_wssr_experimental.py \
  tests/units/updates/test_wssr.py \
  -k cluster_envelope
