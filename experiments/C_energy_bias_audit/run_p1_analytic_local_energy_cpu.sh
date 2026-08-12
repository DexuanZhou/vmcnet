#!/bin/bash
#SBATCH --job-name=C-EL-analytic
#SBATCH --account=def-ortner
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2025a
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"
cd /scratch/dexuan1/vmcnet

python experiments/C_energy_bias_audit/p1_analytic_local_energy.py \
  --npoints 10000 \
  --seed 20260723 \
  --dtype float32 \
  --output experiments/C_energy_bias_audit/results/p1_analytic_local_energy_float32.json

python experiments/C_energy_bias_audit/p1_analytic_local_energy.py \
  --npoints 10000 \
  --seed 20260723 \
  --dtype float64 \
  --output experiments/C_energy_bias_audit/results/p1_analytic_local_energy_float64.json

python experiments/C_energy_bias_audit/merge_p1_results.py \
  --float32 experiments/C_energy_bias_audit/results/p1_analytic_local_energy_float32.json \
  --float64 experiments/C_energy_bias_audit/results/p1_analytic_local_energy_float64.json \
  --output experiments/C_energy_bias_audit/results/p1_analytic_local_energy.json
