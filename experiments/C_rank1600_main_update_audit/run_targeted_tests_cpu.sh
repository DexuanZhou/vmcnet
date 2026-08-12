#!/bin/bash
#SBATCH --job-name=C-wssr-split-tests
#SBATCH --account=def-ortner
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
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

python -m py_compile \
  vmcnet/updates/wssr.py \
  vmcnet/train/default_config.py \
  experiments/C_rank1600_main_update_audit/audit_checkpoint_x64.py
pytest -q tests/units/updates/test_wssr.py \
  -k 'default_config_contains_wssr_warm_svd_right or fixed_tikhonov_lambda_is_independent or relative_cutoff_is_independent or tikhonov_complement_weight_matches_formula or hard_floor_default_matches_previous_formula'
