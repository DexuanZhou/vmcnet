#!/bin/bash
#SBATCH --job-name=wssr-burst-tests
#SBATCH --account=def-ortner
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2025a
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
python -m pip install --quiet pytest-mock
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=8"
export WANDB_DISABLED=true
export WANDB_MODE=disabled
cd /scratch/dexuan1/vmcnet

echo "hostname=$(hostname)"
python - <<'PY'
import platform
import jax
import jaxlib
print("python", platform.python_version())
print("jax", jax.__version__)
print("jaxlib", jaxlib.__version__)
print("devices", jax.devices())
assert all(device.platform == "cpu" for device in jax.devices())
PY

pytest -ra --durations=25 \
  tests/units/train/test_burst_diagnostics.py \
  tests/units/updates/test_wssr_experimental.py \
  tests/units/updates/test_wssr.py \
  tests/units/train/test_parse_config_flags.py
