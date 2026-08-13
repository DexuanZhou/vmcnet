#!/bin/bash
#SBATCH --job-name=C-grass-unit
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1

set -euo pipefail
REPOSITORY="${VMCNET_REPOSITORY:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ -n "${VMCNET_MODULE_SETUP:-}" ]]; then
  eval "${VMCNET_MODULE_SETUP}"
fi
if [[ -n "${VMCNET_ENV_ACTIVATE:-}" ]]; then
  source "${VMCNET_ENV_ACTIVATE}"
fi
export PYTHONPATH="${REPOSITORY}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "${REPOSITORY}"
python -m pytest -q tests/units/updates/test_wssr.py \
  -k 'procrustes_grassmann or grassmann_ritz'
