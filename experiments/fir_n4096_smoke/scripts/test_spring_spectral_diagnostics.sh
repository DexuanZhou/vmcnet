#!/bin/bash
#SBATCH --job-name=spring-spec-unit
#SBATCH --account=def-ortner
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
set -euo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
cd /scratch/dexuan1/vmcnet
export PYTHONUNBUFFERED=1 JAX_ENABLE_X64=True
date
python -u -c 'import jax; print("jax", jax.__version__, jax.devices(), flush=True)'
pytest -vv -s tests/units/updates/test_spring_history.py
date
