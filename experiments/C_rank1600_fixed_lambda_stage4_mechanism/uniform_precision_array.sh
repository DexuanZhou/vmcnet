#!/bin/bash
#SBATCH --job-name=C-uniform-x64
#SBATCH --account=def-ortner
#SBATCH --time=00:06:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-1%2
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail
case "${SLURM_ARRAY_TASK_ID}" in
  0) LABEL=legacy_epoch102000 ;;
  1) LABEL=fixed_lambda_1e-3_epoch102000 ;;
  *) exit 2 ;;
esac

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /scratch/dexuan1/vmcnet
python experiments/C_rank1600_fixed_lambda_stage4_mechanism/uniform_precision_audit.py \
  --label "${LABEL}"
