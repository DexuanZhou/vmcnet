#!/bin/bash
#SBATCH --job-name=C-wssr100k-validate
#SBATCH --account=def-ortner
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export JAX_PLATFORMS=cpu
export WANDB_MODE=disabled
cd /scratch/dexuan1/vmcnet
python experiments/C_wssr_official_matched_E100000/validate_checkpoint.py
