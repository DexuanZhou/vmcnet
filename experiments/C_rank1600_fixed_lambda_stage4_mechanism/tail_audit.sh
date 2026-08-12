#!/bin/bash
#SBATCH --job-name=C-lam4-tail
#SBATCH --account=def-ortner
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /scratch/dexuan1/vmcnet
python experiments/C_rank1600_fixed_lambda_stage4_mechanism/tail_audit.py
