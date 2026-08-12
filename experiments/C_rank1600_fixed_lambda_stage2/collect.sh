#!/bin/bash
#SBATCH --job-name=C-r1600-lam2-collect
#SBATCH --account=def-ortner
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
cd /scratch/dexuan1/vmcnet
python experiments/C_rank1600_fixed_lambda_stage2/collect.py
