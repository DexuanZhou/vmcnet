#!/bin/bash
#SBATCH --job-name=C-wssr-rel-collect
#SBATCH --account=def-ortner
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2025a
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
cd /scratch/dexuan1/vmcnet
python experiments/C_rank1600_reliability_indicator_replay/collect.py
