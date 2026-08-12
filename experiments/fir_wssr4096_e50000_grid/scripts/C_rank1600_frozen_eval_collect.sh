#!/bin/bash
#SBATCH --job-name=C-r1600-eval-collect
#SBATCH --account=def-ortner
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
set -euo pipefail
cd /scratch/dexuan1/vmcnet
module --force purge;module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export PYTHONPATH=/scratch/dexuan1/vmcnet${PYTHONPATH:+:${PYTHONPATH}}
python experiments/fir_wssr4096_e50000_grid/collect_C_rank1600_frozen_eval.py
