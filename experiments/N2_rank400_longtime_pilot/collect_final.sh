#!/bin/bash
#SBATCH --job-name=N2-ltp-final
#SBATCH --account=def-ortner
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
set -euo pipefail
cd /scratch/dexuan1/vmcnet
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate || true
python experiments/N2_rank400_longtime_pilot/collect_final.py
