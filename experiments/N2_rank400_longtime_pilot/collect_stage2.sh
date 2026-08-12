#!/bin/bash
#SBATCH --job-name=N2-ltp-col2
#SBATCH --account=def-ortner
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
set -euo pipefail
cd /scratch/dexuan1/vmcnet
python experiments/N2_rank400_longtime_pilot/collect_stage2.py
