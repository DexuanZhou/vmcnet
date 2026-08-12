#!/bin/bash
#SBATCH --job-name=N2-ltp-release
#SBATCH --account=def-ortner
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
set -euo pipefail
cd /scratch/dexuan1/vmcnet
python experiments/N2_rank400_longtime_pilot/release_stage2.py
