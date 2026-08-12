#!/bin/bash
#SBATCH --job-name=C-r1600-eta-collect
#SBATCH --account=def-ortner
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail
cd /scratch/dexuan1/vmcnet
python experiments/C_rank1600_eta_memory_E5000_20260802/collect.py
