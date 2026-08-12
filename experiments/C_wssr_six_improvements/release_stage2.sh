#!/bin/bash
#SBATCH --job-name=C-wssr6-release2
#SBATCH --account=def-ortner
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
set -euo pipefail
cd /scratch/dexuan1/vmcnet;python experiments/C_wssr_six_improvements/release_stage2.py
