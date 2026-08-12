#!/bin/bash
#SBATCH --job-name=C-e20k-focus-collect
#SBATCH --account=def-ortner
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
set -euo pipefail
cd /scratch/dexuan1/vmcnet/sweeps
python3 collect_results.py --manifest e20000_c_focus/manifest.csv --out-dir results/e20000_c_focus
python3 e20000_c_focus/finalize.py
