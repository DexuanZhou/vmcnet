#!/bin/bash
#SBATCH --job-name=C-exactfirst-release
#SBATCH --account=def-ortner
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
set -euo pipefail
module --force purge; module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
cd /scratch/dexuan1/vmcnet
python experiments/C_exact_first_complement/release_after_subspace.py
