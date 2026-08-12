#!/bin/bash
#SBATCH --job-name=N2-minsr4096-collect
#SBATCH --account=def-ortner
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
set -euo pipefail
module --force purge; module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
cd /scratch/dexuan1/vmcnet; python experiments/fir_n4096_smoke/collect_n2_minsr_mu0.py
