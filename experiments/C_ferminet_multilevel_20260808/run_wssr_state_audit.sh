#!/bin/bash
#SBATCH --job-name=C-WSSR-ML-audit
#SBATCH --account=def-ortner
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export JAX_PLATFORMS=cpu
export PYTHONPATH=/scratch/dexuan1/vmcnet

cd /scratch/dexuan1/vmcnet
python experiments/C_ferminet_multilevel_20260808/audit_wssr_five_level_state.py
