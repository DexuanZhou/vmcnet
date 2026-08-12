#!/bin/bash
#SBATCH --job-name=wssr-tests
#SBATCH --account=def-ortner
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load python/3.11.5

source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate

pip install pytest-timeout

cd /scratch/dexuan1/vmcnet
python -m pytest tests/units/updates/test_wssr.py -q --timeout=300
