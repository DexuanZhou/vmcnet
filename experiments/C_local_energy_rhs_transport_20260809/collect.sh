#!/bin/bash
#SBATCH --job-name=C-ELrhs-isocollect
#SBATCH --account=def-ortner
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail
ROOT=/scratch/dexuan1/runs/C_local_energy_rhs_transport_x64isolated_20260809
SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_rhs_transport_x64isolated_20260809
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export PYTHONPATH="${SNAPSHOT}"
python "${SNAPSHOT}/experiments/C_local_energy_rhs_transport_20260809/collect.py" --root "${ROOT}"
