#!/bin/bash
#SBATCH --job-name=C-WSSR-unit-local
#SBATCH --account=def-ortner_cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export XLA_FLAGS=--xla_force_host_platform_device_count=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

LOCAL_REPO="${SLURM_TMPDIR}/vmcnet-unit"
mkdir -p "${LOCAL_REPO}"
rsync -a /scratch/dexuan1/vmcnet/vmcnet "${LOCAL_REPO}/"
rsync -a /scratch/dexuan1/vmcnet/tests "${LOCAL_REPO}/"
cd "${LOCAL_REPO}"

python -m pytest -q tests/units/updates/test_wssr.py -k \
  'cached_current_action or right_ssi_returned_current_action or cached_and_explicit_galerkin or delayed_refresh_reuses or full_current_batch_galerkin or full_current_batch_large_lambda or full_current_batch_error_feedback'
