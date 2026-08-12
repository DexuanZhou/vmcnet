#!/bin/bash
#SBATCH --job-name=C-gpu-galerkin-time
#SBATCH --account=def-ortner
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-1
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

HERE=/scratch/dexuan1/vmcnet/experiments/C_gpu_direct_galerkin_20260807
RUNNER=/scratch/dexuan1/vmcnet/experiments/C_zero_cost_triage_20260804/task2_largeN_timing/run_array.sh
ROOT=/scratch/dexuan1/runs/C_gpu_direct_galerkin_20260807

case "${SLURM_ARRAY_TASK_ID}" in
  0) N=1000 ;;
  1) N=4096 ;;
  *) echo "unexpected task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

for BACKEND in host_fp64 device_cholesky; do
  TIMING_ROOT="${ROOT}/${BACKEND}" \
  TIMING_MANIFEST="${HERE}/manifest_n${N}.tsv" \
  TIMING_EPOCHS=125 \
  TIMING_PROFILE_START=121 \
  GALERKIN_SOLVE_BACKEND="${BACKEND}" \
  SLURM_ARRAY_TASK_ID=0 \
    bash "${RUNNER}"
done
