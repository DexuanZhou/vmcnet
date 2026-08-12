#!/bin/bash
#SBATCH --job-name=C-rank-svd-e100k
#SBATCH --account=def-ortner
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-5%2
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err
set -euo pipefail

COMBOS=(200:2 400:1 400:2 400:4 800:2 1600:2)
IFS=: read -r RANK MAXITER_WARM <<< "${COMBOS[${SLURM_ARRAY_TASK_ID}]}"
export SYSTEM=C RUN_NAME="C_wssr_eta08_lr02_rank${RANK}_svdwarm${MAXITER_WARM}_E100000"
export SOURCE_DIR=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1 SOURCE_EPOCH=1000
export LEARNING_RATE=0.02 ETA=0.8 RANK MAXITER_WARM HOST_MEM_GB=48
export ION_POS='((0.0,0.0,0.0),)' ION_CHARGES='(6.0,)' NELEC='(4,2)'
source /scratch/dexuan1/vmcnet/sweeps/rank_svd_benchmark/run_benchmark.sh
