#!/bin/bash
#SBATCH --job-name=N2-rank-svd-e100k
#SBATCH --account=def-ortner
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-5%2
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err
set -euo pipefail

COMBOS=(200:2 400:1 400:2 400:4 800:2 1600:2)
IFS=: read -r RANK MAXITER_WARM <<< "${COMBOS[${SLURM_ARRAY_TASK_ID}]}"
export SYSTEM=N2eq_R2068 RUN_NAME="N2eq_R2068_wssr_eta03_lr0002_rank${RANK}_svdwarm${MAXITER_WARM}_E100000"
export SOURCE_DIR=/scratch/dexuan1/runs/kfac_pre/N2eq_R2068_kfac_pre5000 SOURCE_EPOCH=5000
export LEARNING_RATE=0.002 ETA=0.3 RANK MAXITER_WARM HOST_MEM_GB=80
export ION_POS='((0.0,0.0,-1.034),(0.0,0.0,1.034))' ION_CHARGES='(7.0,7.0)' NELEC='(7,7)'
source /scratch/dexuan1/vmcnet/sweeps/rank_svd_benchmark/run_benchmark.sh
