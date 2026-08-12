#!/bin/bash
#SBATCH --job-name=C-wssr-center
#SBATCH --account=def-ortner
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
export SYSTEM=C RUN_NAME=wssr_center
export ION_POS='((0.0,0.0,0.0),)'
export PRE_DIR=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1 PRE_EPOCH=1000
export LEARNING_RATE=0.002 ETA=0.8 RANK=1600 STORAGE_RANK=1600 WORKING_RANK=1600 MAXITER_WARM=2
export SPECTRAL_REG=hard_floor
source /scratch/dexuan1/vmcnet/sweeps/templates/wssr_run.sh
