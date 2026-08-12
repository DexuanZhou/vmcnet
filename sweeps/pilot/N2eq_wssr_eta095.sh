#!/bin/bash
#SBATCH --job-name=N2eq-wssr-eta095
#SBATCH --account=def-ortner
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
export SYSTEM=N2eq RUN_NAME=wssr_eta095
export ION_POS='((0.0,0.0,-1.034),(0.0,0.0,1.034))'
export PRE_DIR=/scratch/dexuan1/runs/pilot/N2eq/N2eq_kfac_pre2000 PRE_EPOCH=2000
export LEARNING_RATE=0.002 ETA=0.95 RANK=1600 STORAGE_RANK=1600 WORKING_RANK=1600 MAXITER_WARM=2
export SPECTRAL_REG=hard_floor
source /scratch/dexuan1/vmcnet/sweeps/templates/wssr_run.sh
