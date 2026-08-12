#!/bin/bash
#SBATCH --account=def-ortner
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

export RUN_ROOT=/scratch/dexuan1/runs/e5000 NEPOCHS=5000 CHECKPOINT_EVERY=500
export SYSTEM=C RUN_NAME=spring_lr02 LEARNING_RATE=0.02 MU=0.99 NORM_CONSTRAINT=0.001
export ION_POS='((0.0,0.0,0.0),)'
export PRE_DIR=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1 PRE_EPOCH=1000
source /scratch/dexuan1/vmcnet/sweeps/templates/spring_ref.sh
