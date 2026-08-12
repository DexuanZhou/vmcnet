#!/bin/bash
#SBATCH --account=def-ortner
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

export RUN_ROOT=/scratch/dexuan1/runs/e20000_c_focus NEPOCHS=20000 CHECKPOINT_EVERY=2000
export SYSTEM=C RANK=1600 STORAGE_RANK=1600 WORKING_RANK=1600
export SPECTRAL_REG=tikhonov
export ION_POS='((0.0,0.0,0.0),)'
export PRE_DIR=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1 PRE_EPOCH=1000
source /scratch/dexuan1/vmcnet/sweeps/templates/wssr_run.sh
