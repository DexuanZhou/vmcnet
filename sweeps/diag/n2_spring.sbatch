#!/bin/bash
#SBATCH --account=def-ortner
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

export RUN_ROOT=/scratch/dexuan1/runs/diag NEPOCHS=150 CHECKPOINT_EVERY=50
export SYSTEM=N2eq MU=0.99
export ION_POS='((0.0,0.0,-1.034),(0.0,0.0,1.034))'
export PRE_DIR=/scratch/dexuan1/runs/pilot/N2eq/N2eq_kfac_pre2000 PRE_EPOCH=2000
source /scratch/dexuan1/vmcnet/sweeps/templates/spring_ref.sh
