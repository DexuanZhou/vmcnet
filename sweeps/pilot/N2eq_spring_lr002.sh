#!/bin/bash
#SBATCH --job-name=N2eq-spring-lr002
#SBATCH --account=def-ortner
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
export SYSTEM=N2eq RUN_NAME=spring_lr002_mu099_nc001 LEARNING_RATE=0.002
export MU=0.99 NORM_CONSTRAINT=0.001
export ION_POS='((0.0,0.0,-1.034),(0.0,0.0,1.034))'
export PRE_DIR=/scratch/dexuan1/runs/pilot/N2eq/N2eq_kfac_pre2000 PRE_EPOCH=2000
source /scratch/dexuan1/vmcnet/sweeps/templates/spring_ref.sh
