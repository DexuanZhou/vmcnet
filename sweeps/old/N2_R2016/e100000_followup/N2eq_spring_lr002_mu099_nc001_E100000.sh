#!/bin/bash
#SBATCH --job-name=N2eq-spring-l002-e100k
#SBATCH --account=def-ortner
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
export SYSTEM=N2eq RUN_NAME=N2eq_spring_lr002_mu099_nc001_E100000
export SOURCE_DIR=/scratch/dexuan1/runs/e100000_followup/N2eq/N2eq_kfac_pre5000 SOURCE_EPOCH=5000 RESUME_OPTIMIZER=False
export LEARNING_RATE=0.002 ION_POS='((0.0,0.0,-1.008),(0.0,0.0,1.008))' ION_CHARGES='(7.0,7.0)' NELEC='(7,7)'
source /scratch/dexuan1/vmcnet/sweeps/e100000_followup/run_spring_100k.sh
