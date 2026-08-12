#!/bin/bash
#SBATCH --job-name=C-wssr-e02-l002-s2-fresh-e100k
#SBATCH --account=def-ortner
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
export SYSTEM=C RUN_NAME=C_wssr_eta02_lr0002_tikhonov_svd8_2_fresh_E100000_spin42
export SOURCE_DIR=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1 SOURCE_EPOCH=1000 RESUME_OPTIMIZER=False
export LEARNING_RATE=0.002 ETA=0.2 NORM_CONSTRAINT=0.001 MAXITER_WARM=2
export ION_POS='((0.0,0.0,0.0),)' ION_CHARGES='(6.0,)' NELEC='(4,2)'
source /scratch/dexuan1/vmcnet/sweeps/e100000_followup/run_wssr_100k.sh
