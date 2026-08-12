#!/bin/bash
#SBATCH --job-name=C-kfac-pre1000
#SBATCH --account=def-ortner
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
export SYSTEM=C RUN_NAME=C_kfac_pre1000 NEPOCHS=1000
export ION_POS='((0.0,0.0,0.0),)' ION_CHARGES='(6.0,)' NELEC='(4,2)'
source /scratch/dexuan1/vmcnet/sweeps/templates/kfac_pre.sh
