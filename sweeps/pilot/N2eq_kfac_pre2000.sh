#!/bin/bash
#SBATCH --job-name=N2eq-kfac-pre2000
#SBATCH --account=def-ortner
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
export SYSTEM=N2eq RUN_NAME=N2eq_kfac_pre2000 NEPOCHS=2000
export ION_POS='((0.0,0.0,-1.034),(0.0,0.0,1.034))' ION_CHARGES='(7.0,7.0)' NELEC='(7,7)'
source /scratch/dexuan1/vmcnet/sweeps/templates/kfac_pre.sh
