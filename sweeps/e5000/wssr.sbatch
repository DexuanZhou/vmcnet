#!/bin/bash
#SBATCH --account=def-ortner
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

export RUN_ROOT=/scratch/dexuan1/runs/e5000 NEPOCHS=5000 CHECKPOINT_EVERY=500
export RANK=1600 STORAGE_RANK=1600 WORKING_RANK=1600 MAXITER_WARM=2
export SPECTRAL_REG=tikhonov
if [[ "${SYSTEM}" == C ]]; then
  export ION_POS='((0.0,0.0,0.0),)'
  export PRE_DIR=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1 PRE_EPOCH=1000
else
  export ION_POS='((0.0,0.0,-1.034),(0.0,0.0,1.034))'
  export PRE_DIR=/scratch/dexuan1/runs/pilot/N2eq/N2eq_kfac_pre2000 PRE_EPOCH=2000
fi
source /scratch/dexuan1/vmcnet/sweeps/templates/wssr_run.sh
