#!/bin/bash
#SBATCH --job-name=wssr-k5k-select
#SBATCH --account=def-ortner
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420
#SBATCH --array=0-5%4
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err
set -euo pipefail
GLOBAL_TASK_ID=${SLURM_ARRAY_TASK_ID}
if (( GLOBAL_TASK_ID < 3 )); then
  export SYSTEM=C LOCAL_TASK_ID=${GLOBAL_TASK_ID}
  export SOURCE_OVERRIDE=/scratch/dexuan1/runs/wssr_C_4096_protocol/C_KFAC4096_pre5000
  export ROOT_OVERRIDE=/scratch/dexuan1/runs/wssr_C_4096_kfac5000_selection_v2
else
  export SYSTEM=N2 LOCAL_TASK_ID=$((GLOBAL_TASK_ID-3))
  export SOURCE_OVERRIDE=/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2/kfac_preliminary
  export ROOT_OVERRIDE=/scratch/dexuan1/runs/wssr_N2_4096_kfac5000_selection_v2
fi
export CHECKPOINT_NAME=5000.npz
exec bash /scratch/dexuan1/vmcnet/experiments/fir_wssr4096_kfac1000/scripts/select_array.sh
