#!/bin/bash
#SBATCH --job-name=C-lam4-x64
#SBATCH --account=def-ortner
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-3%4
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail
case "${SLURM_ARRAY_TASK_ID}" in
  0)
    LABEL=legacy_epoch101500
    RUN=/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage3_legacy_E1500
    EPOCH=101500
    ;;
  1)
    LABEL=fixed_lambda_1e-3_epoch101500
    RUN=/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage2_E2000
    EPOCH=101500
    ;;
  2)
    LABEL=legacy_epoch102000
    RUN=/scratch/dexuan1/runs/C_wssr_rank1600_hard_matched_control_E5000
    EPOCH=102000
    ;;
  3)
    LABEL=fixed_lambda_1e-3_epoch102000
    RUN=/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage2_E2000
    EPOCH=102000
    ;;
  *) exit 2 ;;
esac

OUTDIR=/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage4_x64_counterfactual
OUT=${OUTDIR}/${LABEL}.json
test -f "${RUN}/checkpoints/${EPOCH}.npz"
[[ ! -e "${OUT}" ]] || { echo "refusing to overwrite ${OUT}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${OUTDIR}"
cd /scratch/dexuan1/vmcnet
python experiments/C_rank1600_main_update_audit/audit_checkpoint_x64.py \
  --run "${RUN}" --epoch "${EPOCH}" --out "${OUT}"
test -s "${OUT}"
