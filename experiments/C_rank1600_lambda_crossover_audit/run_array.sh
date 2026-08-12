#!/bin/bash
#SBATCH --job-name=C-lambda-cross
#SBATCH --account=def-ortner
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-1%2
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail
ROOT=${ROOT:-/scratch/dexuan1/runs/C_wssr_rank1600_lambda_crossover_audit_20260802}
REPLICATES=${REPLICATES:-6}

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    SOURCE_RUN=/scratch/dexuan1/runs/C_wssr_rank1600_eta03_lr004_E100000_resume1
    SOURCE_EPOCH=100000
    LABEL=legacy_epoch100000
    ;;
  1)
    SOURCE_RUN=/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage2_E2000
    SOURCE_EPOCH=102000
    LABEL=fixedlambda_epoch102000
    ;;
  *) exit 2 ;;
esac

OUT=${ROOT}/${LABEL}
[[ ! -e "${OUT}" ]] || { echo "refusing to overwrite ${OUT}" >&2; exit 2; }
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

mkdir -p "${OUT}" /scratch/dexuan1/runs/logs
cd /scratch/dexuan1/vmcnet
git rev-parse HEAD > "${OUT}/git_commit.txt"
git status --short > "${OUT}/git_status_short.txt"
scontrol show job "${SLURM_JOB_ID}" > "${OUT}/slurm_job.txt"
cp experiments/C_rank1600_lambda_crossover_audit/README.md \
  "${OUT}/preregistered_protocol.md"
python experiments/C_rank1600_lambda_crossover_audit/audit.py \
  --run "${SOURCE_RUN}" --epoch "${SOURCE_EPOCH}" \
  --replicates "${REPLICATES}" --output "${OUT}/audit.json"
test -s "${OUT}/audit.json"
