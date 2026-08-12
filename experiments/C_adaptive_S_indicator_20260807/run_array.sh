#!/bin/bash
#SBATCH --job-name=C-Smix-indicator
#SBATCH --account=def-ortner
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-6%4
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

ROOT=${ROOT:-/scratch/dexuan1/runs/C_adaptive_S_indicator_20260807}
REPLICATES=${REPLICATES:-4}
STATIC_FLAG=()

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    LABEL=static_pre1000
    RUN=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1
    CHECKPOINT=${RUN}/checkpoints/1000.npz
    ETA=0.95
    STATIC_FLAG=(--static-history)
    ;;
  1|2|3)
    case "${SLURM_ARRAY_TASK_ID}" in
      1) EPOCH=1000 ;;
      2) EPOCH=3000 ;;
      3) EPOCH=5000 ;;
    esac
    LABEL=early_eta095_e${EPOCH}
    RUN=/scratch/dexuan1/runs/C_wssr_split_eta_E5000_20260803/train/A_etaS095_etaG0/seed0
    CHECKPOINT=${RUN}/checkpoints/${EPOCH}.npz
    ETA=0.95
    ;;
  4|5|6)
    case "${SLURM_ARRAY_TASK_ID}" in
      4) TAG=03; ETA=0.3 ;;
      5) TAG=08; ETA=0.8 ;;
      6) TAG=095; ETA=0.95 ;;
    esac
    LABEL=late_eta${TAG}
    RUN=/scratch/dexuan1/runs/C_late_spring50k_eta_drift_20260803/train/eta${TAG}
    CHECKPOINT=${RUN}/checkpoints/5000.npz
    ;;
  *)
    echo "unexpected array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

OUT=${ROOT}/${LABEL}
[[ ! -e "${OUT}" ]] || {
  echo "refusing to overwrite ${OUT}" >&2
  exit 2
}

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
cp experiments/C_adaptive_S_indicator_20260807/README.md \
  "${OUT}/preregistered_protocol.md"

# Legacy rank-1600 checkpoints contain a single ~8 GiB object member.  Loading
# that member directly from the shared filesystem degenerates into many small
# reads, so stage the immutable source archive on the node-local SSD first.
# This changes only checkpoint I/O, not any replay state or scientific setting.
SOURCE_CHECKPOINT=${CHECKPOINT}
if [[ "${STAGE_CHECKPOINT:-1}" == "1" ]]; then
  LOCAL_CHECKPOINT=${SLURM_TMPDIR}/$(basename "${CHECKPOINT}")
  cp "${CHECKPOINT}" "${LOCAL_CHECKPOINT}"
  CHECKPOINT=${LOCAL_CHECKPOINT}
fi
printf '%s\n' "${SOURCE_CHECKPOINT}" > "${OUT}/source_checkpoint.txt"

python experiments/C_adaptive_S_indicator_20260807/audit.py \
  --run "${RUN}" \
  --checkpoint "${CHECKPOINT}" \
  --label "${LABEL}" \
  --eta "${ETA}" \
  --replicates "${REPLICATES}" \
  "${STATIC_FLAG[@]}" \
  --output "${OUT}/audit.json"

test -s "${OUT}/audit.json"
