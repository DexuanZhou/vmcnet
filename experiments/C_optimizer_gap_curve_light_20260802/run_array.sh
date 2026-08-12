#!/bin/bash
#SBATCH --job-name=C-gapcurve-eval
#SBATCH --account=def-ortner
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-8%4
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/scratch/dexuan1/runs/C_optimizer_gap_curve_light_20260802
case "${SLURM_ARRAY_TASK_ID}" in
  0)
    METHOD=shared_kfac_pre1000
    EPOCH=1000
    SOURCE=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1
    ;;
  1|2|3|4)
    METHOD=spring_replicate2
    SOURCE=/scratch/dexuan1/runs/C_spring_official_kfac1000_E100000_replicate2
    case "${SLURM_ARRAY_TASK_ID}" in
      1) EPOCH=5000 ;;
      2) EPOCH=20000 ;;
      3) EPOCH=50000 ;;
      4) EPOCH=100000 ;;
    esac
    ;;
  5|6)
    METHOD=wssr_rank1600_eta03_lr004
    SOURCE=/scratch/dexuan1/runs/C_wssr_rank1600_eta03_lr004_E100000/lr0p04_eta0p3
    case "${SLURM_ARRAY_TASK_ID}" in
      5) EPOCH=5000 ;;
      6) EPOCH=20000 ;;
    esac
    ;;
  7|8)
    METHOD=wssr_rank1600_eta03_lr004
    SOURCE=/scratch/dexuan1/runs/C_wssr_rank1600_eta03_lr004_E100000_resume1
    case "${SLURM_ARRAY_TASK_ID}" in
      7) EPOCH=50000 ;;
      8) EPOCH=100000 ;;
    esac
    ;;
  *)
    echo "unexpected array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

RUN=${ROOT}/${METHOD}/epoch${EPOCH}
CHECKPOINT=${SOURCE}/checkpoints/${EPOCH}.npz
test -f "${CHECKPOINT}"
[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite ${RUN}" >&2; exit 2; }

mkdir -p "${RUN}/metadata"
printf '%s\n' \
  "purpose=matched_light_frozen_gap_curve" \
  "method=${METHOD}" \
  "source=${SOURCE}" \
  "checkpoint=${CHECKPOINT}" \
  "walkers=1000" \
  "burn_in=5000" \
  "measurement_epochs=2000" \
  "mcmc_steps=10" \
  "training=False" \
  > "${RUN}/metadata/protocol.txt"

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export VMCNET_DISABLE_CHECKPOINTS=1

cd /scratch/dexuan1/vmcnet
vmc-molecule \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path="checkpoints/${EPOCH}.npz" \
  --reload.new_optimizer_state=True --reload.reburn=False \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${RUN}" --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nepochs=0 --config.vmc.disable_checkpointing=True \
  --config.eval.nchains=1000 --config.eval.nburn=5000 \
  --config.eval.nepochs=2000 --config.eval.nsteps_per_param_update=10 \
  --config.eval.use_data_from_training=False \
  --config.eval.record_local_energies=True --config.wandb.mode=disabled

test -f "${RUN}/eval/statistics.json"
test -f "${RUN}/eval/accept_ratio.txt"
