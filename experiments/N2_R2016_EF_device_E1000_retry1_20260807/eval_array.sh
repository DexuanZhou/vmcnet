#!/bin/bash
#SBATCH --job-name=N2-EF-frozen
#SBATCH --account=def-ortner
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-5%6
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_N2_EF_device_E1000_retry1_20260807
REPO=/scratch/dexuan1/vmcnet
ROOT=/scratch/dexuan1/runs/N2_R2016_EF_device_E1000_retry1_20260807
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

case "${SLURM_ARRAY_TASK_ID}" in
  0) ARM=A_vanilla; STEP=500 ;;
  1) ARM=A_vanilla; STEP=1000 ;;
  2) ARM=B_EF_mu095; STEP=500 ;;
  3) ARM=B_EF_mu095; STEP=1000 ;;
  4) ARM=SPRING_mu095; STEP=500 ;;
  5) ARM=SPRING_mu095; STEP=1000 ;;
  *) exit 2 ;;
esac

SOURCE=${ROOT}/train/${ARM}
RUN=${ROOT}/frozen/${ARM}_e${STEP}
META=${ROOT}/metadata/frozen/${ARM}_e${STEP}
CHECKPOINT=${SOURCE}/checkpoints/${STEP}.npz
[[ -d "${SNAPSHOT}" ]] || exit 2
[[ -f "${CHECKPOINT}" ]] || exit 2
[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite ${RUN}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export PYTHONPATH="${SNAPSHOT}"
export GIT_DIR="${REPO}/.git"
export GIT_WORK_TREE="${SNAPSHOT}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export VMCNET_DISABLE_CHECKPOINTS=1
mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd "${SNAPSHOT}"

hostname > "${META}/hostname.txt"
nvidia-smi > "${META}/nvidia_smi_start.txt"
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"

vmc-molecule \
  --reload.logdir="${SOURCE}" \
  --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path="checkpoints/${STEP}.npz" \
  --reload.new_optimizer_state=True \
  --reload.reburn=False \
  --reload.append=False \
  --reload.same_logdir=False \
  --config.logdir="${RUN}" \
  --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name=NONE \
  --config.vmc.nepochs=0 \
  --config.vmc.disable_checkpointing=True \
  --config.vmc.optimizer_type=adam \
  --config.eval.use_data_from_training=False \
  --config.eval.nchains=2000 \
  --config.eval.nburn=10000 \
  --config.eval.nepochs=2000 \
  --config.eval.nsteps_per_param_update=10 \
  --config.eval.record_local_energies=True \
  --config.wandb.mode=disabled

[[ -f "${RUN}/eval/statistics.json" ]] || exit 5
nvidia-smi > "${META}/nvidia_smi_end.txt"
