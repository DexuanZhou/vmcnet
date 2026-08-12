#!/bin/bash
#SBATCH --job-name=N2-env-frozen
#SBATCH --account=def-ortner
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --array=0-5%6
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail
ROOT=/scratch/dexuan1/runs/N2_cluster_envelope_E5000_20260809
SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_cluster_envelope_20260809
REPO=/scratch/dexuan1/vmcnet
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311
ARMS=(A_hard_rank200 B_cluster_envelope)
STEPS=(1000 2500 5000)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}
STEP=${STEPS[$((SLURM_ARRAY_TASK_ID % 3))]}
SOURCE=${ROOT}/${ARM}
RUN=${ROOT}/frozen/${ARM}_e${STEP}
CHECKPOINT=${SOURCE}/checkpoints/${STEP}.npz
[[ -f "${CHECKPOINT}" ]] || exit 2
[[ ! -e "${RUN}" ]] || exit 2

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export PYTHONPATH="${SNAPSHOT}"
export GIT_DIR="${REPO}/.git"
export GIT_WORK_TREE="${SNAPSHOT}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${RUN}/metadata" /scratch/dexuan1/runs/logs
cd "${SNAPSHOT}"
hostname > "${RUN}/metadata/hostname.txt"
nvidia-smi > "${RUN}/metadata/nvidia_smi_start.txt"

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

[[ -f "${RUN}/eval/statistics.json" ]] || {
  # Some current runners write scalar eval series at the logdir root.
  [[ -f "${RUN}/energy.txt" && -f "${RUN}/variance.txt" ]] || exit 5
}
nvidia-smi > "${RUN}/metadata/nvidia_smi_end.txt"
