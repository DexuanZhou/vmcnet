#!/bin/bash
#SBATCH --job-name=C-r1600-prox-eval
#SBATCH --account=def-ortner
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

TRAIN_ROOT=/scratch/dexuan1/runs/C_wssr_rank1600_native_proximal_gamma3e4_matched_E500_20260802
EVAL_ROOT=/scratch/dexuan1/runs/C_wssr_rank1600_native_proximal_gamma3e4_matched_E500_20260802_frozen
labels=(hard_control native_proximal)
label=${labels[${SLURM_ARRAY_TASK_ID}]}
SOURCE=${TRAIN_ROOT}/${label}
RUN=${EVAL_ROOT}/${label}/epoch102500

test -f "${SOURCE}/checkpoints/102500.npz"
[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite ${RUN}" >&2; exit 2; }

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
  --reload.checkpoint_relative_file_path=checkpoints/102500.npz \
  --reload.new_optimizer_state=True --reload.reburn=False \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${RUN}" --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nepochs=0 --config.vmc.disable_checkpointing=True \
  --config.eval.nchains=2000 --config.eval.nburn=5000 \
  --config.eval.nepochs=20000 --config.eval.nsteps_per_param_update=10 \
  --config.eval.use_data_from_training=False \
  --config.eval.record_local_energies=True --config.wandb.mode=disabled

test -f "${RUN}/eval/statistics.json"
test -f "${RUN}/eval/accept_ratio.txt"
