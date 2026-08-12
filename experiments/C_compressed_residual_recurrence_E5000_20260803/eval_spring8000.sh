#!/bin/bash
#SBATCH --job-name=C-spring8k-equal-time-eval
#SBATCH --account=def-ortner
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

SOURCE=/scratch/dexuan1/runs/C_spring_official_kfac1000_E100000_replicate2
ROOT=/scratch/dexuan1/runs/C_compressed_residual_recurrence_E5000_20260803
RUN=${ROOT}/frozen_equal_time/spring_replicate2_epoch8000
test -f "${SOURCE}/checkpoints/8000.npz"
[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite ${RUN}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false VMCNET_DISABLE_CHECKPOINTS=1

mkdir -p "$(dirname "${RUN}")" /scratch/dexuan1/runs/logs
cd /scratch/dexuan1/vmcnet
vmc-molecule \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/8000.npz \
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
