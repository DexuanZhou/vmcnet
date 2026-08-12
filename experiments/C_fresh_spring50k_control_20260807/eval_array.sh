#!/bin/bash
#SBATCH --job-name=C-freshSPRING-eval
#SBATCH --account=def-ortner
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100_3g.40gb:1
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_C_stagewise_switch_20260807
REPO=/scratch/dexuan1/vmcnet
HERE=/scratch/dexuan1/vmcnet/experiments/C_fresh_spring50k_control_20260807
SOURCE=/scratch/dexuan1/runs/C_fresh_spring50k_control_20260807/train
ROOT=/scratch/dexuan1/runs/C_fresh_spring50k_control_20260807
SEED=${SLURM_ARRAY_TASK_ID}
RUN=${ROOT}/frozen_seed${SEED}

test -f "${SOURCE}/checkpoints/5000.npz"
[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite ${RUN}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export PYTHONPATH="${SNAPSHOT}"
export GIT_DIR="${REPO}/.git"
export GIT_WORK_TREE="${SNAPSHOT}"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false VMCNET_DISABLE_CHECKPOINTS=1
export FROZEN_SEED="${SEED}"

mkdir -p "$(dirname "${RUN}")" /scratch/dexuan1/runs/logs
cd "${SNAPSHOT}"
python "${HERE}/../C_late_spring50k_eta_drift_20260803/eval_seeded_launcher.py" \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/5000.npz \
  --reload.new_optimizer_state=True --reload.reburn=False \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${RUN}" --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nepochs=0 --config.vmc.disable_checkpointing=True \
  --config.eval.use_data_from_training=False \
  --config.eval.nchains=1000 --config.eval.nburn=5000 \
  --config.eval.nepochs=2000 --config.eval.nsteps_per_param_update=10 \
  --config.eval.record_local_energies=True --config.eval.nan_safe=False \
  --config.wandb.mode=disabled

test -f "${RUN}/eval/statistics.json"
