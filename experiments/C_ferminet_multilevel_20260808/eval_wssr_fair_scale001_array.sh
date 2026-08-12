#!/bin/bash
# Matched independent frozen evaluation for the scale-0.01 WSSR array.
#SBATCH --job-name=C-WSSR-ML001-E
#SBATCH --account=def-ortner
#SBATCH --time=00:35:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100_3g.40gb:1
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_C_arch_multilevel_fivelevel_20260808
REPO=/scratch/dexuan1/vmcnet
HERE=${SNAPSHOT}/experiments/C_ferminet_multilevel_20260808
RUN_LABEL=${RUN_LABEL:-scale001}
ROOT=/scratch/dexuan1/runs/C_ferminet_multilevel_fair_${RUN_LABEL}_20260808
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311
RUN_SUFFIX=${RUN_SUFFIX:-}

case "${SLURM_ARRAY_TASK_ID}" in
  0) LEVELS=three; REPLICATE=0 ;;
  1) LEVELS=three; REPLICATE=1 ;;
  2) LEVELS=five;  REPLICATE=0 ;;
  3) LEVELS=five;  REPLICATE=1 ;;
  *) exit 2 ;;
esac

SOURCE=${ROOT}/${LEVELS}_level/seed${REPLICATE}/train_L2_to2000
RUN=${ROOT}/frozen/${LEVELS}_level/seed${REPLICATE}${RUN_SUFFIX}
[[ -f "${SOURCE}/checkpoints/2000.npz" ]] || exit 2
[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite ${RUN}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export PYTHONPATH="${SNAPSHOT}"
export GIT_DIR="${REPO}/.git" GIT_WORK_TREE="${SNAPSHOT}"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false VMCNET_DISABLE_CHECKPOINTS=1
mkdir -p "$(dirname "${RUN}")" /scratch/dexuan1/runs/logs
cd "${SNAPSHOT}"

python "${HERE}/eval_seeded_launcher.py" \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/2000.npz \
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
