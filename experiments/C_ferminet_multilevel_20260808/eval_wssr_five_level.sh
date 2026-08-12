#!/bin/bash
# Independent frozen evaluation for the completed five-level WSSR endpoint.
#SBATCH --job-name=C-WSSR-5L-eval
#SBATCH --account=def-ortner
#SBATCH --time=00:35:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100_3g.40gb:1
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_C_arch_multilevel_fivelevel_20260808
REPO=/scratch/dexuan1/vmcnet
HERE=${SNAPSHOT}/experiments/C_ferminet_multilevel_20260808
ROOT=/scratch/dexuan1/runs/C_ferminet_multilevel_pilot_20260808
SOURCE=${ROOT}/pilot/wssr_warm_svd_right/five_level/train_L2_to2000
RUN=${ROOT}/frozen/wssr_warm_svd_right/five_level
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

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
