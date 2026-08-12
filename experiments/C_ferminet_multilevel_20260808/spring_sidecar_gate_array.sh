#!/bin/bash
#SBATCH --job-name=C-ML-sidecar
#SBATCH --account=def-ortner
#SBATCH --time=00:35:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100_3g.40gb:1
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_C_arch_multilevel_sidecar_20260808
REPO=/scratch/dexuan1/vmcnet
HERE=${SNAPSHOT}/experiments/C_ferminet_multilevel_20260808
ROOT=/scratch/dexuan1/runs/C_ferminet_multilevel_pilot_20260808
SOURCE=${ROOT}/pilot/spring/multilevel/train_L1_to1000
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

case "${SLURM_ARRAY_TASK_ID}" in
  0) SCALE=0.1; TAG=scale01 ;;
  1) SCALE=0.01; TAG=scale001 ;;
  *) exit 2 ;;
esac
ARM=${ROOT}/spring_sidecar_gate/${TAG}
CONFIG=${ARM}/config_L2
CONVERTED=${ARM}/converted_L2
RUN=${ARM}/train_L2_to1200
[[ -f "${SOURCE}/checkpoints/1000.npz" ]] || exit 2
[[ ! -e "${ARM}" ]] || { echo "refusing to overwrite ${ARM}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export PYTHONPATH="${SNAPSHOT}"
export GIT_DIR="${REPO}/.git" GIT_WORK_TREE="${SNAPSHOT}"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${CONFIG}" "${CONVERTED}" /scratch/dexuan1/runs/logs
cd "${SNAPSHOT}"

python "${HERE}/prepare_level_config.py" \
  --base-config="${SOURCE}/config.json" --level=L2 --output-dir="${CONFIG}"
JAX_PLATFORMS=cpu python -m vmcnet.train.multilevel_checkpoint \
  --coarse-config="${SOURCE}/config.json" --fine-config="${CONFIG}/config.json" \
  --input-checkpoint="${SOURCE}/checkpoints/1000.npz" \
  --output-dir="${CONVERTED}" --output-name=multilevel.npz \
  --template-seed=314159 --verify-samples=8 --sidecar-scale="${SCALE}"

vmc-molecule \
  --reload.logdir="${CONVERTED}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=multilevel.npz \
  --reload.new_optimizer_state=False --reload.reburn=False \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${RUN}" --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nchains=1000 --config.vmc.nburn=0 \
  --config.vmc.nepochs=1200 --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.clip_threshold=5.0 --config.vmc.clip_center=mean \
  --config.vmc.check_for_nans=True \
  --config.vmc.checkpoint_every=1200 --config.vmc.best_checkpoint_every=1200 \
  --config.eval.nepochs=0 --config.wandb.mode=disabled

test -f "${RUN}/checkpoints/1200.npz"
