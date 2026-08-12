#!/bin/bash
# Continue the manually selected, stable L1 -> L2 sidecar gate to epoch 2000.
# Submit with, for example: sbatch --export=ALL,TAG=scale001 spring_sidecar_continue.sh
#SBATCH --job-name=C-ML-sidecar-cont
#SBATCH --account=def-ortner
#SBATCH --time=00:35:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100_3g.40gb:1
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

: "${TAG:?submit with --export=ALL,TAG=scale01 or scale001}"
case "${TAG}" in
  scale01|scale001) ;;
  *) echo "unsupported sidecar tag: ${TAG}" >&2; exit 2 ;;
esac

SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_C_arch_multilevel_sidecar_20260808
REPO=/scratch/dexuan1/vmcnet
ROOT=/scratch/dexuan1/runs/C_ferminet_multilevel_pilot_20260808
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311
SOURCE=${ROOT}/spring_sidecar_gate/${TAG}/train_L2_to1200
RUN=${ROOT}/spring_sidecar_gate/${TAG}/train_L2_to2000

[[ -f "${SOURCE}/checkpoints/1200.npz" ]] || exit 2
[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite ${RUN}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export PYTHONPATH="${SNAPSHOT}"
export GIT_DIR="${REPO}/.git" GIT_WORK_TREE="${SNAPSHOT}"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$(dirname "${RUN}")" /scratch/dexuan1/runs/logs
cd "${SNAPSHOT}"

vmc-molecule \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/1200.npz \
  --reload.new_optimizer_state=False --reload.reburn=False \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${RUN}" --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nchains=1000 --config.vmc.nburn=0 \
  --config.vmc.nepochs=2000 --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.clip_threshold=5.0 --config.vmc.clip_center=mean \
  --config.vmc.check_for_nans=True \
  --config.vmc.checkpoint_every=2000 --config.vmc.best_checkpoint_every=2000 \
  --config.eval.nepochs=0 --config.wandb.mode=disabled

test -f "${RUN}/checkpoints/2000.npz"
