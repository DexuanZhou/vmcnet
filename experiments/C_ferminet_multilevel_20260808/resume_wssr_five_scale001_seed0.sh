#!/bin/bash
# Resume the completed epoch-1000 five-level seed0 arm after fp32 guard audit.
#SBATCH --job-name=C-WSSR-ML001-R
#SBATCH --account=def-ortner
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gpus-per-node=h100_3g.40gb:1
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_C_arch_multilevel_fivelevel_20260808
REPO=/scratch/dexuan1/vmcnet
ROOT=/scratch/dexuan1/runs/C_ferminet_multilevel_fair_scale001_20260808
ARM=${ROOT}/five_level/seed0
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

SOURCE=${ARM}/train_L15_to1000
CONVERTED=${ARM}/converted_L2_retry
RUN=${ARM}/train_L2_to2000
[[ -f "${SOURCE}/checkpoints/1000.npz" ]] || exit 2
[[ ! -e "${CONVERTED}" ]] || { echo "refusing to overwrite ${CONVERTED}" >&2; exit 2; }
[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite ${RUN}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export PYTHONPATH="${SNAPSHOT}"
export GIT_DIR="${REPO}/.git" GIT_WORK_TREE="${SNAPSHOT}"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${CONVERTED}" /scratch/dexuan1/runs/logs
cd "${SNAPSHOT}"

JAX_PLATFORMS=cpu python -m vmcnet.train.multilevel_checkpoint \
  --coarse-config="${SOURCE}/config.json" \
  --fine-config="${ARM}/config_L2/config.json" \
  --input-checkpoint="${SOURCE}/checkpoints/1000.npz" \
  --output-dir="${CONVERTED}" --output-name=multilevel.npz \
  --template-seed=314159 --verify-samples=8 --sidecar-scale=0.01 \
  --equality-atol=2e-5

vmc-molecule \
  --reload.logdir="${CONVERTED}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=multilevel.npz \
  --reload.new_optimizer_state=False --reload.reburn=False \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${RUN}" --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nchains=1000 --config.vmc.nburn=0 \
  --config.vmc.nepochs=2000 --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.clip_threshold=5.0 --config.vmc.clip_center=mean \
  --config.vmc.check_for_nans=True --config.vmc.checkpoint_every=2000 \
  --config.vmc.best_checkpoint_every=2000 \
  --config.eval.nepochs=0 --config.wandb.mode=disabled

test -f "${RUN}/checkpoints/2000.npz"
