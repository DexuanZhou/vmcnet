#!/bin/bash
#SBATCH --job-name=C-ML-KFAC-L0
#SBATCH --account=def-ortner
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_C_arch_multilevel_20260808
REPO=/scratch/dexuan1/vmcnet
HERE=${SNAPSHOT}/experiments/C_ferminet_multilevel_20260808
BASE=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1/config.json
ROOT=/scratch/dexuan1/runs/C_ferminet_multilevel_pilot_20260808
CONFIG_ROOT=${ROOT}/configs/L0_initial
RUN=${ROOT}/kfac_l0_pre1000
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

[[ -d "${SNAPSHOT}" ]] || exit 2
[[ -f "${BASE}" ]] || exit 2
[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite ${RUN}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export PYTHONPATH="${SNAPSHOT}"
export GIT_DIR="${REPO}/.git" GIT_WORK_TREE="${SNAPSHOT}"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${CONFIG_ROOT}" /scratch/dexuan1/runs/logs
cd "${SNAPSHOT}"

python "${HERE}/prepare_level_config.py" \
  --base-config="${BASE}" --level=L0 --output-dir="${CONFIG_ROOT}"

vmc-molecule \
  --reload.logdir="${CONFIG_ROOT}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=False \
  --config.logdir="${RUN}" --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.initial_seed=0 \
  --config.vmc.nchains=1000 --config.vmc.nburn=5000 \
  --config.vmc.nepochs=1000 --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.clip_threshold=5.0 --config.vmc.clip_center=mean \
  --config.vmc.check_for_nans=True \
  --config.vmc.checkpoint_every=1000 --config.vmc.best_checkpoint_every=1000 \
  --config.eval.nepochs=0 \
  --config.vmc.optimizer_type=kfac \
  --config.vmc.optimizer.kfac.schedule_type=inverse_time \
  --config.vmc.optimizer.kfac.learning_rate=0.05 \
  --config.vmc.optimizer.kfac.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.kfac.damping=0.001 \
  --config.vmc.optimizer.kfac.norm_constraint=0.001 \
  --config.wandb.mode=disabled

test -f "${RUN}/checkpoints/1000.npz"
