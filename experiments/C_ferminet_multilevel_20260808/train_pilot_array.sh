#!/bin/bash
#SBATCH --job-name=C-ML-pilot
#SBATCH --account=def-ortner
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gpus-per-node=h100_3g.40gb:1
#SBATCH --array=0-3%4
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_C_arch_multilevel_20260808
REPO=/scratch/dexuan1/vmcnet
HERE=${SNAPSHOT}/experiments/C_ferminet_multilevel_20260808
ROOT=/scratch/dexuan1/runs/C_ferminet_multilevel_pilot_20260808
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

case "${SLURM_ARRAY_TASK_ID}" in
  0) METHOD=spring; MODE=direct_L2 ;;
  1) METHOD=spring; MODE=multilevel ;;
  2) METHOD=wssr_warm_svd_right; MODE=direct_L2 ;;
  3) METHOD=wssr_warm_svd_right; MODE=multilevel ;;
  *) exit 2 ;;
esac

INITIAL=${ROOT}/optimizer_initial/${METHOD}
ARM=${ROOT}/pilot/${METHOD}/${MODE}
[[ -f "${INITIAL}/checkpoints/1.npz" ]] || exit 2
[[ ! -e "${ARM}" ]] || { echo "refusing to overwrite ${ARM}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export PYTHONPATH="${SNAPSHOT}"
export GIT_DIR="${REPO}/.git" GIT_WORK_TREE="${SNAPSHOT}"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${ARM}" /scratch/dexuan1/runs/logs
cd "${SNAPSHOT}"

make_level_config() {
  local base_config=$1
  local level=$2
  local output_dir=$3
  mkdir -p "${output_dir}"
  python "${HERE}/prepare_level_config.py" \
    --base-config="${base_config}" --level="${level}" \
    --output-dir="${output_dir}"
}

convert_level() {
  local coarse_config=$1
  local fine_config=$2
  local input_checkpoint=$3
  local output_dir=$4
  mkdir -p "${output_dir}"
  JAX_PLATFORMS=cpu python -m vmcnet.train.multilevel_checkpoint \
    --coarse-config="${coarse_config}" --fine-config="${fine_config}" \
    --input-checkpoint="${input_checkpoint}" \
    --output-dir="${output_dir}" --output-name=multilevel.npz \
    --template-seed=314159 --verify-samples=8
}

continue_level() {
  local reload_dir=$1
  local checkpoint_path=$2
  local run_dir=$3
  local target_epoch=$4
  vmc-molecule \
    --reload.logdir="${reload_dir}" --reload.use_config_file=True \
    --reload.use_checkpoint_file=True \
    --reload.checkpoint_relative_file_path="${checkpoint_path}" \
    --reload.new_optimizer_state=False --reload.reburn=False \
    --reload.append=False --reload.same_logdir=False \
    --config.logdir="${run_dir}" --config.base_logdir="${run_dir}" \
    --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
    --config.vmc.nchains=1000 --config.vmc.nburn=0 \
    --config.vmc.nepochs="${target_epoch}" \
    --config.vmc.nsteps_per_param_update=10 \
    --config.vmc.clip_threshold=5.0 --config.vmc.clip_center=mean \
    --config.vmc.check_for_nans=True \
    --config.vmc.checkpoint_every="${target_epoch}" \
    --config.vmc.best_checkpoint_every="${target_epoch}" \
    --config.eval.nepochs=0 --config.wandb.mode=disabled
  test -f "${run_dir}/checkpoints/${target_epoch}.npz"
}

if [[ "${MODE}" == direct_L2 ]]; then
  CONFIG_L2=${ARM}/config_L2
  CONVERTED_L2=${ARM}/converted_L2
  RUN_L2=${ARM}/train_L2
  make_level_config "${INITIAL}/config.json" L2 "${CONFIG_L2}"
  convert_level "${INITIAL}/config.json" "${CONFIG_L2}/config.json" \
    "${INITIAL}/checkpoints/1.npz" "${CONVERTED_L2}"
  continue_level "${CONVERTED_L2}" multilevel.npz "${RUN_L2}" 2000
else
  RUN_L0=${ARM}/train_L0_to500
  CONFIG_L1=${ARM}/config_L1
  CONVERTED_L1=${ARM}/converted_L1
  RUN_L1=${ARM}/train_L1_to1000
  CONFIG_L2=${ARM}/config_L2
  CONVERTED_L2=${ARM}/converted_L2
  RUN_L2=${ARM}/train_L2_to2000

  continue_level "${INITIAL}" checkpoints/1.npz "${RUN_L0}" 500
  make_level_config "${RUN_L0}/config.json" L1 "${CONFIG_L1}"
  convert_level "${RUN_L0}/config.json" "${CONFIG_L1}/config.json" \
    "${RUN_L0}/checkpoints/500.npz" "${CONVERTED_L1}"
  continue_level "${CONVERTED_L1}" multilevel.npz "${RUN_L1}" 1000
  make_level_config "${RUN_L1}/config.json" L2 "${CONFIG_L2}"
  convert_level "${RUN_L1}/config.json" "${CONFIG_L2}/config.json" \
    "${RUN_L1}/checkpoints/1000.npz" "${CONVERTED_L2}"
  continue_level "${CONVERTED_L2}" multilevel.npz "${RUN_L2}" 2000
fi
