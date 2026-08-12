#!/bin/bash
# Paired three-vs-five-level WSSR test with controlled sidecar activation.
#SBATCH --job-name=C-WSSR-ML001
#SBATCH --account=def-ortner
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gpus-per-node=h100_3g.40gb:1
#SBATCH --array=0-3%4
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_C_arch_multilevel_fivelevel_20260808
REPO=/scratch/dexuan1/vmcnet
HERE=${SNAPSHOT}/experiments/C_ferminet_multilevel_20260808
RUN_LABEL=${RUN_LABEL:-scale001}
SIDECAR_SCALE=${SIDECAR_SCALE:-0.01}
ROOT=/scratch/dexuan1/runs/C_ferminet_multilevel_fair_${RUN_LABEL}_20260808
INITIAL=/scratch/dexuan1/runs/C_ferminet_multilevel_pilot_20260808/optimizer_initial/wssr_warm_svd_right
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

case "${SLURM_ARRAY_TASK_ID}" in
  0) LEVELS=three; TEMPLATE_SEED=314159; REPLICATE=0 ;;
  1) LEVELS=three; TEMPLATE_SEED=271828; REPLICATE=1 ;;
  2) LEVELS=five;  TEMPLATE_SEED=314159; REPLICATE=0 ;;
  3) LEVELS=five;  TEMPLATE_SEED=271828; REPLICATE=1 ;;
  *) exit 2 ;;
esac

ARM=${ROOT}/${LEVELS}_level/seed${REPLICATE}
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
  local base_config=$1 level=$2 output_dir=$3
  mkdir -p "${output_dir}"
  python "${HERE}/prepare_level_config.py" \
    --base-config="${base_config}" --level="${level}" \
    --output-dir="${output_dir}"
}

convert_level() {
  local coarse_config=$1 fine_config=$2 input_checkpoint=$3 output_dir=$4
  mkdir -p "${output_dir}"
  JAX_PLATFORMS=cpu python -m vmcnet.train.multilevel_checkpoint \
    --coarse-config="${coarse_config}" --fine-config="${fine_config}" \
    --input-checkpoint="${input_checkpoint}" \
    --output-dir="${output_dir}" --output-name=multilevel.npz \
    --template-seed="${TEMPLATE_SEED}" --verify-samples=8 \
    --sidecar-scale="${SIDECAR_SCALE}" --equality-atol=2e-5
}

continue_level() {
  local reload_dir=$1 checkpoint_path=$2 run_dir=$3 target_epoch=$4
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

if [[ "${LEVELS}" == three ]]; then
  continue_level "${INITIAL}" checkpoints/1.npz "${ARM}/train_L0_to500" 500
  make_level_config "${ARM}/train_L0_to500/config.json" L1 "${ARM}/config_L1"
  convert_level "${ARM}/train_L0_to500/config.json" "${ARM}/config_L1/config.json" \
    "${ARM}/train_L0_to500/checkpoints/500.npz" "${ARM}/converted_L1"
  continue_level "${ARM}/converted_L1" multilevel.npz "${ARM}/train_L1_to1000" 1000
  make_level_config "${ARM}/train_L1_to1000/config.json" L2 "${ARM}/config_L2"
  convert_level "${ARM}/train_L1_to1000/config.json" "${ARM}/config_L2/config.json" \
    "${ARM}/train_L1_to1000/checkpoints/1000.npz" "${ARM}/converted_L2"
  continue_level "${ARM}/converted_L2" multilevel.npz "${ARM}/train_L2_to2000" 2000
else
  continue_level "${INITIAL}" checkpoints/1.npz "${ARM}/train_L0_to250" 250
  make_level_config "${ARM}/train_L0_to250/config.json" L05 "${ARM}/config_L05"
  convert_level "${ARM}/train_L0_to250/config.json" "${ARM}/config_L05/config.json" \
    "${ARM}/train_L0_to250/checkpoints/250.npz" "${ARM}/converted_L05"
  continue_level "${ARM}/converted_L05" multilevel.npz "${ARM}/train_L05_to500" 500
  make_level_config "${ARM}/train_L05_to500/config.json" L1 "${ARM}/config_L1"
  convert_level "${ARM}/train_L05_to500/config.json" "${ARM}/config_L1/config.json" \
    "${ARM}/train_L05_to500/checkpoints/500.npz" "${ARM}/converted_L1"
  continue_level "${ARM}/converted_L1" multilevel.npz "${ARM}/train_L1_to750" 750
  make_level_config "${ARM}/train_L1_to750/config.json" L15 "${ARM}/config_L15"
  convert_level "${ARM}/train_L1_to750/config.json" "${ARM}/config_L15/config.json" \
    "${ARM}/train_L1_to750/checkpoints/750.npz" "${ARM}/converted_L15"
  continue_level "${ARM}/converted_L15" multilevel.npz "${ARM}/train_L15_to1000" 1000
  make_level_config "${ARM}/train_L15_to1000/config.json" L2 "${ARM}/config_L2"
  convert_level "${ARM}/train_L15_to1000/config.json" "${ARM}/config_L2/config.json" \
    "${ARM}/train_L15_to1000/checkpoints/1000.npz" "${ARM}/converted_L2"
  continue_level "${ARM}/converted_L2" multilevel.npz "${ARM}/train_L2_to2000" 2000
fi
