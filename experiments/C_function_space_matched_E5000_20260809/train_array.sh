#!/bin/bash
#SBATCH --job-name=C-fnorm-match
#SBATCH --account=def-ortner
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-3%4
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

REPO=/scratch/dexuan1/vmcnet
SOURCE=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1
ROOT=/scratch/dexuan1/runs/C_function_space_matched_E5000_20260809
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311
TARGET_EPOCH=${TARGET_EPOCH:-5000}
RUN_LABEL=${RUN_LABEL:-train}
FUNCTION_RADIUS=${FUNCTION_RADIUS:-0.0008}
WSSR_ETA_S=${WSSR_ETA_S:--1.0}
WSSR_ETA_G=${WSSR_ETA_G:--1.0}
WSSR_COMPLEMENT_WEIGHT=${WSSR_COMPLEMENT_WEIGHT:-0.0}
WSSR_SSI_WARM=${WSSR_SSI_WARM:-2}
if [[ "${TARGET_EPOCH}" -lt 5000 ]]; then
  SAVE_EVERY=1
else
  SAVE_EVERY=5000
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0) METHOD=spring; SEED=0 ;;
  1) METHOD=wssr_rank1600; SEED=0 ;;
  2) METHOD=spring; SEED=1 ;;
  3) METHOD=wssr_rank1600; SEED=1 ;;
  *) echo "invalid task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

RUN=${ROOT}/${RUN_LABEL}/${METHOD}/seed${SEED}
META=${ROOT}/metadata_${RUN_LABEL}/${METHOD}/seed${SEED}
CHECKPOINT=${SOURCE}/checkpoints/1000.npz
test -f "${CHECKPOINT}"
[[ ! -e "${RUN}" && ! -e "${META}" ]] || {
  echo "refusing to overwrite ${RUN} or ${META}" >&2
  exit 2
}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export FUNCTION_CAP_METADATA="${META}"
export FUNCTION_CAP_SEED="${SEED}"
export FUNCTION_CAP_METHOD="${METHOD}"
export FUNCTION_CAP_RADIUS="${FUNCTION_RADIUS}"
export FUNCTION_CAP_TARGET_EPOCH="${TARGET_EPOCH}"
export FUNCTION_CAP_ETA_S="${WSSR_ETA_S}"
export FUNCTION_CAP_ETA_G="${WSSR_ETA_G}"
export FUNCTION_CAP_COMPLEMENT_WEIGHT="${WSSR_COMPLEMENT_WEIGHT}"
export FUNCTION_CAP_SSI_WARM="${WSSR_SSI_WARM}"

mkdir -p "$(dirname "${RUN}")" /scratch/dexuan1/runs/logs
cd "${REPO}"

COMMON=(
  --reload.logdir="${SOURCE}"
  --reload.use_config_file=True
  --reload.use_checkpoint_file=True
  --reload.checkpoint_relative_file_path=checkpoints/1000.npz
  --reload.new_optimizer_state=True
  --reload.reburn=False
  --reload.append=False
  --reload.same_logdir=False
  --config.logdir="${RUN}"
  --config.base_logdir="${RUN}"
  --config.save_to_current_datetime_subfolder=False
  --config.subfolder_name=NONE
  --config.vmc.nchains=1000
  --config.vmc.nburn=0
  --config.vmc.nepochs="${TARGET_EPOCH}"
  --config.vmc.nsteps_per_param_update=10
  --config.vmc.clip_center=mean
  --config.vmc.clip_threshold=5.0
  --config.vmc.check_for_nans=True
  --config.vmc.checkpoint_every="${SAVE_EVERY}"
  --config.vmc.best_checkpoint_every="${TARGET_EPOCH}"
  --config.eval.nburn=0
  --config.eval.nepochs=0
  --config.wandb.mode=disabled
)

if [[ "${METHOD}" == spring ]]; then
  OPT=(
    --config.vmc.optimizer_type=spring
    --config.vmc.optimizer.spring.schedule_type=inverse_time
    --config.vmc.optimizer.spring.learning_rate=0.02
    --config.vmc.optimizer.spring.learning_decay_rate=0.0001
    --config.vmc.optimizer.spring.mu=0.99
    --config.vmc.optimizer.spring.damping=0.001
    --config.vmc.optimizer.spring.constrain_norm=True
    --config.vmc.optimizer.spring.norm_constraint_mode=function_space
    --config.vmc.optimizer.spring.function_norm_constraint="${FUNCTION_RADIUS}"
  )
else
  OPT=(
    --config.vmc.optimizer_type=wssr_warm_svd_right
    --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time
    --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.04
    --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001
    --config.vmc.optimizer.wssr_warm_svd_right.eta=0.3
    --config.vmc.optimizer.wssr_warm_svd_right.eta_S="${WSSR_ETA_S}"
    --config.vmc.optimizer.wssr_warm_svd_right.eta_g="${WSSR_ETA_G}"
    --config.vmc.optimizer.wssr_warm_svd_right.enable_gradient_transport=False
    --config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003
    --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov
    --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True
    --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint_mode=function_space
    --config.vmc.optimizer.wssr_warm_svd_right.function_norm_constraint="${FUNCTION_RADIUS}"
    --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=1600
    --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=1600
    --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank=1600
    --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=1600
    --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=40
    --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm="${WSSR_SSI_WARM}"
    --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=True
    --config.vmc.optimizer.wssr_warm_svd_right.exact_first=False
    --config.vmc.optimizer.wssr_warm_svd_right.exact_first_force=False
    --config.vmc.optimizer.wssr_warm_svd_right.exact_reference_diagnostics=False
    --config.vmc.optimizer.wssr_warm_svd_right.update_diagnostics=False
    --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=none
    --config.vmc.optimizer.wssr_warm_svd_right.complement_weight="${WSSR_COMPLEMENT_WEIGHT}"
    --config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_beta=0.0
  )
fi

python experiments/C_function_space_matched_E5000_20260809/launcher.py \
  "${COMMON[@]}" "${OPT[@]}"

test -f "${RUN}/checkpoints/${TARGET_EPOCH}.npz"
test -f "${RUN}/training_metrics.csv"
