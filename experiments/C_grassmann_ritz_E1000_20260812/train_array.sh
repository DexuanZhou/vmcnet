#!/bin/bash
#SBATCH --job-name=C-grass-E1k
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-3%2

set -euo pipefail
REPOSITORY="${VMCNET_REPOSITORY:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SOURCE="${C_KFAC_PRE1000:-${REPOSITORY}/reproducibility/kfac_initializers/C_kfac_pre1000}"
ROOT="${C_GRASSMANN_ROOT:-${SCRATCH:-/tmp/${USER}}/runs/C_grassmann_ritz_E1000_20260812}"
case "${SLURM_ARRAY_TASK_ID}" in
  0) ARM=alpha100; ALPHA=1.0 ;;
  1) ARM=alpha075; ALPHA=0.75 ;;
  2) ARM=alpha050; ALPHA=0.5 ;;
  3) ARM=alpha025; ALPHA=0.25 ;;
  *) exit 2 ;;
esac
RUN=${ROOT}/train/${ARM}
MONITOR=${ROOT}/metadata/${ARM}
test -f "${SOURCE}/checkpoints/1000.npz"
[[ ! -e "${RUN}" && ! -e "${MONITOR}" ]] || exit 2

if [[ -n "${VMCNET_MODULE_SETUP:-}" ]]; then
  eval "${VMCNET_MODULE_SETUP}"
fi
if [[ -n "${VMCNET_ENV_ACTIVATE:-}" ]]; then
  source "${VMCNET_ENV_ACTIVATE}"
fi
export PYTHONPATH="${REPOSITORY}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WSSR_MONITOR_DIR="${MONITOR}"
export WSSR_GRASSMANN_ALPHA="${ALPHA}"
export WSSR_SOURCE_CHECKPOINT="${SOURCE}/checkpoints/1000.npz"
mkdir -p "$(dirname "${RUN}")"
cd "${REPOSITORY}"
python experiments/C_grassmann_ritz_E1000_20260812/training_launcher.py \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/1000.npz \
  --reload.new_optimizer_state=True --reload.reburn=False \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${RUN}" --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nchains=1000 --config.vmc.nburn=0 \
  --config.vmc.nepochs=1000 --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.clip_threshold=5.0 --config.vmc.clip_center=mean \
  --config.vmc.check_for_nans=True --config.vmc.checkpoint_every=500 \
  --config.vmc.best_checkpoint_every=1000 --config.eval.nepochs=0 \
  --config.vmc.optimizer_type=wssr_warm_svd_right \
  --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.04 \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_S=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_g=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.relative_singular_value_cutoff=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.tikhonov_lambda=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True \
  --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint_mode=euclidean \
  --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=200 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=200 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank=200 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=200 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=40 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 \
  --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=True \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first=False \
  --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=grassmann_ritz \
  --config.vmc.optimizer.wssr_warm_svd_right.cluster_envelope_rank=200 \
  --config.vmc.optimizer.wssr_warm_svd_right.cluster_envelope_history=2 \
  --config.vmc.optimizer.wssr_warm_svd_right.grassmann_smoothing_alpha="${ALPHA}" \
  --config.wandb.mode=disabled

test -f "${RUN}/checkpoints/1000.npz"
test -f "${MONITOR}/monitor.csv"
