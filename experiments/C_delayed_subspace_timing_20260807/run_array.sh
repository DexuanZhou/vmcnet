#!/bin/bash
#SBATCH --job-name=C-delaySSI-200
#SBATCH --account=def-ortner
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-2%3
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

case "${SLURM_ARRAY_TASK_ID}" in
  0) PERIOD=1 ;;
  1) PERIOD=2 ;;
  2) PERIOD=5 ;;
  *) echo "unexpected task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_C_delayed_refresh_20260807
REPO=/scratch/dexuan1/vmcnet
SOURCE=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1
ROOT=/scratch/dexuan1/runs/C_delayed_subspace_timing_20260807
ARM=refresh${PERIOD}
RUN=${ROOT}/train/${ARM}
MONITOR=${ROOT}/metadata/${ARM}

test -f "${SNAPSHOT}/SOURCE_SNAPSHOT.txt"
test -f "${SOURCE}/checkpoints/1000.npz"
[[ ! -e "${RUN}" && ! -e "${MONITOR}" ]] || {
  echo "refusing to overwrite ${RUN} or ${MONITOR}" >&2
  exit 2
}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export PYTHONPATH="${SNAPSHOT}"
export GIT_DIR="${REPO}/.git"
export GIT_WORK_TREE="${SNAPSHOT}"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WSSR_MONITOR_DIR="${MONITOR}"
export WSSR_ARM="${ARM}" WSSR_RANK=800 WSSR_SEED=0
export WSSR_UPDATES=200 WSSR_MU=0.99
export WSSR_RESIDUAL_EVALUATION=full_current_batch
export WSSR_DUAL_DIAGNOSTICS=0 WSSR_ERROR_FEEDBACK=0
export WSSR_SUBSPACE_ETA_S=0.0
export WSSR_SUBSPACE_REFRESH_PERIOD="${PERIOD}"

mkdir -p "$(dirname "${RUN}")" /scratch/dexuan1/runs/logs
cd "${SNAPSHOT}"
python experiments/C_galerkin_residual_recurrence_20260803/training_launcher.py \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/1000.npz \
  --reload.new_optimizer_state=True --reload.reburn=False \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${RUN}" --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nchains=1000 --config.vmc.nburn=0 \
  --config.vmc.nepochs=200 --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.check_for_nans=True --config.vmc.disable_checkpointing=True \
  --config.eval.nepochs=0 \
  --config.vmc.optimizer_type=wssr_warm_svd_right \
  --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.04 \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_S=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_g=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_bias_correction=False \
  --config.vmc.optimizer.wssr_warm_svd_right.solution_recurrence_mode=residual \
  --config.vmc.optimizer.wssr_warm_svd_right.solution_recurrence_mu=0.99 \
  --config.vmc.optimizer.wssr_warm_svd_right.residual_evaluation=full_current_batch \
  --config.vmc.optimizer.wssr_warm_svd_right.residual_dual_mode_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.solution_error_feedback=False \
  --config.vmc.optimizer.wssr_warm_svd_right.subspace_eta_S=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.subspace_refresh_period="${PERIOD}" \
  --config.vmc.optimizer.wssr_warm_svd_right.recurrence_telemetry=True \
  --config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.relative_singular_value_cutoff=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.tikhonov_lambda=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov \
  --config.vmc.optimizer.wssr_warm_svd_right.mixed_precision_solve=True \
  --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True \
  --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=800 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=800 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank=800 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=800 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=40 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 \
  --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=True \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first=False \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_reference_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.update_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.reliability_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=none \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0.0 \
  --config.wandb.mode=disabled

test -f "${RUN}/training_metrics.csv"
test -f "${MONITOR}/monitor.csv"
