#!/bin/bash
#SBATCH --job-name=C-aniso-S
#SBATCH --account=def-ortner
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-5%3
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail
SOURCE=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1
ROOT=/scratch/dexuan1/runs/C_anisotropic_matrix_history_E5000_20260812
SEED=$((SLURM_ARRAY_TASK_ID / 3))
case "$((SLURM_ARRAY_TASK_ID % 3))" in
  0) ARM=current_only;       ETA_S=0.0;  ANISO=False; ANISO_ENV=0 ;;
  1) ARM=uniform_matrix;     ETA_S=0.3;  ANISO=False; ANISO_ENV=0 ;;
  2) ARM=anisotropic_matrix; ETA_S=0.95; ANISO=True;  ANISO_ENV=1 ;;
  *) exit 2 ;;
esac
RUN=${ROOT}/train/${ARM}/seed${SEED}
MONITOR=${ROOT}/metadata/${ARM}/seed${SEED}
test -f "${SOURCE}/checkpoints/1000.npz"
[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite ${RUN}" >&2; exit 2; }
[[ ! -e "${MONITOR}" ]] || { echo "refusing to overwrite ${MONITOR}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WSSR_MONITOR_DIR="${MONITOR}"
export WSSR_PAIRED_SEED="${SEED}"
export WSSR_ARM="${ARM}"
export WSSR_ETA_S="${ETA_S}"
export WSSR_REDUCED_METRIC_HISTORY_MODE=none
export WSSR_ANISOTROPIC_MATRIX_HISTORY="${ANISO_ENV}"

mkdir -p "$(dirname "${RUN}")" /scratch/dexuan1/runs/logs
cd /scratch/dexuan1/vmcnet
python experiments/C_current_subspace_spectral_history_E5000_20260812/training_launcher.py \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/1000.npz \
  --reload.new_optimizer_state=True --reload.reburn=False \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${RUN}" --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nchains=1000 --config.vmc.nburn=0 \
  --config.vmc.nepochs=5000 --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.check_for_nans=True --config.vmc.checkpoint_every=1000 \
  --config.vmc.best_checkpoint_every=5000 --config.eval.nepochs=0 \
  --config.vmc.optimizer_type=wssr_warm_svd_right \
  --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.04 \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_S="${ETA_S}" \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_g=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.reduced_metric_history_mode=none \
  --config.vmc.optimizer.wssr_warm_svd_right.anisotropic_matrix_history="${ANISO}" \
  --config.vmc.optimizer.wssr_warm_svd_right.anisotropic_matrix_history_noise_scale=1.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.relative_singular_value_cutoff=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.tikhonov_lambda=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.mixed_precision_solve=False \
  --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True \
  --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint_mode=euclidean \
  --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=800 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=800 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank=800 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=800 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=40 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 \
  --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=True \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first=False \
  --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=none \
  --config.wandb.mode=disabled
test -f "${RUN}/checkpoints/5000.npz"
test -f "${MONITOR}/monitor.csv"
