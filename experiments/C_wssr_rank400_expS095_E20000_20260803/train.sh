#!/bin/bash
#SBATCH --job-name=C-r400-expS-E20k
#SBATCH --account=def-ortner
#SBATCH --time=03:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

SOURCE=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1
ROOT=/scratch/dexuan1/runs/C_wssr_rank400_expS095_E20000_20260803
RUN=${ROOT}/train/seed0
MONITOR=${ROOT}/metadata/seed0
test -f "${SOURCE}/checkpoints/1000.npz"
[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite ${RUN}" >&2; exit 2; }
[[ ! -e "${MONITOR}" ]] || { echo "refusing to overwrite ${MONITOR}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WSSR_MONITOR_DIR="${MONITOR}"
export WSSR_PAIRED_SEED=0

mkdir -p "$(dirname "${RUN}")" /scratch/dexuan1/runs/logs
cd /scratch/dexuan1/vmcnet

python experiments/C_wssr_rank400_expS095_E20000_20260803/training_launcher.py \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/1000.npz \
  --reload.new_optimizer_state=True --reload.reburn=False \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${RUN}" --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nchains=1000 --config.vmc.nburn=0 \
  --config.vmc.nepochs=20000 --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.check_for_nans=True \
  --config.vmc.checkpoint_every=1000 --config.vmc.best_checkpoint_every=5000 \
  --config.eval.nepochs=0 \
  --config.vmc.optimizer_type=wssr_warm_svd_right \
  --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.04 \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_S=-1.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_g=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.adaptive_S_average=True \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_S_schedule=exponential_growth \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_S_max=0.95 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_S_warmup_steps=1000 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_S_tau=1000.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_bias_correction=False \
  --config.vmc.optimizer.wssr_warm_svd_right.enable_gradient_transport=False \
  --config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.relative_singular_value_cutoff=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.tikhonov_lambda=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov \
  --config.vmc.optimizer.wssr_warm_svd_right.mixed_precision_solve=True \
  --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True \
  --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=400 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=400 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank=400 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=400 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=40 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 \
  --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=True \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first=False \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_reference_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.update_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.reliability_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=none \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_beta=0.0 \
  --config.wandb.mode=disabled

test -f "${RUN}/checkpoints/20000.npz"
test -f "${RUN}/training_metrics.csv"
test -f "${MONITOR}/early_stop.json"
