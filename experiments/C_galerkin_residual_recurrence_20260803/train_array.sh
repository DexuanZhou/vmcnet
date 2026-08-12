#!/bin/bash
#SBATCH --job-name=C-galerkin-rec
#SBATCH --account=def-ortner
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

test -n "${WSSR_MANIFEST:-}"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${WSSR_MANIFEST}")
IFS=$'\t' read -r ARM RANK SEED UPDATES MU RESIDUAL DUAL EF ETA_S <<< "${LINE}"
test -n "${ARM}"
if [[ "${DUAL}" == "1" ]]; then DUAL_BOOL=True; else DUAL_BOOL=False; fi
if [[ "${EF}" == "1" ]]; then EF_BOOL=True; else EF_BOOL=False; fi

SOURCE=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1
ROOT=/scratch/dexuan1/runs/C_galerkin_residual_recurrence_20260803
RUN=${ROOT}/train/${ARM}/seed${SEED}
MONITOR=${ROOT}/metadata_train/${ARM}/seed${SEED}
test -f "${SOURCE}/checkpoints/1000.npz"
[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite ${RUN}" >&2; exit 2; }
[[ ! -e "${MONITOR}" ]] || { echo "refusing to overwrite ${MONITOR}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WSSR_MONITOR_DIR="${MONITOR}"
export WSSR_ARM="${ARM}" WSSR_RANK="${RANK}" WSSR_SEED="${SEED}"
export WSSR_UPDATES="${UPDATES}" WSSR_MU="${MU}"
export WSSR_RESIDUAL_EVALUATION="${RESIDUAL}"
export WSSR_DUAL_DIAGNOSTICS="${DUAL}" WSSR_ERROR_FEEDBACK="${EF}"
export WSSR_SUBSPACE_ETA_S="${ETA_S}"

mkdir -p "$(dirname "${RUN}")" /scratch/dexuan1/runs/logs
cd /scratch/dexuan1/vmcnet
python experiments/C_galerkin_residual_recurrence_20260803/training_launcher.py \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/1000.npz \
  --reload.new_optimizer_state=True --reload.reburn=False \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${RUN}" --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nchains=1000 --config.vmc.nburn=0 \
  --config.vmc.nepochs="${UPDATES}" --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.check_for_nans=True --config.vmc.checkpoint_every=500 \
  --config.vmc.best_checkpoint_every="${UPDATES}" \
  --config.eval.nburn=0 --config.eval.nepochs=0 \
  --config.vmc.optimizer_type=wssr_warm_svd_right \
  --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.04 \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_S=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_g=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_bias_correction=False \
  --config.vmc.optimizer.wssr_warm_svd_right.solution_recurrence_mode=residual \
  --config.vmc.optimizer.wssr_warm_svd_right.solution_recurrence_mu="${MU}" \
  --config.vmc.optimizer.wssr_warm_svd_right.residual_evaluation="${RESIDUAL}" \
  --config.vmc.optimizer.wssr_warm_svd_right.residual_dual_mode_diagnostics="${DUAL_BOOL}" \
  --config.vmc.optimizer.wssr_warm_svd_right.solution_error_feedback="${EF_BOOL}" \
  --config.vmc.optimizer.wssr_warm_svd_right.error_feedback_norm_cap=10.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.subspace_eta_S="${ETA_S}" \
  --config.vmc.optimizer.wssr_warm_svd_right.recurrence_telemetry=True \
  --config.vmc.optimizer.wssr_warm_svd_right.drift_gate_monitoring=True \
  --config.vmc.optimizer.wssr_warm_svd_right.drift_gate_hypothetical_eta_g=0.8 \
  --config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.relative_singular_value_cutoff=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.tikhonov_lambda=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov \
  --config.vmc.optimizer.wssr_warm_svd_right.mixed_precision_solve=True \
  --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True \
  --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank="${RANK}" \
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

if (( UPDATES >= 500 )); then
  test -f "${RUN}/checkpoints/${UPDATES}.npz"
fi
test -f "${RUN}/training_metrics.csv"
test -f "${MONITOR}/monitor.csv"
test -f "${MONITOR}/early_stop.json"
