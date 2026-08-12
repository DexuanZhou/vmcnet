#!/bin/bash
#SBATCH --job-name=C-lazySSI-200
#SBATCH --account=def-ortner
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-6%4
#SBATCH --exclude=fc10404,fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

HERE=/scratch/dexuan1/vmcnet/experiments/C_lazy_ssi_rank200_400_20260808
CURRENT=/scratch/dexuan1/vmcnet_snapshot_C_lazy_ssi_20260808
REPO=/scratch/dexuan1/vmcnet
SOURCE=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1
ROOT=/scratch/dexuan1/runs/C_lazy_ssi_rank200_400_20260808
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

case "${SLURM_ARRAY_TASK_ID}" in
  0) ARM=spring; METHOD=spring; RANK=0; PERIOD=1; MODE=ritz ;;
  1) ARM=r200_k1; METHOD=wssr; RANK=200; PERIOD=1; MODE=ritz ;;
  2) ARM=r200_k10; METHOD=wssr; RANK=200; PERIOD=10; MODE=fixed_basis ;;
  3) ARM=r200_k20; METHOD=wssr; RANK=200; PERIOD=20; MODE=fixed_basis ;;
  4) ARM=r400_k1; METHOD=wssr; RANK=400; PERIOD=1; MODE=ritz ;;
  5) ARM=r400_k10; METHOD=wssr; RANK=400; PERIOD=10; MODE=fixed_basis ;;
  6) ARM=r400_k20; METHOD=wssr; RANK=400; PERIOD=20; MODE=fixed_basis ;;
  *) exit 2 ;;
esac

WORKTREE=${CURRENT}
RUN=${ROOT}/train/${ARM}
META=${ROOT}/metadata/${ARM}

[[ -d "${WORKTREE}" ]] || exit 2
[[ -f "${SOURCE}/checkpoints/1000.npz" ]] || exit 2
[[ ! -e "${RUN}" && ! -e "${META}" ]] || exit 2

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export PYTHONPATH="${WORKTREE}"
export GIT_DIR="${REPO}/.git"
export GIT_WORK_TREE="${WORKTREE}"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TIMING_MONITOR_DIR="${META}" TIMING_ARM="${ARM}" TIMING_SEED=0
mkdir -p "$(dirname "${RUN}")" "$(dirname "${META}")" \
  /scratch/dexuan1/runs/logs
cd "${WORKTREE}"

hostname > "${META}.hostname.tmp"
nvidia-smi > "${META}.nvidia_smi.tmp"

COMMON_ARGS=(
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
  --config.vmc.nepochs=200
  --config.vmc.nsteps_per_param_update=10
  --config.vmc.check_for_nans=True
  --config.vmc.disable_checkpointing=True
  --config.eval.nepochs=0
  --config.wandb.mode=disabled
)

if [[ "${METHOD}" == spring ]]; then
  "${VENV}/bin/python" "${HERE}/timing_launcher.py" \
    "${COMMON_ARGS[@]}" \
    --config.vmc.optimizer_type=spring \
    --config.vmc.optimizer.spring.schedule_type=inverse_time \
    --config.vmc.optimizer.spring.learning_rate=0.02 \
    --config.vmc.optimizer.spring.learning_decay_rate=0.0001 \
    --config.vmc.optimizer.spring.damping=0.001 \
    --config.vmc.optimizer.spring.mu=0.99 \
    --config.vmc.optimizer.spring.constrain_norm=True \
    --config.vmc.optimizer.spring.norm_constraint=0.001
else
  "${VENV}/bin/python" "${HERE}/timing_launcher.py" \
    "${COMMON_ARGS[@]}" \
    --config.vmc.optimizer_type=wssr_warm_svd_right \
    --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time \
    --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.04 \
    --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001 \
    --config.vmc.optimizer.wssr_warm_svd_right.eta=0.0 \
    --config.vmc.optimizer.wssr_warm_svd_right.eta_S=0.0 \
    --config.vmc.optimizer.wssr_warm_svd_right.eta_g=0.0 \
    --config.vmc.optimizer.wssr_warm_svd_right.subspace_eta_S=0.0 \
    --config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003 \
    --config.vmc.optimizer.wssr_warm_svd_right.relative_singular_value_cutoff=0.0003 \
    --config.vmc.optimizer.wssr_warm_svd_right.tikhonov_lambda=0.001 \
    --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov \
    --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0.0 \
    --config.vmc.optimizer.wssr_warm_svd_right.mixed_precision_solve=True \
    --config.vmc.optimizer.wssr_warm_svd_right.solution_recurrence_mode=residual \
    --config.vmc.optimizer.wssr_warm_svd_right.solution_recurrence_mu=0.99 \
    --config.vmc.optimizer.wssr_warm_svd_right.residual_evaluation=full_current_batch \
    --config.vmc.optimizer.wssr_warm_svd_right.galerkin_solve_backend=device_cholesky \
    --config.vmc.optimizer.wssr_warm_svd_right.solution_error_feedback=False \
    --config.vmc.optimizer.wssr_warm_svd_right.recurrence_telemetry=True \
    --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True \
    --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001 \
    --config.vmc.optimizer.wssr_warm_svd_right.sr_rank="${RANK}" \
    --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max="${RANK}" \
    --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank="${RANK}" \
    --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank="${RANK}" \
    --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=40 \
    --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 \
    --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=True \
    --config.vmc.optimizer.wssr_warm_svd_right.subspace_refresh_period="${PERIOD}" \
    --config.vmc.optimizer.wssr_warm_svd_right.subspace_refresh_mode="${MODE}" \
    --config.vmc.optimizer.wssr_warm_svd_right.exact_first=False \
    --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=none
fi

mv "${META}.hostname.tmp" "${META}/hostname.txt"
mv "${META}.nvidia_smi.tmp" "${META}/nvidia_smi_start.txt"
[[ -f "${META}/monitor.csv" ]] || exit 4
