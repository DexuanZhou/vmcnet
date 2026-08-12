#!/bin/bash
#SBATCH --job-name=N2-r200-recovery
#SBATCH --account=def-ortner
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-3%4
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

HERE=/scratch/dexuan1/vmcnet/experiments/N2_R2016_rank200_recovery_E5000_20260808
SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_N2_rank200_recovery_20260808
REPO=/scratch/dexuan1/vmcnet
SOURCE=/scratch/dexuan1/runs/old/N2_R2016/e100000_followup/N2eq/N2eq_kfac_pre5000
ROOT=/scratch/dexuan1/runs/N2_R2016_rank200_recovery_E5000_20260808
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

case "${SLURM_ARRAY_TASK_ID}" in
  0) ARM=A_hard_rank200; MODE=standard; COMPLEMENT=0.0; ETA_G=0.0 ;;
  1) ARM=B_isotropic_floor; MODE=standard; COMPLEMENT=0.0001; ETA_G=0.0 ;;
  2) ARM=C_rotating_EF; MODE=error_feedback; COMPLEMENT=0.0; ETA_G=0.0 ;;
  3) ARM=D_momentum_floor; MODE=standard; COMPLEMENT=0.0001; ETA_G=0.9 ;;
  *) exit 2 ;;
esac

RUN=${ROOT}/train/${ARM}
META=${ROOT}/metadata/train/${ARM}
[[ -d "${SNAPSHOT}" ]] || { echo "missing snapshot ${SNAPSHOT}" >&2; exit 2; }
[[ -f "${SOURCE}/checkpoints/5000.npz" ]] || exit 2
[[ ! -e "${RUN}" && ! -e "${META}" ]] || {
  echo "refusing to overwrite ${RUN} or ${META}" >&2
  exit 2
}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export PYTHONPATH="${SNAPSHOT}"
export GIT_DIR="${REPO}/.git"
export GIT_WORK_TREE="${SNAPSHOT}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd "${SNAPSHOT}"

hostname > "${META}/hostname.txt"
nvidia-smi > "${META}/nvidia_smi_start.txt"
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"
cp "${HERE}/manifest.tsv" "${META}/manifest.tsv"
sha256sum vmcnet/updates/wssr.py vmcnet/train/default_config.py \
  > "${META}/source_sha256.txt"

(
  while true; do
    nvidia-smi --query-compute-apps=timestamp,pid,used_memory \
      --format=csv,noheader,nounits || true
    sleep 2
  done
) > "${META}/gpu_memory_mib.csv" &
MONITOR_PID=$!
cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

COMMON_ARGS=(
  --reload.logdir="${SOURCE}"
  --reload.use_config_file=True
  --reload.use_checkpoint_file=True
  --reload.checkpoint_relative_file_path=checkpoints/5000.npz
  --reload.new_optimizer_state=True
  --reload.reburn=False
  --reload.append=False
  --reload.same_logdir=False
  --config.logdir="${RUN}"
  --config.base_logdir="${RUN}"
  --config.save_to_current_datetime_subfolder=False
  --config.subfolder_name=NONE
  --config.problem.ion_pos='((0.,0.,-1.008),(0.,0.,1.008))'
  --config.problem.ion_charges='(7.,7.)'
  --config.problem.nelec='(7,7)'
  --config.vmc.nchains=1000
  --config.vmc.nepochs=5000
  --config.vmc.nsteps_per_param_update=10
  --config.vmc.clip_threshold=5.0
  --config.vmc.clip_center=mean
  --config.vmc.check_for_nans=True
  --config.vmc.disable_checkpointing=False
  --config.vmc.checkpoint_every=1000
  --config.vmc.best_checkpoint_every=1000
  --config.eval.nepochs=0
  --config.vmc.optimizer_type=wssr_warm_svd_right
  --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time
  --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.002
  --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001
  --config.vmc.optimizer.wssr_warm_svd_right.eta=0.0
  --config.vmc.optimizer.wssr_warm_svd_right.eta_S=0.0
  --config.vmc.optimizer.wssr_warm_svd_right.eta_g="${ETA_G}"
  --config.vmc.optimizer.wssr_warm_svd_right.enable_gradient_transport=False
  --config.vmc.optimizer.wssr_warm_svd_right.subspace_eta_S=0.0
  --config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003
  --config.vmc.optimizer.wssr_warm_svd_right.relative_singular_value_cutoff=0.0003
  --config.vmc.optimizer.wssr_warm_svd_right.tikhonov_lambda=0.001
  --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov
  --config.vmc.optimizer.wssr_warm_svd_right.complement_weight="${COMPLEMENT}"
  --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True
  --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=200
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=200
  --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank=200
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=200
  --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=True
  --config.vmc.optimizer.wssr_warm_svd_right.semi_matrix_free_augmented=False
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=40
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first=False
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first_force=False
  --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=none
  --config.wandb.mode=disabled
)

if [[ "${MODE}" == error_feedback ]]; then
  vmc-molecule "${COMMON_ARGS[@]}" \
    --config.vmc.optimizer.wssr_warm_svd_right.mixed_precision_solve=True \
    --config.vmc.optimizer.wssr_warm_svd_right.solution_recurrence_mode=residual \
    --config.vmc.optimizer.wssr_warm_svd_right.solution_recurrence_mu=0.95 \
    --config.vmc.optimizer.wssr_warm_svd_right.residual_evaluation=full_current_batch \
    --config.vmc.optimizer.wssr_warm_svd_right.galerkin_solve_backend=device_cholesky \
    --config.vmc.optimizer.wssr_warm_svd_right.solution_error_feedback=True \
    --config.vmc.optimizer.wssr_warm_svd_right.error_feedback_decay=0.95 \
    --config.vmc.optimizer.wssr_warm_svd_right.error_feedback_norm_cap=10.0 \
    --config.vmc.optimizer.wssr_warm_svd_right.error_feedback_cap_reference=sample_residual \
    --config.vmc.optimizer.wssr_warm_svd_right.recurrence_telemetry=True
else
  vmc-molecule "${COMMON_ARGS[@]}" \
    --config.vmc.optimizer.wssr_warm_svd_right.mixed_precision_solve=False \
    --config.vmc.optimizer.wssr_warm_svd_right.solution_recurrence_mode=none
fi

cleanup
trap - EXIT INT TERM
[[ -f "${RUN}/checkpoints/1000.npz" ]] || exit 4
[[ -f "${RUN}/checkpoints/5000.npz" ]] || exit 4
nvidia-smi > "${META}/nvidia_smi_end.txt"
