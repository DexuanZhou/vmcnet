#!/bin/bash
#SBATCH --job-name=C-largeN-cost
#SBATCH --account=def-ortner
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-7%8
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

REPO=/scratch/dexuan1/vmcnet
SOURCE=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1
ROOT=${TIMING_ROOT:-/scratch/dexuan1/runs/C_zero_cost_triage_20260804/task2_largeN_timing_retry2}
MANIFEST=${TIMING_MANIFEST:-${REPO}/experiments/C_zero_cost_triage_20260804/task2_largeN_timing/manifest.tsv}
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${MANIFEST}")
IFS=$'\t' read -r METHOD RANK WALKERS <<< "${LINE}"
ARM=${METHOD}_r${RANK}_n${WALKERS}
RUN=${ROOT}/runs/${ARM}
META=${ROOT}/metadata/${ARM}

test -f "${SOURCE}/checkpoints/1000.npz"
[[ ! -e "${RUN}" && ! -e "${META}" ]] || {
  echo "refusing to overwrite ${RUN} or ${META}" >&2
  exit 2
}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false VMCNET_DISABLE_CHECKPOINTS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TIMING_META_DIR="${META}"

mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd "${REPO}"
hostname > "${META}/hostname.txt"
nvidia-smi --query-gpu=name,memory.total,mig.mode.current,driver_version --format=csv,noheader,nounits > "${META}/gpu_identity.csv"
git rev-parse HEAD > "${META}/git_commit.txt"
git diff -- vmcnet/updates/wssr.py | sha256sum > "${META}/wssr_diff_sha256.txt"
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"
cat > "${META}/protocol.txt" <<EOF
purpose=timing_only_no_scientific_training_value
system=C
source=${SOURCE}/checkpoints/1000.npz
method=${METHOD}
rank=${RANK}
walkers=${WALKERS}
mcmc_steps=10
warmup_epochs=20
timed_epochs=21-${TIMING_EPOCHS:-105}
profile_epochs=${TIMING_PROFILE_START:-101}-$(( ${TIMING_PROFILE_START:-101} + 4 ))
w_cache=disabled_pre_optimization
spring_learning_rate=${SPRING_LR:-0.02}
galerkin_solve_backend=${GALERKIN_SOLVE_BACKEND:-host_fp64}
EOF

echo 'timestamp_unix,memory_used_mib,utilization_gpu_percent' > "${META}/gpu_memory_poll.csv"
( while true; do
    TS=$(date +%s.%N)
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
      | awk -v ts="${TS}" -F', ' '{print ts "," $1 "," $2}' \
      >> "${META}/gpu_memory_poll.csv" 2>/dev/null || true
    sleep 0.2
  done ) &
MON=$!
cleanup() { kill "${MON}" 2>/dev/null || true; wait "${MON}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

COMMON=(
  --reload.logdir="${SOURCE}" --reload.use_config_file=True
  --reload.use_checkpoint_file=True
  --reload.checkpoint_relative_file_path=checkpoints/1000.npz
  --reload.new_optimizer_state=True --reload.reburn=False
  --reload.append=False --reload.same_logdir=False
  --config.logdir="${RUN}" --config.base_logdir="${RUN}"
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE
  --config.vmc.nchains="${WALKERS}" --config.vmc.nburn=0
  --config.vmc.nepochs="${TIMING_EPOCHS:-105}" --config.vmc.nsteps_per_param_update=10
  --config.vmc.clip_center=mean --config.vmc.clip_threshold=5.0
  --config.vmc.check_for_nans=True --config.vmc.disable_checkpointing=True
  --config.eval.nburn=0 --config.eval.nepochs=0 --config.wandb.mode=disabled
)

if [[ "${METHOD}" == spring ]]; then
  OPT=(
    --config.vmc.optimizer_type=spring
    --config.vmc.optimizer.spring.schedule_type=inverse_time
    --config.vmc.optimizer.spring.learning_rate="${SPRING_LR:-0.02}"
    --config.vmc.optimizer.spring.learning_decay_rate=0.0001
    --config.vmc.optimizer.spring.mu=0.99
    --config.vmc.optimizer.spring.damping=0.001
    --config.vmc.optimizer.spring.constrain_norm=True
    --config.vmc.optimizer.spring.norm_constraint=0.001
  )
else
  OPT=(
    --config.vmc.optimizer_type=wssr_warm_svd_right
    --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time
    --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.04
    --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001
    --config.vmc.optimizer.wssr_warm_svd_right.eta=0.0
    --config.vmc.optimizer.wssr_warm_svd_right.eta_S=0.0
    --config.vmc.optimizer.wssr_warm_svd_right.eta_g=0.0
    --config.vmc.optimizer.wssr_warm_svd_right.solution_recurrence_mode=residual
    --config.vmc.optimizer.wssr_warm_svd_right.solution_recurrence_mu=0.99
    --config.vmc.optimizer.wssr_warm_svd_right.residual_evaluation=full_current_batch
    --config.vmc.optimizer.wssr_warm_svd_right.galerkin_solve_backend="${GALERKIN_SOLVE_BACKEND:-host_fp64}"
    --config.vmc.optimizer.wssr_warm_svd_right.residual_dual_mode_diagnostics=False
    --config.vmc.optimizer.wssr_warm_svd_right.solution_error_feedback=False
    --config.vmc.optimizer.wssr_warm_svd_right.subspace_eta_S=0.0
    --config.vmc.optimizer.wssr_warm_svd_right.recurrence_telemetry=False
    --config.vmc.optimizer.wssr_warm_svd_right.drift_gate_monitoring=False
    --config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003
    --config.vmc.optimizer.wssr_warm_svd_right.relative_singular_value_cutoff=0.0003
    --config.vmc.optimizer.wssr_warm_svd_right.tikhonov_lambda=0.001
    --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov
    --config.vmc.optimizer.wssr_warm_svd_right.mixed_precision_solve=True
    --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True
    --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001
    --config.vmc.optimizer.wssr_warm_svd_right.sr_rank="${RANK}"
    --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max="${RANK}"
    --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank="${RANK}"
    --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank="${RANK}"
    --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=40
    --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2
    --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=True
    --config.vmc.optimizer.wssr_warm_svd_right.exact_first=False
    --config.vmc.optimizer.wssr_warm_svd_right.exact_reference_diagnostics=False
    --config.vmc.optimizer.wssr_warm_svd_right.update_diagnostics=False
    --config.vmc.optimizer.wssr_warm_svd_right.reliability_diagnostics=False
    --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=none
    --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0.0
    --config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_beta=0.0
  )
fi

set +e
python experiments/C_zero_cost_triage_20260804/task2_largeN_timing/timing_launcher.py \
  "${COMMON[@]}" "${OPT[@]}"
STATUS=$?
set -e
cleanup; trap - EXIT INT TERM
printf '%s\n' "${STATUS}" > "${META}/exit_status.txt"
exit "${STATUS}"
