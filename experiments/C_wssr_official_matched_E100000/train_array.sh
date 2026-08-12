#!/bin/bash
#SBATCH --job-name=C-wssr-matched-100k
#SBATCH --account=def-ortner
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --array=0-2%3
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

REPO=/scratch/dexuan1/vmcnet
SOURCE=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1
CHECKPOINT=${SOURCE}/checkpoints/1000.npz
ROOT=/scratch/dexuan1/runs/C_wssr_official_matched_E100000
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    VARIANT=A_rank400_hard_exactfirst
    RANK=400
    MODE=none
    BETA=0.0
    COMPLEMENT=0.0
    ;;
  1)
    VARIANT=B_rank400_adaptive_beta02_exactfirst
    RANK=400
    MODE=adaptive_complement
    BETA=0.2
    COMPLEMENT=0.0001
    ;;
  2)
    VARIANT=C_rank800_hard_exactfirst
    RANK=800
    MODE=none
    BETA=0.0
    COMPLEMENT=0.0
    ;;
  *)
    echo "invalid array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

RANK=${RANK_OVERRIDE:-${RANK}}
BETA=${BETA_OVERRIDE:-${BETA}}
RUN_SUFFIX=${RUN_SUFFIX:-}
UPDATE_DIAGNOSTICS=${UPDATE_DIAGNOSTICS:-True}
EXACT_FIRST=${EXACT_FIRST:-True}
SVD_INITIAL=${SVD_INITIAL:-8}
VARIANT=${VARIANT_OVERRIDE:-${VARIANT}}
RUN=${ROOT}/${VARIANT}${RUN_SUFFIX}
META=${RUN}_metadata
[[ -f "${CHECKPOINT}" ]] || { echo "missing ${CHECKPOINT}" >&2; exit 2; }
[[ ! -e "${RUN}" && ! -e "${META}" ]] || {
  echo "refusing to overwrite ${RUN} or ${META}" >&2
  exit 2
}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export VMCNET_PROFILE_TIMING=1
export SMOKE_TIMING_DIR="${META}"

mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd "${REPO}"
hostname > "${META}/hostname.txt"
source experiments/fir_n4096_smoke/scripts/gpu_health_check.sh \
  "${META}/nvidia_smi_start.txt"
python - <<'PY' > "${META}/jax_backend.txt"
import jax
print(jax.default_backend())
print(jax.devices())
assert jax.default_backend() == "gpu"
PY
git rev-parse HEAD > "${META}/git_commit.txt"
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"

cat > "${META}/protocol.txt" <<EOF
system=C
electron_configuration=(4,2)
source_checkpoint=${CHECKPOINT}
checkpoint_internal_epoch=999
preliminary_optimizer=kfac
preliminary_epochs=1000
training_nchains=1000
training_epochs=100000
optimizer=wssr_warm_svd_right
rank=${RANK}
all_rank_fields=${RANK}
exact_first=${EXACT_FIRST}
svd_maxiter_initial=${SVD_INITIAL}
svd_maxiter_warm=2
experimental_mode=${MODE}
adaptive_complement_beta=${BETA}
complement_weight=${COMPLEMENT}
update_diagnostics=${UPDATE_DIAGNOSTICS}
learning_rate=0.02
eta=0.8
damping=0.0003
norm_constraint=0.001
schedule_type=inverse_time
learning_decay_rate=0.0001
reload_new_optimizer_state=True
reload_reburn=False
evaluation_nchains=2000
evaluation_burn=5000
evaluation_epochs=20000
nsteps_per_param_update=10
EOF

echo 'timestamp,memory_used_mib,utilization_gpu_percent' \
  > "${META}/gpu_memory_poll.csv"
( while true; do
    nvidia-smi \
      --query-gpu=timestamp,memory.used,utilization.gpu \
      --format=csv,noheader,nounits >> "${META}/gpu_memory_poll.csv" || true
    sleep 1
  done ) &
MON=$!
cleanup() {
  kill "${MON}" 2>/dev/null || true
  wait "${MON}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

vmc-molecule \
  --reload.logdir="${SOURCE}" \
  --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/1000.npz \
  --reload.new_optimizer_state=True \
  --reload.reburn=False \
  --reload.append=False \
  --reload.same_logdir=False \
  --config.logdir="${RUN}" \
  --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name=NONE \
  --config.problem.nelec='(4,2)' \
  --config.vmc.nchains=1000 \
  --config.vmc.nburn=5000 \
  --config.vmc.nepochs=100000 \
  --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.clip_center=mean \
  --config.vmc.clip_threshold=5.0 \
  --config.vmc.check_for_nans=True \
  --config.vmc.checkpoint_every=5000 \
  --config.vmc.best_checkpoint_every=5000 \
  --config.eval.use_data_from_training=False \
  --config.eval.nchains=2000 \
  --config.eval.nburn=5000 \
  --config.eval.nepochs=20000 \
  --config.eval.nsteps_per_param_update=10 \
  --config.eval.record_local_energies=True \
  --config.vmc.optimizer_type=wssr_warm_svd_right \
  --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.02 \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta=0.8 \
  --config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov \
  --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True \
  --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=True \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial="${SVD_INITIAL}" \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first="${EXACT_FIRST}" \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first_force=False \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_reference_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.update_diagnostics="${UPDATE_DIAGNOSTICS}" \
  --config.vmc.optimizer.wssr_warm_svd_right.burst_diagnostics_payload=False \
  --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode="${MODE}" \
  --config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_beta="${BETA}" \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_weight="${COMPLEMENT}" \
  --config.wandb.mode=disabled

cleanup
trap - EXIT INT TERM

[[ -f "${RUN}/checkpoints/100000.npz" ]] || {
  echo "missing final training checkpoint" >&2
  exit 4
}
[[ -f "${RUN}/eval/statistics.json" ]] || {
  echo "missing frozen-evaluation statistics" >&2
  exit 5
}
[[ -f "${RUN}/eval/local_energies.txt" ]] || {
  echo "missing frozen local energies" >&2
  exit 5
}
