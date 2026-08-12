#!/bin/bash
#SBATCH --job-name=wssr-n4096-smoke
#SBATCH --account=def-ortner
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-5%2
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err
set -euo pipefail

REPO=/scratch/dexuan1/vmcnet
PKG=${REPO}/experiments/fir_n4096_smoke
MANIFEST=${PKG}/manifest.csv
RUN_ROOT=/scratch/dexuan1/runs/fir_n4096_smoke
VMCNET_VENV="${VMCNET_VENV:-/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311}"

eval "$(python - "${MANIFEST}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import csv, shlex, sys
rows=list(csv.DictReader(open(sys.argv[1])))
r=next(x for x in rows if x['task_id']==sys.argv[2])
for key in ('system','rank','warm','source_dir','source_epoch','eta','learning_rate','ion_pos','ion_charges','nelec','seed'):
    print(f'{key.upper()}={shlex.quote(r[key])}')
PY
)"

RUN_NAME="${SYSTEM}_rank${RANK}_warm${WARM}_n4096_e50"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
SOURCE_CHECKPOINT="${SOURCE_DIR}/checkpoints/${SOURCE_EPOCH}.npz"
[[ -f "${SOURCE_CHECKPOINT}" ]] || { echo "Missing source checkpoint: ${SOURCE_CHECKPOINT}" >&2; exit 2; }
[[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to overwrite: ${RUN_DIR}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VMCNET_VENV}/bin/activate"
export TMPDIR="${SLURM_TMPDIR:-/scratch/dexuan1/tmp}"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export VMCNET_DISABLE_CHECKPOINTS=1
export SMOKE_TIMING_DIR="${RUN_DIR}"
mkdir -p "${TMPDIR}" "${RUN_DIR}" /scratch/dexuan1/runs/logs
cd "${REPO}"

date +%s > "${RUN_DIR}/job_start_unix.txt"
hostname > "${RUN_DIR}/hostname.txt"
printf '%s\n' "${SLURM_JOB_ID}" > "${RUN_DIR}/slurm_job_id.txt"
printf '%s\n' "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}" > "${RUN_DIR}/slurm_array_job_id.txt"
printf '%s\n' "${SLURM_ARRAY_TASK_ID}" > "${RUN_DIR}/slurm_array_task_id.txt"
scontrol show job "${SLURM_JOB_ID}" > "${RUN_DIR}/slurm_job.txt"
nvidia-smi > "${RUN_DIR}/nvidia_smi_start.txt"
nvidia-smi --query-gpu=name,memory.total,mig.mode.current --format=csv,noheader,nounits > "${RUN_DIR}/gpu_identity.csv"
GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
MIG_MODE=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | head -1)
[[ "${GPU_TOTAL}" -ge 80000 ]] || { echo "GPU has only ${GPU_TOTAL} MiB" >&2; exit 3; }
[[ "${MIG_MODE}" != *Enabled* ]] || { echo "MIG is enabled; refusing shared/MIG GPU" >&2; exit 3; }
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
printf '%s\n' "${SOURCE_CHECKPOINT}" > "${RUN_DIR}/source_checkpoint.txt"
python - <<'PY' > "${RUN_DIR}/versions.txt"
import platform, jax, jaxlib
print('python='+platform.python_version())
print('jax='+jax.__version__)
print('jaxlib='+jaxlib.__version__)
PY

echo 'timestamp_unix,gpu_name,memory_total_mib,memory_used_mib,utilization_gpu_percent' > "${RUN_DIR}/gpu_memory_poll.csv"
( while true; do
    TS=$(date +%s.%N)
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -v ts="${TS}" -F', ' '{print ts "," $1 "," $2 "," $3 "," $4}' >> "${RUN_DIR}/gpu_memory_poll.csv" 2>/dev/null || true
    sleep 1
  done ) &
GPU_MON_PID=$!
cleanup() { kill "${GPU_MON_PID}" 2>/dev/null || true; wait "${GPU_MON_PID}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

START=$(date +%s)
set +e
python "${PKG}/timed_launcher.py" \
  --reload.logdir="${SOURCE_DIR}" \
  --reload.checkpoint_relative_file_path="checkpoints/${SOURCE_EPOCH}.npz" \
  --reload.use_config_file=True --reload.use_checkpoint_file=True \
  --reload.new_optimizer_state=True --reload.reburn=True \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${RUN_DIR}" --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.initial_seed="${SEED}" --config.distribute=False \
  --config.problem.ion_pos="${ION_POS}" --config.problem.ion_charges="${ION_CHARGES}" --config.problem.nelec="${NELEC}" \
  --config.eval.nepochs=0 --config.wandb.mode=disabled \
  --config.vmc.nchains=4096 --config.vmc.nburn=5000 --config.vmc.nepochs=50 \
  --config.vmc.nsteps_per_param_update=10 --config.vmc.nmoves_per_width_update=100 --config.vmc.std_move=0.25 \
  --config.vmc.clip_center=mean --config.vmc.clip_threshold=5.0 \
  --config.vmc.check_for_nans=True --config.vmc.nan_safe=True --config.vmc.disable_checkpointing=True \
  --config.vmc.optimizer_type=wssr_warm_svd_right \
  --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_rate="${LEARNING_RATE}" \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta="${ETA}" \
  --config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True \
  --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=False \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=8 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm="${WARM}"
STATUS=$?
set -e
END=$(date +%s)
cleanup
trap - EXIT INT TERM
printf 'start_unix,end_unix,elapsed_seconds,exit_status\n%s,%s,%s,%s\n' "${START}" "${END}" "$((END-START))" "${STATUS}" > "${RUN_DIR}/run_timing.csv"
find "${RUN_DIR}" -type f -name '*.npz' -printf '%p\n' > "${RUN_DIR}/npz_files.txt"
exit "${STATUS}"
