#!/bin/bash
# Execution template sourced by an instantiated pilot Slurm script.
set -euo pipefail

: "${SYSTEM:?SYSTEM is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${PRE_DIR:?PRE_DIR is required}"
: "${PRE_EPOCH:?PRE_EPOCH is required}"
: "${LEARNING_RATE:?LEARNING_RATE is required}"
: "${MU:?MU is required}"
: "${NORM_CONSTRAINT:?NORM_CONSTRAINT is required}"
: "${ION_POS:?ION_POS is required}"

REPO=/scratch/dexuan1/vmcnet
RUN_ROOT="${RUN_ROOT:-/scratch/dexuan1/runs/pilot}"
RUN_DIR="${RUN_ROOT}/${SYSTEM}/${RUN_NAME}"
METADATA_TMP="${RUN_DIR}.run_metadata.json.tmp"
GPUMEM_TMP="${RUN_DIR}.gpumem.csv.tmp"
VMCNET_VENV="${VMCNET_VENV:-/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311}"
NEPOCHS="${NEPOCHS:-500}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
INITIAL_SEED="${INITIAL_SEED:-0}"
REBURN="${REBURN:-True}"
[[ -f "${PRE_DIR}/checkpoints/${PRE_EPOCH}.npz" ]] || { echo "Missing pre checkpoint: ${PRE_DIR}/checkpoints/${PRE_EPOCH}.npz" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VMCNET_VENV}/bin/activate"
export TMPDIR="${SLURM_TMPDIR:-/scratch/dexuan1/tmp}"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${TMPDIR}" "$(dirname "${RUN_DIR}")"
cd "${REPO}"

[[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to reuse existing run directory: ${RUN_DIR}" >&2; exit 2; }
rm -f "${METADATA_TMP}" "${GPUMEM_TMP}"
cat > "${METADATA_TMP}" <<EOF
{"system":"${SYSTEM}","run_name":"${RUN_NAME}","family":"spring","nepochs":${NEPOCHS},"initial_seed":${INITIAL_SEED},"reburn":${REBURN,,},"nchains":1000,"nburn":5000,"nsteps_per_param_update":10,"clip_threshold":5.0,"schedule_type":"inverse_time","learning_rate":${LEARNING_RATE},"learning_decay_rate":0.0001,"mu":${MU},"constrain_norm":true,"norm_constraint":${NORM_CONSTRAINT},"ion_pos":"${ION_POS}","reload":"${PRE_DIR}/checkpoints/${PRE_EPOCH}.npz"}
EOF
echo 'timestamp,index,memory_used_mib' > "${GPUMEM_TMP}"
( while true; do nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits >> "${GPUMEM_TMP}" 2>/dev/null || true; sleep 10; done ) &
GPU_MON_PID=$!
cleanup_monitor() { kill "${GPU_MON_PID}" 2>/dev/null || true; wait "${GPU_MON_PID}" 2>/dev/null || true; }
trap cleanup_monitor EXIT INT TERM

START_UNIX=$(date +%s)
set +e
vmc-molecule \
  --reload.logdir="${PRE_DIR}" \
  --reload.checkpoint_relative_file_path="checkpoints/${PRE_EPOCH}.npz" \
  --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.new_optimizer_state=True \
  --reload.append=False \
  --reload.reburn="${REBURN}" \
  --config.logdir="${RUN_DIR}" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name=NONE \
  --config.initial_seed="${INITIAL_SEED}" \
  --config.problem.ion_pos="${ION_POS}" \
  --config.eval.nepochs=0 \
  --config.wandb.mode=disabled \
  --config.vmc.nchains=1000 \
  --config.vmc.nburn=5000 \
  --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.clip_threshold=5.0 \
  --config.vmc.clip_center=mean \
  --config.vmc.nepochs="${NEPOCHS}" \
  --config.vmc.checkpoint_every="${CHECKPOINT_EVERY}" \
  --config.vmc.best_checkpoint_every="${CHECKPOINT_EVERY}" \
  --config.vmc.optimizer_type=spring \
  --config.vmc.optimizer.spring.schedule_type=inverse_time \
  --config.vmc.optimizer.spring.learning_rate="${LEARNING_RATE}" \
  --config.vmc.optimizer.spring.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.spring.mu="${MU}" \
  --config.vmc.optimizer.spring.constrain_norm=True \
  --config.vmc.optimizer.spring.norm_constraint="${NORM_CONSTRAINT}"
STATUS=$?
set -e
END_UNIX=$(date +%s)
cleanup_monitor
trap - EXIT INT TERM
mkdir -p "${RUN_DIR}"
mv "${METADATA_TMP}" "${RUN_DIR}/run_metadata.json"
mv "${GPUMEM_TMP}" "${RUN_DIR}/gpumem.csv"
printf 'start_unix,end_unix,elapsed_seconds,nepochs,status\n%s,%s,%s,%s,%s\n' "${START_UNIX}" "${END_UNIX}" "$((END_UNIX - START_UNIX))" "${NEPOCHS}" "${STATUS}" > "${RUN_DIR}/run_timing.csv"
if [[ ${STATUS} -eq 0 ]]; then
  rm -f "${RUN_DIR}/best_checkpoint.npz"
  find "${RUN_DIR}/checkpoints" -maxdepth 1 -type f -name '*.npz' ! -name "${NEPOCHS}.npz" -delete
fi
exit "${STATUS}"
