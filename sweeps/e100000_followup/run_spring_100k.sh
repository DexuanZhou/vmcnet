#!/bin/bash
# Common sourced runner. This file is not submitted directly.
set -euo pipefail
: "${SYSTEM:?}" "${RUN_NAME:?}" "${SOURCE_DIR:?}" "${SOURCE_EPOCH:?}"
: "${RESUME_OPTIMIZER:?}" "${LEARNING_RATE:?}" "${ION_POS:?}" "${ION_CHARGES:?}" "${NELEC:?}"
REPO=/scratch/dexuan1/vmcnet
RUN_ROOT=/scratch/dexuan1/runs/e100000_followup
RUN_DIR="${RUN_ROOT}/${SYSTEM}/${RUN_NAME}"
NEPOCHS=100000; CHECKPOINT_EVERY=100000; MU=0.99; DAMPING=0.001; NORM_CONSTRAINT=0.001
VMCNET_VENV="${VMCNET_VENV:-/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311}"
SOURCE_CHECKPOINT="${SOURCE_DIR}/checkpoints/${SOURCE_EPOCH}.npz"
[[ -f "${SOURCE_CHECKPOINT}" ]] || { echo "Missing required checkpoint: ${SOURCE_CHECKPOINT}" >&2; exit 2; }
[[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to reuse ${RUN_DIR}" >&2; exit 2; }
if [[ "${RESUME_OPTIMIZER}" == True ]]; then NEW_OPTIMIZER_STATE=False; REBURN=False; APPEND=True; else NEW_OPTIMIZER_STATE=True; REBURN=True; APPEND=False; fi
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VMCNET_VENV}/bin/activate"
export TMPDIR="${SLURM_TMPDIR:-/scratch/dexuan1/tmp}"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${TMPDIR}" "$(dirname "${RUN_DIR}")"; cd "${REPO}"
METADATA_TMP="${RUN_DIR}.run_metadata.json.tmp"; GPUMEM_TMP="${RUN_DIR}.gpumem.csv.tmp"
cat > "${METADATA_TMP}" <<EOF
{"system":"${SYSTEM}","run_name":"${RUN_NAME}","family":"spring","target_epoch":${NEPOCHS},"source_checkpoint":"${SOURCE_CHECKPOINT}","resume_optimizer":${RESUME_OPTIMIZER,,},"new_optimizer_state":${NEW_OPTIMIZER_STATE,,},"reburn":${REBURN,,},"append_history":${APPEND,,},"nchains":1000,"nburn":5000,"nsteps_per_param_update":10,"clip_center":"mean","clip_threshold":5.0,"eval_nepochs":0,"schedule_type":"inverse_time","learning_rate":${LEARNING_RATE},"learning_decay_rate":0.0001,"mu":${MU},"damping":${DAMPING},"constrain_norm":true,"norm_constraint":${NORM_CONSTRAINT},"checkpoint_every":${CHECKPOINT_EVERY},"ion_pos":"${ION_POS}","ion_charges":"${ION_CHARGES}","nelec":"${NELEC}"}
EOF
echo 'timestamp,index,memory_used_mib' > "${GPUMEM_TMP}"
( while true; do nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits >> "${GPUMEM_TMP}" 2>/dev/null || true; sleep 10; done ) &
GPU_MON_PID=$!; cleanup() { kill "${GPU_MON_PID}" 2>/dev/null || true; wait "${GPU_MON_PID}" 2>/dev/null || true; }; trap cleanup EXIT INT TERM
START_UNIX=$(date +%s); set +e
vmc-molecule \
  --reload.logdir="${SOURCE_DIR}" --reload.checkpoint_relative_file_path="checkpoints/${SOURCE_EPOCH}.npz" \
  --reload.use_config_file=True --reload.use_checkpoint_file=True --reload.new_optimizer_state="${NEW_OPTIMIZER_STATE}" \
  --reload.reburn="${REBURN}" --reload.append="${APPEND}" --reload.same_logdir=False \
  --config.logdir="${RUN_DIR}" --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.problem.ion_pos="${ION_POS}" --config.problem.ion_charges="${ION_CHARGES}" --config.problem.nelec="${NELEC}" \
  --config.eval.nepochs=0 --config.wandb.mode=disabled \
  --config.vmc.nchains=1000 --config.vmc.nburn=5000 --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.clip_threshold=5.0 --config.vmc.clip_center=mean --config.vmc.nepochs=${NEPOCHS} \
  --config.vmc.checkpoint_every=${CHECKPOINT_EVERY} --config.vmc.best_checkpoint_every=${CHECKPOINT_EVERY} \
  --config.vmc.optimizer_type=spring --config.vmc.optimizer.spring.schedule_type=inverse_time \
  --config.vmc.optimizer.spring.learning_rate="${LEARNING_RATE}" --config.vmc.optimizer.spring.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.spring.mu=${MU} --config.vmc.optimizer.spring.damping=${DAMPING} \
  --config.vmc.optimizer.spring.constrain_norm=True --config.vmc.optimizer.spring.norm_constraint=${NORM_CONSTRAINT}
STATUS=$?; set -e; END_UNIX=$(date +%s); cleanup; trap - EXIT INT TERM
mkdir -p "${RUN_DIR}"; mv "${METADATA_TMP}" "${RUN_DIR}/run_metadata.json"; mv "${GPUMEM_TMP}" "${RUN_DIR}/gpumem.csv"
printf 'start_unix,end_unix,elapsed_seconds,target_epoch,status\n%s,%s,%s,%s,%s\n' "${START_UNIX}" "${END_UNIX}" "$((END_UNIX-START_UNIX))" "${NEPOCHS}" "${STATUS}" > "${RUN_DIR}/run_timing.csv"
exit "${STATUS}"
