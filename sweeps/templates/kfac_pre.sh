#!/bin/bash
# Execution template sourced by an instantiated pilot Slurm script.
set -euo pipefail

: "${SYSTEM:?SYSTEM is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${NEPOCHS:?NEPOCHS is required}"
: "${ION_POS:?ION_POS is required}"
: "${ION_CHARGES:?ION_CHARGES is required}"
: "${NELEC:?NELEC is required}"

REPO=/scratch/dexuan1/vmcnet
RUN_DIR="/scratch/dexuan1/runs/pilot/${SYSTEM}/${RUN_NAME}"
METADATA_TMP="${RUN_DIR}.run_metadata.json.tmp"
GPUMEM_TMP="${RUN_DIR}.gpumem.csv.tmp"
VMCNET_VENV="${VMCNET_VENV:-/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311}"

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
{"system":"${SYSTEM}","run_name":"${RUN_NAME}","family":"kfac_pre","nepochs":${NEPOCHS},"nchains":1000,"nburn":5000,"nsteps_per_param_update":10,"clip_threshold":5.0,"schedule_type":"inverse_time","learning_rate":0.05,"learning_decay_rate":0.0001}
EOF

echo 'timestamp,index,memory_used_mib' > "${GPUMEM_TMP}"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits \
      >> "${GPUMEM_TMP}" 2>/dev/null || true
    sleep 10
  done
) &
GPU_MON_PID=$!
cleanup_monitor() { kill "${GPU_MON_PID}" 2>/dev/null || true; wait "${GPU_MON_PID}" 2>/dev/null || true; }
trap cleanup_monitor EXIT INT TERM

START_UNIX=$(date +%s)
set +e
vmc-molecule \
  --config.logdir="${RUN_DIR}" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name=NONE \
  --config.problem.ion_pos="${ION_POS}" \
  --config.problem.ion_charges="${ION_CHARGES}" \
  --config.problem.nelec="${NELEC}" \
  --config.distribute=False \
  --config.eval.nepochs=0 \
  --config.wandb.mode=disabled \
  --config.vmc.nchains=1000 \
  --config.vmc.nburn=5000 \
  --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.clip_threshold=5.0 \
  --config.vmc.clip_center=mean \
  --config.vmc.nepochs="${NEPOCHS}" \
  --config.vmc.checkpoint_every="${NEPOCHS}" \
  --config.vmc.best_checkpoint_every="${NEPOCHS}" \
  --config.vmc.optimizer_type=kfac \
  --config.vmc.optimizer.kfac.schedule_type=inverse_time \
  --config.vmc.optimizer.kfac.learning_rate=0.05 \
  --config.vmc.optimizer.kfac.learning_decay_rate=0.0001
STATUS=$?
set -e
END_UNIX=$(date +%s)
cleanup_monitor
trap - EXIT INT TERM
mv "${METADATA_TMP}" "${RUN_DIR}/run_metadata.json"
mv "${GPUMEM_TMP}" "${RUN_DIR}/gpumem.csv"
printf 'start_unix,end_unix,elapsed_seconds,nepochs,status\n%s,%s,%s,%s,%s\n' \
  "${START_UNIX}" "${END_UNIX}" "$((END_UNIX - START_UNIX))" "${NEPOCHS}" "${STATUS}" \
  > "${RUN_DIR}/run_timing.csv"

if [[ ${STATUS} -eq 0 ]]; then
  rm -f "${RUN_DIR}/best_checkpoint.npz"
  find "${RUN_DIR}/checkpoints" -maxdepth 1 -type f -name '*.npz' \
    ! -name "${NEPOCHS}.npz" -delete
fi
exit "${STATUS}"
