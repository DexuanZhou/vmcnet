#!/bin/bash
#SBATCH --job-name=N2eq-kfac-pre5000-r2068
#SBATCH --account=def-ortner
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
set -euo pipefail

REPO=/scratch/dexuan1/vmcnet
RUN_DIR=/scratch/dexuan1/runs/kfac_pre/N2eq_R2068_kfac_pre5000
VMCNET_VENV="${VMCNET_VENV:-/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311}"
[[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to reuse ${RUN_DIR}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VMCNET_VENV}/bin/activate"
export TMPDIR="${SLURM_TMPDIR:-/scratch/dexuan1/tmp}"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${TMPDIR}" "$(dirname "${RUN_DIR}")" /scratch/dexuan1/runs/logs
cd "${REPO}"

METADATA_TMP="${RUN_DIR}.run_metadata.json.tmp"
GPUMEM_TMP="${RUN_DIR}.gpumem.csv.tmp"
cat > "${METADATA_TMP}" <<EOF
{"system":"N2eq","bond_length_bohr":2.068,"run_name":"N2eq_R2068_kfac_pre5000","family":"kfac_pre","nepochs":5000,"nchains":1000,"nburn":5000,"nsteps_per_param_update":10,"clip_center":"mean","clip_threshold":5.0,"eval_nepochs":0,"optimizer_type":"kfac","learning_rate":0.05,"schedule_type":"inverse_time","learning_decay_rate":0.0001,"checkpoint_every":5000,"ion_pos":"((0.0,0.0,-1.034),(0.0,0.0,1.034))","ion_charges":"(7.0,7.0)","nelec":"(7,7)"}
EOF
echo 'timestamp,index,memory_used_mib' > "${GPUMEM_TMP}"
( while true; do nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits >> "${GPUMEM_TMP}" 2>/dev/null || true; sleep 10; done ) &
GPU_MON_PID=$!
cleanup() { kill "${GPU_MON_PID}" 2>/dev/null || true; wait "${GPU_MON_PID}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

START_UNIX=$(date +%s)
set +e
vmc-molecule --config.logdir="${RUN_DIR}" --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
 --config.problem.ion_pos='((0.0,0.0,-1.034),(0.0,0.0,1.034))' --config.problem.ion_charges='(7.0,7.0)' --config.problem.nelec='(7,7)' \
 --config.distribute=False --config.eval.nepochs=0 --config.wandb.mode=disabled \
 --config.vmc.nchains=1000 --config.vmc.nburn=5000 --config.vmc.nsteps_per_param_update=10 \
 --config.vmc.clip_threshold=5.0 --config.vmc.clip_center=mean --config.vmc.nepochs=5000 \
 --config.vmc.checkpoint_every=5000 --config.vmc.best_checkpoint_every=5000 \
 --config.vmc.optimizer_type=kfac --config.vmc.optimizer.kfac.schedule_type=inverse_time \
 --config.vmc.optimizer.kfac.learning_rate=0.05 --config.vmc.optimizer.kfac.learning_decay_rate=0.0001
STATUS=$?
set -e
END_UNIX=$(date +%s)
cleanup
trap - EXIT INT TERM
mkdir -p "${RUN_DIR}"
mv "${METADATA_TMP}" "${RUN_DIR}/run_metadata.json"
mv "${GPUMEM_TMP}" "${RUN_DIR}/gpumem.csv"
printf 'start_unix,end_unix,elapsed_seconds,nepochs,status\n%s,%s,%s,%s,%s\n' "${START_UNIX}" "${END_UNIX}" "$((END_UNIX-START_UNIX))" 5000 "${STATUS}" > "${RUN_DIR}/run_timing.csv"
exit "${STATUS}"
