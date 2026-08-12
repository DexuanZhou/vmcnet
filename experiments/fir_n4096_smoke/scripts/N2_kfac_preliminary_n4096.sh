#!/bin/bash
#SBATCH --job-name=N2-kfac-n4096
#SBATCH --account=def-ortner
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
set -euo pipefail

REPO=/scratch/dexuan1/vmcnet
SOURCE=/scratch/dexuan1/runs/kfac_pre/N2eq_R2068_kfac_pre5000
ROOT=${CORRECTED_ROOT:-/scratch/dexuan1/runs/fir_n4096_corrected/N2}
RUN_DIR=${ROOT}/kfac_preliminary
META_DIR=${ROOT}/metadata/kfac_preliminary
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311
[[ -f "${SOURCE}/config.json" ]] || { echo 'Missing formal config' >&2; exit 2; }
[[ ! -e "${RUN_DIR}" && ! -e "${META_DIR}" ]] || { echo 'Refusing to overwrite corrected KFAC output' >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="${SLURM_TMPDIR:-/scratch/dexuan1/tmp}"
mkdir -p "${META_DIR}" "${TMPDIR}" /scratch/dexuan1/runs/logs
cd "${REPO}"
hostname > "${META_DIR}/hostname.txt"
source "${REPO}/experiments/fir_n4096_smoke/scripts/gpu_health_check.sh" "${META_DIR}/nvidia_smi_start.txt"
nvidia-smi --query-gpu=name,memory.total,mig.mode.current --format=csv,noheader,nounits > "${META_DIR}/gpu_identity.csv"
git rev-parse HEAD > "${META_DIR}/git_commit.txt"
scontrol show job "${SLURM_JOB_ID}" > "${META_DIR}/slurm_job.txt"
printf '%s\n' "${SOURCE}/config.json" > "${META_DIR}/source_config.txt"
echo 'timestamp_unix,gpu_name,memory_total_mib,memory_used_mib,utilization_gpu_percent' > "${META_DIR}/gpu_memory_poll.csv"
( while true; do TS=$(date +%s.%N); nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -v ts="${TS}" -F', ' '{print ts "," $1 "," $2 "," $3 "," $4}' >> "${META_DIR}/gpu_memory_poll.csv" 2>/dev/null || true; sleep 1; done ) &
MON=$!; cleanup(){ kill "${MON}" 2>/dev/null || true; wait "${MON}" 2>/dev/null || true; }; trap cleanup EXIT INT TERM
START=$(date +%s)
set +e
vmc-molecule \
 --reload.logdir="${SOURCE}" --reload.use_config_file=True --reload.use_checkpoint_file=False \
 --reload.append=False --reload.same_logdir=False \
 --config.logdir="${RUN_DIR}" --config.base_logdir="${RUN_DIR}" \
 --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
 --config.vmc.nchains=4096 --config.eval.nchains=4096 \
 --config.wandb.mode=disabled
STATUS=$?
set -e
END=$(date +%s); cleanup; trap - EXIT INT TERM
printf 'start_unix,end_unix,elapsed_seconds,exit_status\n%s,%s,%s,%s\n' "${START}" "${END}" "$((END-START))" "${STATUS}" > "${META_DIR}/run_timing.csv"
[[ ${STATUS} -eq 0 ]] || exit "${STATUS}"
[[ -f "${RUN_DIR}/checkpoints/5000.npz" ]] || { echo 'Missing final checkpoint 5000.npz' >&2; exit 4; }
