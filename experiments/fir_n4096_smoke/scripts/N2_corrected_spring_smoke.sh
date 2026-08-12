#!/bin/bash
#SBATCH --job-name=N2-spring4096-smoke
#SBATCH --account=def-ortner
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
set -euo pipefail
REPO=/scratch/dexuan1/vmcnet
PKG=${REPO}/experiments/fir_n4096_smoke
ROOT=/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2
SOURCE=${ROOT}/kfac_preliminary
RUN_NAME=N2eq_R2068_spring_lr001_n4096_corrected_e50
RUN_DIR=${ROOT}/spring_smoke/${RUN_NAME}
META=${ROOT}/metadata/${RUN_NAME}
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311
[[ -f "${SOURCE}/checkpoints/5000.npz" ]] || { echo 'Missing validated KFAC checkpoint' >&2; exit 2; }
[[ ! -e "${RUN_DIR}" && ! -e "${META}" ]] || { echo 'Refusing overwrite' >&2; exit 2; }
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export VMCNET_DISABLE_CHECKPOINTS=1
export SMOKE_TIMING_DIR="${META}"
mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd "${REPO}"
hostname > "${META}/hostname.txt"
source "${PKG}/scripts/gpu_health_check.sh" "${META}/nvidia_smi_start.txt"
nvidia-smi --query-gpu=name,memory.total,mig.mode.current --format=csv,noheader,nounits > "${META}/gpu_identity.csv"
git rev-parse HEAD > "${META}/git_commit.txt"
printf '%s\n' "${SOURCE}/checkpoints/5000.npz" > "${META}/source_checkpoint.txt"
python - <<'PY' > "${SMOKE_TIMING_DIR}/versions.txt"
import platform, jax, jaxlib
print('Python', platform.python_version())
print('JAX', jax.__version__)
print('jaxlib', jaxlib.__version__)
print('devices', jax.devices())
PY
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"
echo 'timestamp_unix,gpu_name,memory_total_mib,memory_used_mib,utilization_gpu_percent' > "${META}/gpu_memory_poll.csv"
( while true; do TS=$(date +%s.%N); nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -v ts="${TS}" -F', ' '{print ts "," $1 "," $2 "," $3 "," $4}' >> "${META}/gpu_memory_poll.csv" 2>/dev/null || true; sleep 1; done ) &
MON=$!; cleanup(){ kill "${MON}" 2>/dev/null || true; wait "${MON}" 2>/dev/null || true; }; trap cleanup EXIT INT TERM
set +e
python "${PKG}/scripts/timed_reload_launcher.py" \
 --reload.logdir="${SOURCE}" --reload.checkpoint_relative_file_path=checkpoints/5000.npz \
 --reload.use_config_file=True --reload.use_checkpoint_file=True --reload.new_optimizer_state=True \
 --reload.reburn=False --reload.append=False --reload.same_logdir=False \
 --config.logdir="${RUN_DIR}" --config.base_logdir="${RUN_DIR}" --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
 --config.vmc.nchains=4096 --config.vmc.nburn=5000 --config.vmc.nepochs=50 --config.vmc.nsteps_per_param_update=10 \
 --config.vmc.clip_center=mean --config.vmc.clip_threshold=5.0 --config.vmc.check_for_nans=True --config.vmc.disable_checkpointing=True \
 --config.eval.nchains=4096 --config.eval.nepochs=0 --config.eval.nburn=0 --config.wandb.mode=disabled \
 --config.vmc.optimizer_type=spring \
 --config.vmc.optimizer.spring.schedule_type=inverse_time --config.vmc.optimizer.spring.learning_decay_rate=0.0001 \
 --config.vmc.optimizer.spring.learning_rate=0.001 --config.vmc.optimizer.spring.mu=0.99 \
 --config.vmc.optimizer.spring.damping=0.001 --config.vmc.optimizer.spring.constrain_norm=True \
 --config.vmc.optimizer.spring.norm_constraint=0.001
STATUS=$?
set -e
cleanup; trap - EXIT INT TERM
find "${RUN_DIR}" -name '*.npz' -print > "${META}/npz_files.txt" 2>/dev/null || true
exit "${STATUS}"
