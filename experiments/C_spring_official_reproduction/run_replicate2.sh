#!/bin/bash
#SBATCH --job-name=C-spring-official-r2
#SBATCH --account=def-ortner
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

REPO=/scratch/dexuan1/vmcnet
SOURCE=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1
CHECKPOINT=${SOURCE}/checkpoints/1000.npz
ROOT=/scratch/dexuan1/runs/C_spring_official_kfac1000_E100000_replicate2
META=${ROOT}_metadata
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

[[ -f "${CHECKPOINT}" ]] || { echo "missing ${CHECKPOINT}" >&2; exit 2; }
[[ ! -e "${ROOT}" && ! -e "${META}" ]] || {
  echo "refusing to overwrite ${ROOT} or ${META}" >&2
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
protocol=official_C_SPRING_reproduction_replicate2
system=C
electron_configuration=(4,2)
source_checkpoint=${CHECKPOINT}
preliminary_optimizer=kfac
preliminary_epochs=1000
preliminary_nchains=1000
optimizer=spring
learning_rate=0.02
mu=0.99
damping=0.001
norm_constraint=0.001
training_nchains=1000
training_epochs=100000
evaluation_nchains=2000
evaluation_burn=5000
evaluation_epochs=20000
nsteps_per_param_update=10
reload_new_optimizer_state=True
reload_reburn=False
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
  --reload.checkpoint_relative_file_path=checkpoints/1000.npz \
  --reload.new_optimizer_state=True \
  --reload.reburn=False \
  --reload.append=False \
  --config.logdir="${ROOT}" \
  --config.base_logdir="${ROOT}" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name=NONE \
  --config.vmc.nchains=1000 \
  --config.vmc.nepochs=100000 \
  --config.eval.nchains=2000 \
  --config.eval.nburn=5000 \
  --config.eval.nepochs=20000 \
  --config.vmc.optimizer_type=spring \
  --config.vmc.optimizer.spring.learning_rate=0.02 \
  --config.wandb.mode=disabled

cleanup
trap - EXIT INT TERM

[[ -f "${ROOT}/checkpoints/100000.npz" ]] || {
  echo "missing final training checkpoint" >&2
  exit 4
}
[[ -f "${ROOT}/eval/statistics.json" ]] || {
  echo "missing evaluation statistics" >&2
  exit 5
}
[[ -f "${ROOT}/eval/local_energies.txt" ]] || {
  echo "missing evaluation local energies" >&2
  exit 5
}
