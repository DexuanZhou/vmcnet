#!/bin/bash
#SBATCH --job-name=C-kfac-lr02-E50k
#SBATCH --account=def-ortner
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

REPO=/scratch/dexuan1/vmcnet
SOURCE=/scratch/dexuan1/runs/wssr_C_4096_protocol/C_KFAC4096_pre5000
CHECKPOINT=${SOURCE}/checkpoints/5000.npz
ROOT=/scratch/dexuan1/runs/C_kfac_lr02_after_kfac5000_E50000
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
system=C
electron_configuration=(4,2)
pretraining_epochs=5000
source_checkpoint=${CHECKPOINT}
optimizer=kfac
new_optimizer_state=True
learning_rate=0.02
schedule_type=inverse_time
learning_decay_rate=0.0001
damping=0.001
norm_constraint=0.001
nchains=4096
nburn=5000
reburn=False
nepochs=50000
nsteps_per_param_update=10
checkpoint_every=10000
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

python experiments/fir_n4096_smoke/scripts/timed_reload_launcher.py \
  --reload.logdir="${SOURCE}" \
  --reload.checkpoint_relative_file_path=checkpoints/5000.npz \
  --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.new_optimizer_state=True \
  --reload.reburn=False \
  --reload.append=False \
  --reload.same_logdir=False \
  --config.logdir="${ROOT}" \
  --config.base_logdir="${ROOT}" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name=NONE \
  --config.vmc.optimizer_type=kfac \
  --config.vmc.optimizer.kfac.learning_rate=0.02 \
  --config.vmc.optimizer.kfac.schedule_type=inverse_time \
  --config.vmc.optimizer.kfac.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.kfac.damping=0.001 \
  --config.vmc.optimizer.kfac.norm_constraint=0.001 \
  --config.vmc.nchains=4096 \
  --config.eval.nchains=4096 \
  --config.vmc.nburn=5000 \
  --config.vmc.nepochs=50000 \
  --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.check_for_nans=True \
  --config.vmc.disable_checkpointing=False \
  --config.vmc.checkpoint_every=10000 \
  --config.vmc.best_checkpoint_every=10000 \
  --config.eval.nepochs=0 \
  --config.wandb.mode=disabled

cleanup
trap - EXIT INT TERM

for epoch in 10000 20000 30000 40000 50000; do
  [[ -f "${ROOT}/checkpoints/${epoch}.npz" ]] || {
    echo "missing checkpoint ${epoch}.npz" >&2
    exit 4
  }
done
