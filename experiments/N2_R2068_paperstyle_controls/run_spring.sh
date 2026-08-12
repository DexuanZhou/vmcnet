#!/bin/bash
#SBATCH --job-name=N2-R2068-SPRING-100k
#SBATCH --account=def-ortner
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

REPO=/scratch/dexuan1/vmcnet
SOURCE=/scratch/dexuan1/runs/kfac_pre/N2eq_R2068_kfac_pre5000
CHECKPOINT=${SOURCE}/checkpoints/5000.npz
ROOT=/scratch/dexuan1/runs/N2_R2068_spring_paperstyle_after_kfac5000_E100000
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
system=N2
bond_length_bohr=2.068
ion_positions_z_bohr=(-1.034,+1.034)
electron_configuration=(7,7)
source_checkpoint=${CHECKPOINT}
preliminary_optimizer=kfac
preliminary_epochs=5000
preliminary_nchains=1000
optimizer=spring
learning_rate=0.002
mu=0.99
damping=0.001
norm_constraint=0.001
schedule_type=inverse_time
learning_decay_rate=0.0001
training_nchains=1000
training_epochs=100000
training_burn_repeated=False
evaluation_nchains=2000
evaluation_burn=10000
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
  --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/5000.npz \
  --reload.new_optimizer_state=True \
  --reload.reburn=False \
  --reload.append=False \
  --reload.same_logdir=False \
  --config.logdir="${ROOT}" \
  --config.base_logdir="${ROOT}" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name=NONE \
  --config.problem.ion_pos='((0.,0.,-1.034),(0.,0.,1.034))' \
  --config.problem.ion_charges='(7.,7.)' \
  --config.problem.nelec='(7,7)' \
  --config.vmc.nchains=1000 \
  --config.vmc.nepochs=100000 \
  --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.clip_threshold=5.0 \
  --config.vmc.clip_center=mean \
  --config.vmc.check_for_nans=True \
  --config.vmc.optimizer_type=spring \
  --config.vmc.optimizer.spring.schedule_type=inverse_time \
  --config.vmc.optimizer.spring.learning_rate=0.002 \
  --config.vmc.optimizer.spring.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.spring.mu=0.99 \
  --config.vmc.optimizer.spring.damping=0.001 \
  --config.vmc.optimizer.spring.constrain_norm=True \
  --config.vmc.optimizer.spring.norm_constraint=0.001 \
  --config.eval.nchains=2000 \
  --config.eval.nburn=10000 \
  --config.eval.nepochs=20000 \
  --config.eval.nsteps_per_param_update=10 \
  --config.eval.record_local_energies=True \
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
