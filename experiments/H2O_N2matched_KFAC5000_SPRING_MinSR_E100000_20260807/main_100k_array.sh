#!/bin/bash
#SBATCH --job-name=H2O-SPRING-MinSR-100k
#SBATCH --account=def-ortner
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407,fc10405,fc10515
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

WORKTREE=/scratch/dexuan1/vmcnet_paper_18b9b03
SOURCE=/scratch/dexuan1/runs/H2O_kfac_pre5000_N2matched_20260807
CHECKPOINT=${SOURCE}/checkpoints/5000.npz
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311
EXPECTED_COMMIT=18b9b03b68a18c0fef1256be690dd98fa3c114e7

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    METHOD=spring
    LEARNING_RATE=0.002
    MU=0.95
    ROOT=/scratch/dexuan1/runs/H2O_SPRING_mu095_lr002_after_kfac5000_E100000_20260807
    ;;
  1)
    METHOD=minsr
    LEARNING_RATE=0.02
    MU=0.0
    ROOT=/scratch/dexuan1/runs/H2O_MinSR_lr002_after_kfac5000_E100000_20260807
    ;;
  *)
    echo "unexpected array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac
META=${ROOT}_metadata

[[ -f "${CHECKPOINT}" ]] || { echo "missing checkpoint ${CHECKPOINT}" >&2; exit 2; }
[[ "$(git -C "${WORKTREE}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
  echo "unexpected paper worktree commit" >&2
  exit 2
}
[[ -z "$(git -C "${WORKTREE}" status --porcelain)" ]] || {
  echo "paper worktree is not clean" >&2
  exit 2
}
[[ ! -e "${ROOT}" && ! -e "${META}" ]] || {
  echo "refusing to overwrite ${ROOT} or ${META}" >&2
  exit 2
}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"

export PYTHONPATH="/home/dexuan1/paper_vmcnet_compat:${WORKTREE}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd "${WORKTREE}"

hostname > "${META}/hostname.txt"
nvidia-smi > "${META}/nvidia_smi_start.txt"
git rev-parse HEAD > "${META}/git_commit.txt"
git status --porcelain > "${META}/git_status.txt"
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"

SOURCE="${SOURCE}" CHECKPOINT="${CHECKPOINT}" WORKTREE="${WORKTREE}" python - <<'PY' \
  > "${META}/environment_and_checkpoint.txt"
import json
import os
import numpy as np
import jax
import vmcnet

source = os.environ["SOURCE"]
checkpoint = os.environ["CHECKPOINT"]
worktree = os.environ["WORKTREE"]
print("vmcnet_file", vmcnet.__file__)
print("jax_version", jax.__version__)
print("jax_backend", jax.default_backend())
print("jax_devices", jax.devices())
assert os.path.realpath(vmcnet.__file__).startswith(worktree + os.sep)
assert jax.default_backend() == "gpu"
with open(source + "/config.json") as f:
    config = json.load(f)
expected_pos = [[0.0, 0.0, 0.0], [1.43233673, 0.0, 1.10715266], [-1.43233673, 0.0, 1.10715266]]
assert config["problem"]["ion_pos"] == expected_pos
assert config["problem"]["ion_charges"] == [8.0, 1.0, 1.0]
assert config["problem"]["nelec"] == [5, 5]
assert config["vmc"]["nchains"] == 1000
state = np.load(checkpoint, allow_pickle=True)
assert int(state["e"]) == 4999
for key in ("d", "p", "o", "k"):
    assert key in state.files
print("checkpoint_internal_epoch", int(state["e"]))
print("checkpoint_validation", "PASS")
PY

cat > "${META}/protocol.txt" <<EOF
system=H2O
geometry_bohr=O(0,0,0);H(+1.43233673,0,+1.10715266);H(-1.43233673,0,+1.10715266)
electron_configuration=(5,5)
source_checkpoint=${CHECKPOINT}
preliminary_optimizer=kfac
preliminary_epochs=5000
optimizer=${METHOD}
learning_rate=${LEARNING_RATE}
mu=${MU}
damping=0.001
norm_constraint=0.001
schedule_type=inverse_time
learning_decay_rate=0.0001
training_nchains=1000
training_epochs=100000
nsteps_per_param_update=10
clip_threshold=5
clip_center=mean
reload_new_optimizer_state=True
reload_reburn=False
evaluation_nchains=2000
evaluation_burn_in=10000
evaluation_epochs=20000
evaluation_mcmc_steps_between_measurements=10
paper_worktree_commit=${EXPECTED_COMMIT}
EOF

START=$(date +%s)
vmc-molecule \
  --reload.logdir="${SOURCE}" \
  --reload.use_config_file=False \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/5000.npz \
  --reload.new_optimizer_state=True \
  --reload.reburn=False \
  --reload.append=False \
  --reload.same_logdir=False \
  --config.logdir="${ROOT}" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name=NONE \
  --config.problem.ion_pos='((0.,0.,0.),(1.43233673,0.,1.10715266),(-1.43233673,0.,1.10715266))' \
  --config.problem.ion_charges='(8.,1.,1.)' \
  --config.problem.nelec='(5,5)' \
  --config.model.ferminet.ndeterminants=16 \
  --config.model.ferminet.full_det=True \
  --config.vmc.nchains=1000 \
  --config.vmc.nburn=5000 \
  --config.vmc.nepochs=100000 \
  --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.clip_threshold=5.0 \
  --config.vmc.clip_center=mean \
  --config.vmc.check_for_nans=True \
  --config.vmc.checkpoint_every=5000 \
  --config.vmc.best_checkpoint_every=5000 \
  --config.vmc.optimizer_type=spring \
  --config.vmc.optimizer.spring.schedule_type=inverse_time \
  --config.vmc.optimizer.spring.learning_rate="${LEARNING_RATE}" \
  --config.vmc.optimizer.spring.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.spring.mu="${MU}" \
  --config.vmc.optimizer.spring.momentum=0.0 \
  --config.vmc.optimizer.spring.damping=0.001 \
  --config.vmc.optimizer.spring.constrain_norm=True \
  --config.vmc.optimizer.spring.norm_constraint=0.001 \
  --config.eval.use_data_from_training=False \
  --config.eval.nchains=2000 \
  --config.eval.nburn=10000 \
  --config.eval.nepochs=20000 \
  --config.eval.nsteps_per_param_update=10 \
  --config.eval.record_local_energies=True
END=$(date +%s)
printf 'start_unix,end_unix,elapsed_seconds\n%s,%s,%s\n' \
  "${START}" "${END}" "$((END - START))" > "${META}/run_timing.csv"

[[ -f "${ROOT}/checkpoints/100000.npz" ]] || {
  echo "missing epoch-100000 checkpoint" >&2
  exit 4
}
[[ -f "${ROOT}/eval/statistics.json" ]] || {
  echo "missing frozen statistics" >&2
  exit 5
}
[[ -f "${ROOT}/eval/local_energies.txt" ]] || {
  echo "missing frozen local energies" >&2
  exit 5
}
