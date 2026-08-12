#!/bin/bash
#SBATCH --job-name=H2O-MinSR-s1-100k
#SBATCH --account=def-ortner
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407,fc10405,fc10515
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

HERE=/scratch/dexuan1/vmcnet/experiments/H2O_N2matched_KFAC5000_SPRING_MinSR_E100000_20260807
WORKTREE=/scratch/dexuan1/vmcnet_paper_18b9b03
SOURCE=/scratch/dexuan1/runs/H2O_kfac_pre5000_N2matched_20260807
CHECKPOINT=${SOURCE}/checkpoints/5000.npz
ROOT=/scratch/dexuan1/runs/H2O_MinSR_lr002_after_kfac5000_E100000_seed1_20260807
META=${ROOT}_metadata
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311
EXPECTED_COMMIT=18b9b03b68a18c0fef1256be690dd98fa3c114e7

test -f "${CHECKPOINT}"
[[ "$(git -C "${WORKTREE}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
[[ -z "$(git -C "${WORKTREE}" status --porcelain)" ]]
[[ ! -e "${ROOT}" && ! -e "${META}" ]] || {
  echo "refusing to overwrite ${ROOT} or ${META}" >&2
  exit 2
}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"

export PYTHONPATH="/home/dexuan1/paper_vmcnet_compat:${WORKTREE}"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export VMC_RELOAD_SEED=1

mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd "${WORKTREE}"
hostname > "${META}/hostname.txt"
nvidia-smi > "${META}/nvidia_smi_start.txt"
git rev-parse HEAD > "${META}/git_commit.txt"
git status --porcelain > "${META}/git_status.txt"
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"

cat > "${META}/protocol.txt" <<EOF
system=H2O
source_checkpoint=${CHECKPOINT}
source_checkpoint_internal_epoch=4999
independent_reload_seed=1
checkpoint_parameters_and_walkers_preserved=true
checkpoint_prng_key_replaced=true
optimizer=minsr_via_spring_mu0
learning_rate=0.02
mu=0.0
damping=0.001
norm_constraint=0.001
schedule_type=inverse_time
learning_decay_rate=0.0001
training_nchains=1000
training_epochs=100000
nsteps_per_param_update=10
clip_threshold=5
clip_center=mean
reload_new_optimizer_state=true
reload_reburn=false
evaluation_nchains=2000
evaluation_burn_in=10000
evaluation_epochs=20000
paper_worktree_commit=${EXPECTED_COMMIT}
EOF

START=$(date +%s)
python "${HERE}/seeded_reload_launcher.py" \
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
  --config.vmc.optimizer.spring.learning_rate=0.02 \
  --config.vmc.optimizer.spring.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.spring.mu=0.0 \
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

test -f "${ROOT}/checkpoints/100000.npz"
test -f "${ROOT}/eval/statistics.json"
test -f "${ROOT}/eval/local_energies.txt"
