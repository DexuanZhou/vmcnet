#!/bin/bash
#SBATCH --job-name=H2O-KFAC-100k
#SBATCH --account=def-ortner
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407,fc10405,fc10515
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

WORKTREE=/scratch/dexuan1/vmcnet_paper_18b9b03_kfaccompat
SOURCE=/scratch/dexuan1/runs/H2O_kfac_pre5000_N2matched_20260807
CHECKPOINT=${SOURCE}/checkpoints/5000.npz
ROOT=/scratch/dexuan1/runs/H2O_KFAC_lr005_after_kfac5000_E100000_20260807
META=${ROOT}_metadata
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311
EXPECTED_COMMIT=18b9b03b68a18c0fef1256be690dd98fa3c114e7

test -f "${CHECKPOINT}"
[[ "$(git -C "${WORKTREE}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
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

mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd "${WORKTREE}"
hostname > "${META}/hostname.txt"
nvidia-smi > "${META}/nvidia_smi_start.txt"
git rev-parse HEAD > "${META}/git_commit.txt"
git status --porcelain > "${META}/git_status.txt"
git diff -- vmcnet/utils/curvature_tags_and_blocks.py \
  > "${META}/kfac_compat_patch.diff"
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"

cat > "${META}/protocol.txt" <<EOF
system=H2O
source_checkpoint=${CHECKPOINT}
source_checkpoint_internal_epoch=4999
preliminary_optimizer=kfac
preliminary_epochs=5000
optimizer=kfac
learning_rate=0.05
damping=0.001
norm_constraint=0.001
curvature_ema=0.95
inverse_update_period=1
estimation_mode=fisher_exact
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
  --config.vmc.optimizer_type=kfac \
  --config.vmc.optimizer.kfac.schedule_type=inverse_time \
  --config.vmc.optimizer.kfac.learning_rate=0.05 \
  --config.vmc.optimizer.kfac.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.kfac.damping=0.001 \
  --config.vmc.optimizer.kfac.norm_constraint=0.001 \
  --config.vmc.optimizer.kfac.curvature_ema=0.95 \
  --config.vmc.optimizer.kfac.inverse_update_period=1 \
  --config.vmc.optimizer.kfac.estimation_mode=fisher_exact \
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
