#!/bin/bash
#SBATCH --job-name=C-MinSR-paper-100k
#SBATCH --account=def-ortner
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

SOURCE=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1
RUN=/scratch/dexuan1/runs/C_minsr_paper_kfac1000_E100000
META=${RUN}_metadata
CHECKPOINT=${SOURCE}/checkpoints/1000.npz

[[ -f "${CHECKPOINT}" ]] || { echo "missing ${CHECKPOINT}" >&2; exit 2; }
[[ ! -e "${RUN}" && ! -e "${META}" ]] || {
  echo "refusing to overwrite ${RUN} or ${META}" >&2
  exit 2
}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd /scratch/dexuan1/vmcnet

hostname > "${META}/hostname.txt"
git rev-parse HEAD > "${META}/git_commit.txt"
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"

cat > "${META}/protocol.txt" <<EOF
system=C
electron_configuration=(4,2)
source_checkpoint=${CHECKPOINT}
preliminary_optimizer=kfac
preliminary_epochs=1000
training_walkers=1000
training_epochs=100000
optimizer=MinSR_via_SPRING_mu0
mu=0.0
learning_rate=0.1
learning_decay_rate=0.0001
damping=0.001
norm_constraint=0.001
reload_new_optimizer_state=True
reload_reburn=False
evaluation_walkers=2000
evaluation_burn=5000
evaluation_epochs=20000
mcmc_steps=10
EOF

vmc-molecule \
  --reload.logdir="${SOURCE}" \
  --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/1000.npz \
  --reload.new_optimizer_state=True \
  --reload.reburn=False \
  --reload.append=False \
  --reload.same_logdir=False \
  --config.logdir="${RUN}" \
  --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name=NONE \
  --config.problem.nelec='(4,2)' \
  --config.vmc.nchains=1000 \
  --config.vmc.nburn=5000 \
  --config.vmc.nepochs=100000 \
  --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.clip_center=mean \
  --config.vmc.clip_threshold=5.0 \
  --config.vmc.check_for_nans=True \
  --config.vmc.checkpoint_every=5000 \
  --config.vmc.best_checkpoint_every=5000 \
  --config.eval.use_data_from_training=False \
  --config.eval.nchains=2000 \
  --config.eval.nburn=5000 \
  --config.eval.nepochs=20000 \
  --config.eval.nsteps_per_param_update=10 \
  --config.eval.record_local_energies=True \
  --config.vmc.optimizer_type=spring \
  --config.vmc.optimizer.spring.schedule_type=inverse_time \
  --config.vmc.optimizer.spring.learning_rate=0.1 \
  --config.vmc.optimizer.spring.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.spring.mu=0.0 \
  --config.vmc.optimizer.spring.damping=0.001 \
  --config.vmc.optimizer.spring.constrain_norm=True \
  --config.vmc.optimizer.spring.norm_constraint=0.001 \
  --config.wandb.mode=disabled

[[ -f "${RUN}/checkpoints/100000.npz" ]] || {
  echo "missing final training checkpoint" >&2
  exit 4
}
[[ -f "${RUN}/eval/statistics.json" ]] || {
  echo "missing frozen evaluation statistics" >&2
  exit 5
}
