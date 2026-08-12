#!/bin/bash
#SBATCH --job-name=C-freshSPRING50k
#SBATCH --account=def-ortner
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_C_stagewise_switch_20260807
REPO=/scratch/dexuan1/vmcnet
SOURCE=/scratch/dexuan1/runs/C_spring_official_kfac1000_E100000_replicate2
ROOT=/scratch/dexuan1/runs/C_fresh_spring50k_control_20260807
RUN=${ROOT}/train
META=${ROOT}/metadata
SOURCE_EPOCH=50000
UPDATES=5000

test -f "${SOURCE}/checkpoints/${SOURCE_EPOCH}.npz"
test -f "${SNAPSHOT}/SOURCE_SNAPSHOT.txt"
[[ ! -e "${RUN}" && ! -e "${META}" ]] || {
  echo "refusing to overwrite ${RUN} or ${META}" >&2
  exit 2
}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export PYTHONPATH="${SNAPSHOT}"
export GIT_DIR="${REPO}/.git"
export GIT_WORK_TREE="${SNAPSHOT}"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd "${SNAPSHOT}"
cp SOURCE_SNAPSHOT.txt "${META}/"
hostname > "${META}/hostname.txt"
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"
cat > "${META}/protocol.txt" <<EOF
question=does_fresh_SPRING_state_explain_the_epoch50000_WSSR_advantage
source=${SOURCE}/checkpoints/${SOURCE_EPOCH}.npz
source_optimizer=spring
source_epoch=${SOURCE_EPOCH}
optimizer=spring
mu=0.99
updates=${UPDATES}
new_optimizer_state=true
reburn=false
nchains=1000
learning_rate_local=0.0033333333333333335
learning_decay_rate_local=0.0000166666666666667
equivalent_global_schedule=0.02/(1+1e-4*(50000+s))
damping=0.001
norm_constraint=0.001
EOF

vmc-molecule \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path="checkpoints/${SOURCE_EPOCH}.npz" \
  --reload.new_optimizer_state=True --reload.reburn=False \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${RUN}" --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nchains=1000 --config.vmc.nburn=0 \
  --config.vmc.nepochs="${UPDATES}" --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.check_for_nans=True \
  --config.vmc.checkpoint_every="${UPDATES}" \
  --config.vmc.best_checkpoint_every="${UPDATES}" \
  --config.eval.nepochs=0 \
  --config.vmc.optimizer_type=spring \
  --config.vmc.optimizer.spring.schedule_type=inverse_time \
  --config.vmc.optimizer.spring.learning_rate=0.0033333333333333335 \
  --config.vmc.optimizer.spring.learning_decay_rate=0.0000166666666666667 \
  --config.vmc.optimizer.spring.mu=0.99 \
  --config.vmc.optimizer.spring.damping=0.001 \
  --config.vmc.optimizer.spring.constrain_norm=True \
  --config.vmc.optimizer.spring.norm_constraint=0.001 \
  --config.wandb.mode=disabled

test -f "${RUN}/checkpoints/${UPDATES}.npz"
