#!/bin/bash
#SBATCH --job-name=C-b001-frozen-audit
#SBATCH --account=def-ortner
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

case "${SLURM_ARRAY_TASK_ID}" in
  0) EPOCH=102000 ;;
  1) EPOCH=103000 ;;
  2) EPOCH=104000 ;;
  3) EPOCH=105000 ;;
  *) exit 2 ;;
esac

SOURCE=/scratch/dexuan1/runs/C_wssr_rank1600_beta0001_replay_checkpoints_E5000
RUN=/scratch/dexuan1/runs/C_wssr_rank1600_beta0001_frozen_audit/epoch${EPOCH}
META=${RUN}_metadata
CHECKPOINT=${SOURCE}/checkpoints/${EPOCH}.npz

[[ -f "${CHECKPOINT}" ]] || { echo "missing ${CHECKPOINT}" >&2; exit 2; }
[[ ! -e "${RUN}" && ! -e "${META}" ]] || {
  echo "refusing to overwrite ${RUN} or ${META}" >&2
  exit 2
}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd /scratch/dexuan1/vmcnet
hostname > "${META}/hostname.txt"
git rev-parse HEAD > "${META}/git_commit.txt"
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"

cat > "${META}/protocol.txt" <<EOF
mode=independent_frozen_evaluation
source_checkpoint=${CHECKPOINT}
checkpoint_epoch=${EPOCH}
evaluation_walkers=2000
evaluation_burn=5000
evaluation_epochs=20000
mcmc_steps_between_measurements=10
EOF

vmc-molecule \
  --reload.logdir="${SOURCE}" \
  --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path="checkpoints/${EPOCH}.npz" \
  --reload.new_optimizer_state=True \
  --reload.reburn=False \
  --reload.append=False \
  --reload.same_logdir=False \
  --config.logdir="${RUN}" \
  --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name=NONE \
  --config.vmc.nepochs=0 \
  --config.vmc.disable_checkpointing=True \
  --config.eval.use_data_from_training=False \
  --config.eval.nchains=2000 \
  --config.eval.nburn=5000 \
  --config.eval.nepochs=20000 \
  --config.eval.nsteps_per_param_update=10 \
  --config.eval.record_local_energies=True \
  --config.wandb.mode=disabled

[[ -f "${RUN}/eval/statistics.json" ]] || {
  echo "missing frozen statistics" >&2
  exit 5
}
