#!/bin/bash
#SBATCH --job-name=N2-E5k-ref-eval
#SBATCH --account=def-ortner
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-1%2
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    METHOD=spring
    SOURCE=/scratch/dexuan1/runs/N2_R2016_paperSPRING_mu095_lr002_E100000
    ;;
  1)
    METHOD=minsr
    SOURCE=/scratch/dexuan1/runs/N2_R2016_paper_MinSR_lr002_after_kfac5000_E100000
    ;;
  *) exit 2 ;;
esac

REPO=/scratch/dexuan1/vmcnet
RUN=/scratch/dexuan1/runs/N2_R2016_${METHOD}_E5000_light_frozen_20260807
CHECKPOINT=${SOURCE}/checkpoints/5000.npz
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

[[ -f "${CHECKPOINT}" ]] || exit 2
[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite ${RUN}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export VMCNET_DISABLE_CHECKPOINTS=1

mkdir -p "${RUN}/metadata" /scratch/dexuan1/runs/logs
cd "${REPO}"
cat > "${RUN}/metadata/protocol.txt" <<EOF
purpose=N2 E5000 matched light frozen reference
method=${METHOD}
source_checkpoint=${CHECKPOINT}
walkers=2000
burn_in=10000
measurements=2000
mcmc_steps=10
training=False
EOF

vmc-molecule \
  --reload.logdir="${SOURCE}" \
  --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/5000.npz \
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
  --config.vmc.optimizer_type=adam \
  --config.eval.use_data_from_training=False \
  --config.eval.nchains=2000 \
  --config.eval.nburn=10000 \
  --config.eval.nepochs=2000 \
  --config.eval.nsteps_per_param_update=10 \
  --config.eval.record_local_energies=True

[[ -f "${RUN}/eval/statistics.json" ]] || exit 5
