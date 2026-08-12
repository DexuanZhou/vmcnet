#!/bin/bash
#SBATCH --job-name=C-galerkin-eval
#SBATCH --account=def-ortner
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100_3g.40gb:1
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

test -n "${WSSR_EVAL_MANIFEST:-}"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${WSSR_EVAL_MANIFEST}")
IFS=$'\t' read -r ARM SEED CHECKPOINT <<< "${LINE}"
test -n "${ARM}"

ROOT=/scratch/dexuan1/runs/C_galerkin_residual_recurrence_20260803
SOURCE=${ROOT}/train/${ARM}/seed${SEED}
RUN=${ROOT}/frozen/${ARM}/seed${SEED}/epoch${CHECKPOINT}
test -f "${SOURCE}/checkpoints/${CHECKPOINT}.npz"
[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite ${RUN}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false VMCNET_DISABLE_CHECKPOINTS=1
export WSSR_SEED="${SEED}" WSSR_CHECKPOINT="${CHECKPOINT}"

mkdir -p "$(dirname "${RUN}")" /scratch/dexuan1/runs/logs
cd /scratch/dexuan1/vmcnet
python experiments/C_galerkin_residual_recurrence_20260803/eval_seeded_launcher.py \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path="checkpoints/${CHECKPOINT}.npz" \
  --reload.new_optimizer_state=True --reload.reburn=False \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${RUN}" --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nepochs=0 --config.vmc.disable_checkpointing=True \
  --config.eval.nchains=1000 --config.eval.nburn=5000 \
  --config.eval.nepochs=2000 --config.eval.nsteps_per_param_update=10 \
  --config.eval.use_data_from_training=False \
  --config.eval.record_local_energies=True --config.wandb.mode=disabled

test -f "${RUN}/eval/statistics.json"
