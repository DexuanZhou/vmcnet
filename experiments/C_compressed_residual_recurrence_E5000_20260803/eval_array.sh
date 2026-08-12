#!/bin/bash
#SBATCH --job-name=C-compressed-solrec-eval
#SBATCH --account=def-ortner
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100_3g.40gb:1
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/scratch/dexuan1/runs/C_compressed_residual_recurrence_E5000_20260803
case "${SLURM_ARRAY_TASK_ID}" in
  0) RANK=400; SEED=0 ;;
  1) RANK=400; SEED=1 ;;
  2) RANK=800; SEED=0 ;;
  3) RANK=800; SEED=1 ;;
  *) echo "unexpected task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

ARM=residual_rank${RANK}
SOURCE=${ROOT}/train/${ARM}/seed${SEED}
RUN=${ROOT}/frozen/${ARM}/seed${SEED}
test -f "${SOURCE}/checkpoints/5000.npz"
[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite ${RUN}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false VMCNET_DISABLE_CHECKPOINTS=1
export WSSR_PAIRED_SEED="${SEED}"

mkdir -p "$(dirname "${RUN}")" /scratch/dexuan1/runs/logs
cd /scratch/dexuan1/vmcnet
python experiments/C_compressed_residual_recurrence_E5000_20260803/eval_seeded_launcher.py \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/5000.npz \
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
