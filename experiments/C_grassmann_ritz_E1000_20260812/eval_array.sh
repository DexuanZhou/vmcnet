#!/bin/bash
#SBATCH --job-name=C-grass-eval
#SBATCH --time=00:35:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-3%2

set -euo pipefail
REPOSITORY="${VMCNET_REPOSITORY:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ROOT="${C_GRASSMANN_ROOT:-${SCRATCH:-/tmp/${USER}}/runs/C_grassmann_ritz_E1000_20260812}"
case "${SLURM_ARRAY_TASK_ID}" in
  0) ARM=alpha100 ;;
  1) ARM=alpha075 ;;
  2) ARM=alpha050 ;;
  3) ARM=alpha025 ;;
  *) exit 2 ;;
esac
SOURCE=${ROOT}/train/${ARM}
RUN=${ROOT}/frozen/${ARM}
test -f "${SOURCE}/checkpoints/1000.npz"
[[ ! -e "${RUN}" ]] || exit 2

if [[ -n "${VMCNET_MODULE_SETUP:-}" ]]; then
  eval "${VMCNET_MODULE_SETUP}"
fi
if [[ -n "${VMCNET_ENV_ACTIVATE:-}" ]]; then
  source "${VMCNET_ENV_ACTIVATE}"
fi
export PYTHONPATH="${REPOSITORY}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false VMCNET_DISABLE_CHECKPOINTS=1
export WSSR_PAIRED_SEED=0
mkdir -p "$(dirname "${RUN}")"
cd "${REPOSITORY}"
python experiments/C_rank1600_eta_memory_E5000_20260802/eval_seeded_launcher.py \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/1000.npz \
  --reload.new_optimizer_state=True --reload.reburn=False \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${RUN}" --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nepochs=0 --config.vmc.disable_checkpointing=True \
  --config.eval.nchains=1000 --config.eval.nburn=2000 \
  --config.eval.nepochs=2000 --config.eval.nsteps_per_param_update=10 \
  --config.eval.use_data_from_training=False \
  --config.eval.record_local_energies=True --config.wandb.mode=disabled

test -f "${RUN}/eval/statistics.json"
