#!/bin/bash
# Paired transition-only WSSR basis-refresh test from the matched scale-0.01
# three-level L1 -> L2 checkpoints.
#SBATCH --job-name=C-WSSR-ML-refresh
#SBATCH --account=def-ortner
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gpus-per-node=h100_3g.40gb:1
#SBATCH --array=0-3%4
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_C_arch_multilevel_fivelevel_20260808
REPO=/scratch/dexuan1/vmcnet
SOURCE_ROOT=/scratch/dexuan1/runs/C_ferminet_multilevel_fair_scale001_20260808
ROOT=/scratch/dexuan1/runs/C_wssr_transition_basis_refresh_20260808
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

case "${SLURM_ARRAY_TASK_ID}" in
  0) REPLICATE=0; MODE=warm2;  FIRST_ITERS=2 ;;
  1) REPLICATE=0; MODE=refresh40; FIRST_ITERS=40 ;;
  2) REPLICATE=1; MODE=warm2;  FIRST_ITERS=2 ;;
  3) REPLICATE=1; MODE=refresh40; FIRST_ITERS=40 ;;
  *) exit 2 ;;
esac

SOURCE=${SOURCE_ROOT}/three_level/seed${REPLICATE}/converted_L2
ARM=${ROOT}/seed${REPLICATE}/${MODE}
FIRST=${ARM}/first_step
CONTINUE=${ARM}/continue_to1200
[[ -f "${SOURCE}/multilevel.npz" ]] || exit 2
[[ ! -e "${ARM}" ]] || { echo "refusing to overwrite ${ARM}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export PYTHONPATH="${SNAPSHOT}"
export GIT_DIR="${REPO}/.git" GIT_WORK_TREE="${SNAPSHOT}"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${ARM}" /scratch/dexuan1/runs/logs
cd "${SNAPSHOT}"

# Only this first post-transition update differs between the paired arms.
# Forty current-operator subspace iterations provide a transition refresh
# while retaining the prolonged WSSR history and fixed rank 800.
vmc-molecule \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=multilevel.npz \
  --reload.new_optimizer_state=False --reload.reburn=False \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${FIRST}" --config.base_logdir="${FIRST}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nepochs=1001 --config.vmc.nburn=0 \
  --config.vmc.checkpoint_every=1001 --config.vmc.best_checkpoint_every=1001 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm="${FIRST_ITERS}" \
  --config.eval.nepochs=0 --config.wandb.mode=disabled
test -f "${FIRST}/checkpoints/1001.npz"

# Return both arms to the production warm-SSI2 update for the remaining 199
# epochs.  This isolates the effect of the transition refresh.
vmc-molecule \
  --reload.logdir="${FIRST}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/1001.npz \
  --reload.new_optimizer_state=False --reload.reburn=False \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${CONTINUE}" --config.base_logdir="${CONTINUE}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nepochs=1200 --config.vmc.nburn=0 \
  --config.vmc.checkpoint_every=1200 --config.vmc.best_checkpoint_every=1200 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 \
  --config.eval.nepochs=0 --config.wandb.mode=disabled
test -f "${CONTINUE}/checkpoints/1200.npz"
