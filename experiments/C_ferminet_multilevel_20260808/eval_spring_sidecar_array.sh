#!/bin/bash
# Frozen evaluation for the matched scale-0.01 direct and multilevel SPRING arms.
#SBATCH --job-name=C-ML-sidecar-eval
#SBATCH --account=def-ortner
#SBATCH --time=00:35:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100_3g.40gb:1
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_C_arch_multilevel_sidecar_20260808
REPO=/scratch/dexuan1/vmcnet
HERE=${SNAPSHOT}/experiments/C_ferminet_multilevel_20260808
ROOT=/scratch/dexuan1/runs/C_ferminet_multilevel_pilot_20260808
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    SOURCE=${ROOT}/spring_sidecar_gate/direct_L2_scale001/train_L2_to2000
    RUN=${ROOT}/frozen_sidecar/spring/direct_L2_scale001
    ;;
  1)
    # The first pilot continuation was written to the `_1` suffix because its
    # launcher pre-created the nominal output directory.  Prefer the nominal
    # path for corrected future launches, but retain compatibility with that
    # completed numerical run.
    SOURCE=${ROOT}/spring_sidecar_gate/scale001/train_L2_to2000
    if [[ ! -f "${SOURCE}/checkpoints/2000.npz" ]]; then
      SOURCE=${ROOT}/spring_sidecar_gate/scale001/train_L2_to2000_1
    fi
    RUN=${ROOT}/frozen_sidecar/spring/multilevel_scale001
    ;;
  *) exit 2 ;;
esac

[[ -f "${SOURCE}/checkpoints/2000.npz" ]] || exit 2
[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite ${RUN}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export PYTHONPATH="${SNAPSHOT}"
export GIT_DIR="${REPO}/.git" GIT_WORK_TREE="${SNAPSHOT}"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false VMCNET_DISABLE_CHECKPOINTS=1
mkdir -p "$(dirname "${RUN}")" /scratch/dexuan1/runs/logs
cd "${SNAPSHOT}"

python "${HERE}/eval_seeded_launcher.py" \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/2000.npz \
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
