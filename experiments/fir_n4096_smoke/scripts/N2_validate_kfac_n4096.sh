#!/bin/bash
#SBATCH --job-name=N2-kfac4096-validate
#SBATCH --account=def-ortner
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
set -euo pipefail
REPO=/scratch/dexuan1/vmcnet
PKG=${REPO}/experiments/fir_n4096_smoke
ROOT=${CORRECTED_ROOT:-/scratch/dexuan1/runs/fir_n4096_corrected/N2}
SOURCE=${ROOT}/kfac_preliminary
EVAL=${ROOT}/validation_eval
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311
[[ -f "${SOURCE}/checkpoints/5000.npz" ]] || { echo 'Missing corrected KFAC checkpoint' >&2; exit 2; }
[[ ! -e "${EVAL}" ]] || { echo 'Refusing to overwrite validation eval' >&2; exit 2; }
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true XLA_PYTHON_CLIENT_PREALLOCATE=false VMCNET_DISABLE_CHECKPOINTS=1
cd "${REPO}"
mkdir -p "${ROOT}/metadata/validation"
source "${PKG}/scripts/gpu_health_check.sh" "${ROOT}/metadata/validation/nvidia_smi_start.txt"
python "${PKG}/scripts/validate_n2_kfac_checkpoint.py" pre
vmc-molecule \
 --reload.logdir="${SOURCE}" --reload.checkpoint_relative_file_path=checkpoints/5000.npz \
 --reload.use_config_file=True --reload.use_checkpoint_file=True --reload.new_optimizer_state=True \
 --reload.reburn=False --reload.append=False --reload.same_logdir=False \
 --config.logdir="${EVAL}" --config.base_logdir="${EVAL}" --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
 --config.vmc.nchains=4096 --config.vmc.nepochs=0 --config.vmc.disable_checkpointing=True \
 --config.eval.nchains=4096 --config.eval.use_data_from_training=True --config.eval.nburn=0 --config.eval.nepochs=20 \
 --config.eval.nsteps_per_param_update=10 --config.wandb.mode=disabled
python "${PKG}/scripts/validate_n2_kfac_checkpoint.py" post
