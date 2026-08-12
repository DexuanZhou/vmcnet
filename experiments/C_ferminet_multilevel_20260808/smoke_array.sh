#!/bin/bash
#SBATCH --job-name=C-ML-smoke
#SBATCH --account=def-ortner
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --array=0-1%2
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

HERE=/scratch/dexuan1/vmcnet/experiments/C_ferminet_multilevel_20260808
ROOT=/scratch/dexuan1/runs/C_ferminet_multilevel_smoke_20260808
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

case "${SLURM_ARRAY_TASK_ID}" in
  0) OPTIMIZER=spring ;;
  1) OPTIMIZER=wssr_warm_svd_right ;;
  *) exit 2 ;;
esac

ARM=${ROOT}/${OPTIMIZER}
COARSE=${ARM}/coarse
CONVERTED=${ARM}/converted
FINE=${ARM}/fine_continue
[[ ! -e "${ARM}" ]] || { echo "refusing to overwrite ${ARM}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${ARM}" /scratch/dexuan1/runs/logs
cd /scratch/dexuan1/vmcnet
python "${HERE}/prepare_coarse_config.py" --output-dir="${ARM}"

COMMON=(
  --presets.path="${ARM}/coarse_preset.json"
  --config.logdir="${COARSE}"
  --config.base_logdir="${COARSE}"
  --config.save_to_current_datetime_subfolder=False
  --config.subfolder_name=NONE
  --config.initial_seed=17
  --config.problem.ion_pos='((0.,0.,0.),)'
  --config.problem.ion_charges='(6.,)'
  --config.problem.nelec='(4,2)'
  --config.vmc.nchains=32
  --config.vmc.nburn=10
  --config.vmc.nepochs=2
  --config.vmc.nsteps_per_param_update=1
  --config.vmc.clip_threshold=5.0
  --config.vmc.check_for_nans=True
  --config.vmc.checkpoint_every=2
  --config.vmc.best_checkpoint_every=2
  --config.eval.nepochs=0
  --config.wandb.mode=disabled
)

if [[ "${OPTIMIZER}" == spring ]]; then
  vmc-molecule "${COMMON[@]}" \
    --config.vmc.optimizer_type=spring \
    --config.vmc.optimizer.spring.schedule_type=inverse_time \
    --config.vmc.optimizer.spring.learning_rate=0.02 \
    --config.vmc.optimizer.spring.learning_decay_rate=0.0001 \
    --config.vmc.optimizer.spring.mu=0.99 \
    --config.vmc.optimizer.spring.damping=0.001 \
    --config.vmc.optimizer.spring.constrain_norm=True \
    --config.vmc.optimizer.spring.norm_constraint=0.001
else
  vmc-molecule "${COMMON[@]}" \
    --config.vmc.optimizer_type=wssr_warm_svd_right \
    --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time \
    --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.02 \
    --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001 \
    --config.vmc.optimizer.wssr_warm_svd_right.eta=0.0 \
    --config.vmc.optimizer.wssr_warm_svd_right.eta_S=0.0 \
    --config.vmc.optimizer.wssr_warm_svd_right.eta_g=0.0 \
    --config.vmc.optimizer.wssr_warm_svd_right.damping=0.001 \
    --config.vmc.optimizer.wssr_warm_svd_right.tikhonov_lambda=0.001 \
    --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov \
    --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True \
    --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001 \
    --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=16 \
    --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=16 \
    --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank=16 \
    --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=16 \
    --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=4 \
    --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 \
    --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=True \
    --config.vmc.optimizer.wssr_warm_svd_right.exact_first=False \
    --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0.0 \
    --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=none
fi

test -f "${COARSE}/checkpoints/2.npz"
python "${HERE}/prepare_fine_config.py" \
  --coarse-config="${COARSE}/config.json" --output-dir="${ARM}"
python -m vmcnet.train.multilevel_checkpoint \
  --coarse-config="${COARSE}/config.json" \
  --fine-config="${ARM}/fine_config.json" \
  --input-checkpoint="${COARSE}/checkpoints/2.npz" \
  --output-dir="${CONVERTED}" --output-name=multilevel.npz \
  --template-seed=314159 --verify-samples=8

vmc-molecule \
  --reload.logdir="${CONVERTED}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=multilevel.npz \
  --reload.new_optimizer_state=False --reload.reburn=False \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${FINE}" --config.base_logdir="${FINE}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nchains=32 --config.vmc.nburn=0 \
  --config.vmc.nepochs=11 --config.vmc.nsteps_per_param_update=1 \
  --config.vmc.check_for_nans=True \
  --config.vmc.checkpoint_every=11 --config.vmc.best_checkpoint_every=11 \
  --config.eval.nepochs=0 --config.wandb.mode=disabled

test -f "${FINE}/checkpoints/11.npz"
test "$(($(wc -l < "${FINE}/training_metrics.csv") - 1))" -eq 10
