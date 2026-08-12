#!/bin/bash
#SBATCH --job-name=C-r800-lr-eta
#SBATCH --account=def-ortner
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --array=0-8%3
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

REPO=/scratch/dexuan1/vmcnet
SOURCE=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1
CHECKPOINT=${SOURCE}/checkpoints/1000.npz
ROOT=/scratch/dexuan1/runs/C_wssr_rank800_lr_eta_E2000_screen_retry1
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

LONG_TEST=${LONG_TEST:-0}
TUNE2=${TUNE2:-0}
TUNE3=${TUNE3:-0}
RANK1600_TUNE=${RANK1600_TUNE:-0}
RANK1600_LONG=${RANK1600_LONG:-0}
RANK=800
if [[ "${RANK1600_LONG}" == "1" ]]; then
  ROOT=/scratch/dexuan1/runs/C_wssr_rank1600_eta03_lr004_E100000
  RANK=1600
  LRS=(0.04)
  ETAS=(0.3)
  LR_INDEX=0
  ETA_INDEX=0
  TRAIN_EPOCHS=100000
  EVAL_EPOCHS=20000
  CHECKPOINT_EVERY=5000
  PURPOSE=C_rank1600_eta03_lr004_E100000
elif [[ "${RANK1600_TUNE}" == "1" ]]; then
  ROOT=/scratch/dexuan1/runs/C_wssr_rank1600_hard_lr_eta_E10000_screen
  RANK=1600
  LRS=(0.02 0.04)
  ETAS=(0.3 0.5 0.8)
  LR_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
  ETA_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
  TRAIN_EPOCHS=10000
  EVAL_EPOCHS=0
  CHECKPOINT_EVERY=1000
  PURPOSE=C_rank1600_hard_lr_eta_E10000_screen
elif [[ "${TUNE3}" == "1" ]]; then
  ROOT=/scratch/dexuan1/runs/C_wssr_rank800_eta03_lr003_lr004_E100000
  LRS=(0.03 0.04)
  ETAS=(0.3)
  LR_INDEX=${SLURM_ARRAY_TASK_ID}
  ETA_INDEX=0
  TRAIN_EPOCHS=100000
  EVAL_EPOCHS=20000
  CHECKPOINT_EVERY=5000
  PURPOSE=C_rank800_eta03_lr_long_test
elif [[ "${TUNE2}" == "1" ]]; then
  ROOT=/scratch/dexuan1/runs/C_wssr_rank800_hard_lr_eta_E10000_tune2
  LRS=(0.03 0.04 0.06)
  ETAS=(0.3 0.4 0.5 0.6)
  LR_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
  ETA_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
  TRAIN_EPOCHS=10000
  EVAL_EPOCHS=0
  CHECKPOINT_EVERY=1000
  PURPOSE=C_rank800_hard_lr_eta_E10000_tune2
elif [[ "${LONG_TEST}" == "1" ]]; then
  ROOT=/scratch/dexuan1/runs/C_wssr_rank800_eta05_lr002_lr004_E100000
  LRS=(0.02 0.04)
  ETAS=(0.5)
  LR_INDEX=${SLURM_ARRAY_TASK_ID}
  ETA_INDEX=0
  TRAIN_EPOCHS=100000
  EVAL_EPOCHS=20000
  CHECKPOINT_EVERY=5000
  PURPOSE=C_rank800_eta05_lr_long_test
else
  LRS=(0.01 0.02 0.04)
  ETAS=(0.5 0.8 0.95)
  LR_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
  ETA_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
  TRAIN_EPOCHS=2000
  EVAL_EPOCHS=0
  CHECKPOINT_EVERY=500
  PURPOSE=C_rank800_lr_eta_short_screen
fi
LR=${LRS[${LR_INDEX}]}
ETA=${ETAS[${ETA_INDEX}]}
LR_TAG=${LR/./p}
ETA_TAG=${ETA/./p}
RUN=${ROOT}/lr${LR_TAG}_eta${ETA_TAG}
META=${RUN}_metadata

[[ -f "${CHECKPOINT}" ]] || { echo "missing ${CHECKPOINT}" >&2; exit 2; }
[[ ! -e "${RUN}" && ! -e "${META}" ]] || {
  echo "refusing to overwrite ${RUN} or ${META}" >&2
  exit 2
}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export VMCNET_PROFILE_TIMING=1
export SMOKE_TIMING_DIR="${META}"

mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd "${REPO}"
hostname > "${META}/hostname.txt"
source experiments/fir_n4096_smoke/scripts/gpu_health_check.sh \
  "${META}/nvidia_smi_start.txt"
python - <<'PY' > "${META}/jax_backend.txt"
import jax
print(jax.default_backend())
print(jax.devices())
assert jax.default_backend() == "gpu"
PY
git rev-parse HEAD > "${META}/git_commit.txt"
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"
cat > "${META}/protocol.txt" <<EOF
purpose=${PURPOSE}
system=C
electron_configuration=(4,2)
source_checkpoint=${CHECKPOINT}
checkpoint_internal_epoch=999
training_nchains=1000
training_epochs=${TRAIN_EPOCHS}
optimizer=wssr_warm_svd_right
rank=${RANK}
all_rank_fields=${RANK}
exact_first=False
svd_maxiter_initial=40
svd_maxiter_warm=2
experimental_mode=none
adaptive_complement_beta=0.0
complement_weight=0.0
learning_rate=${LR}
eta=${ETA}
damping=0.0003
norm_constraint=0.001
schedule_type=inverse_time
learning_decay_rate=0.0001
reload_new_optimizer_state=True
reload_reburn=False
evaluation_epochs=${EVAL_EPOCHS}
EOF

vmc-molecule \
  --reload.logdir="${SOURCE}" \
  --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/1000.npz \
  --reload.new_optimizer_state=True \
  --reload.reburn=False \
  --reload.append=False \
  --reload.same_logdir=False \
  --config.logdir="${RUN}" \
  --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name=NONE \
  --config.problem.nelec='(4,2)' \
  --config.vmc.nchains=1000 \
  --config.vmc.nepochs="${TRAIN_EPOCHS}" \
  --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.clip_center=mean \
  --config.vmc.clip_threshold=5.0 \
  --config.vmc.check_for_nans=True \
  --config.vmc.checkpoint_every="${CHECKPOINT_EVERY}" \
  --config.vmc.best_checkpoint_every="${CHECKPOINT_EVERY}" \
  --config.eval.use_data_from_training=False \
  --config.eval.nchains=2000 \
  --config.eval.nburn=5000 \
  --config.eval.nepochs="${EVAL_EPOCHS}" \
  --config.eval.nsteps_per_param_update=10 \
  --config.eval.record_local_energies=True \
  --config.vmc.optimizer_type=wssr_warm_svd_right \
  --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_rate="${LR}" \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta="${ETA}" \
  --config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov \
  --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True \
  --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=True \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=40 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first=False \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first_force=False \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_reference_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.update_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.burst_diagnostics_payload=False \
  --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=none \
  --config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_beta=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0.0 \
  --config.wandb.mode=disabled

[[ -f "${RUN}/checkpoints/${TRAIN_EPOCHS}.npz" ]] || {
  echo "missing final checkpoint" >&2
  exit 4
}
if [[ "${EVAL_EPOCHS}" -gt 0 ]]; then
  [[ -f "${RUN}/eval/statistics.json" ]] || {
    echo "missing frozen-evaluation statistics" >&2
    exit 5
  }
  [[ -f "${RUN}/eval/local_energies.txt" ]] || {
    echo "missing frozen local energies" >&2
    exit 5
  }
fi
