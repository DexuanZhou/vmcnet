#!/bin/bash
#SBATCH --job-name=C-r1600-lam-E500
#SBATCH --account=def-ortner
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-1%2
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

SOURCE=/scratch/dexuan1/runs/C_wssr_rank1600_eta03_lr004_E100000_resume1
ROOT=/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage1_E500_retry1

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    LABEL=legacy_hard
    RELATIVE_CUTOFF=-1.0
    TIKHONOV_LAMBDA=-1.0
    ;;
  1)
    LABEL=fixed_lambda_1e-3
    RELATIVE_CUTOFF=0.0003
    TIKHONOV_LAMBDA=0.001
    ;;
  *)
    echo "unexpected array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

RUN=${ROOT}/${LABEL}
META=${ROOT}/metadata/${LABEL}
CHECKPOINT=${SOURCE}/checkpoints/100000.npz

test -f "${CHECKPOINT}"
[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite ${RUN}" >&2; exit 2; }
[[ ! -e "${META}" ]] || { echo "refusing to overwrite ${META}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd /scratch/dexuan1/vmcnet
hostname > "${META}/hostname.txt"
git rev-parse HEAD > "${META}/git_commit.txt"
git status --short > "${META}/git_status_short.txt"
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"
printf '%s\n' \
  "scientific_question=does_decoupling_the_singular_value_cutoff_from_a_fixed_tikhonov_lambda_improve_rank1600_WSSR" \
  "label=${LABEL}" \
  "source_checkpoint=${CHECKPOINT}" \
  "source_epoch=100000" \
  "target_epoch=100500" \
  "training_steps=500" \
  "rank=1600" \
  "eta=0.3" \
  "learning_rate=0.04" \
  "learning_decay_rate=0.0001" \
  "legacy_damping=0.0003" \
  "relative_singular_value_cutoff=${RELATIVE_CUTOFF}" \
  "tikhonov_lambda=${TIKHONOV_LAMBDA}" \
  "norm_constraint=0.001" \
  "svd_maxiter_initial=40" \
  "svd_maxiter_warm=2" \
  "complement_weight=0.0" \
  "reload_new_optimizer_state=false" \
  "reload_reburn=false" \
  > "${META}/protocol.txt"

vmc-molecule \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/100000.npz \
  --reload.new_optimizer_state=False --reload.reburn=False \
  --reload.append=True --reload.same_logdir=False \
  --config.logdir="${RUN}" --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nepochs=100500 --config.vmc.check_for_nans=True \
  --config.vmc.checkpoint_every=500 --config.vmc.best_checkpoint_every=500 \
  --config.eval.nburn=0 --config.eval.nepochs=0 \
  --config.vmc.optimizer_type=wssr_warm_svd_right \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.04 \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta=0.3 \
  --config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.relative_singular_value_cutoff="${RELATIVE_CUTOFF}" \
  --config.vmc.optimizer.wssr_warm_svd_right.tikhonov_lambda="${TIKHONOV_LAMBDA}" \
  --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=40 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 \
  --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=none \
  --config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_beta=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_state_decay=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_state_relative_cap=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.multilevel_complement_period=1 \
  --config.vmc.optimizer.wssr_warm_svd_right.multilevel_complement_cosine_threshold=0.0 \
  --config.wandb.mode=disabled

test -f "${RUN}/checkpoints/100500.npz"
test -f "${RUN}/training_metrics.csv"
