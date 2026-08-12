#!/bin/bash
#SBATCH --job-name=C-r1600-prox-E500
#SBATCH --account=def-ortner
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

SOURCE=/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage2_E2000
CHECKPOINT=${SOURCE}/checkpoints/102000.npz
ROOT=/scratch/dexuan1/runs/C_wssr_rank1600_native_proximal_gamma3e4_matched_E500_20260802

labels=(hard_control native_proximal)
modes=(none native_proximal)
gammas=(0.0 0.0003)
label=${labels[${SLURM_ARRAY_TASK_ID}]}
mode=${modes[${SLURM_ARRAY_TASK_ID}]}
gamma=${gammas[${SLURM_ARRAY_TASK_ID}]}
RUN=${ROOT}/${label}
META=${RUN}_metadata

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
scontrol show job "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
  > "${META}/slurm_job.txt"
printf '%s\n' \
  "scientific_question=does_native_factor_temporal_proximal_regularization_improve_rank1600_WSSR_over_a_matched_hard_control" \
  "variant=${label}" \
  "source_checkpoint=${CHECKPOINT}" \
  "source_epoch=102000" \
  "target_epoch=102500" \
  "additional_training_steps=500" \
  "rank=1600" \
  "eta=0.3" \
  "learning_rate=0.04" \
  "learning_decay_rate=0.0001" \
  "damping=0.0003" \
  "relative_singular_value_cutoff=0.0003" \
  "tikhonov_lambda=0.001" \
  "norm_constraint=0.001" \
  "svd_maxiter_warm=2" \
  "complement_weight=0.0" \
  "experimental_mode=${mode}" \
  "native_proximal_gamma=${gamma}" \
  "reload_new_optimizer_state=false" \
  "reload_reburn=false" \
  "note=the_new_proximal_previous_direction_field_is_zero_only_on_the_first_continuation_step_then_persists" \
  > "${META}/protocol.txt"

vmc-molecule \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/102000.npz \
  --reload.new_optimizer_state=False --reload.reburn=False \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${RUN}" --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nepochs=102500 --config.vmc.check_for_nans=True \
  --config.vmc.checkpoint_every=500 --config.vmc.best_checkpoint_every=500 \
  --config.eval.nburn=0 --config.eval.nepochs=0 \
  --config.vmc.optimizer_type=wssr_warm_svd_right \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.04 \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta=0.3 \
  --config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.relative_singular_value_cutoff=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.tikhonov_lambda=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=40 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 \
  --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode="${mode}" \
  --config.vmc.optimizer.wssr_warm_svd_right.native_proximal_gamma="${gamma}" \
  --config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_beta=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_state_decay=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_state_relative_cap=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.multilevel_complement_period=1 \
  --config.vmc.optimizer.wssr_warm_svd_right.multilevel_complement_cosine_threshold=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.update_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.reliability_diagnostics=False \
  --config.wandb.mode=disabled

test -f "${RUN}/checkpoints/102500.npz"
test -f "${RUN}/training_metrics.csv"
