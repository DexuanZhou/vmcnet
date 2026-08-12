#!/bin/bash
#SBATCH --job-name=C-switch-WSSR-5k
#SBATCH --account=def-ortner
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-1%2
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_C_stagewise_switch_20260807
REPO=/scratch/dexuan1/vmcnet
SOURCE=/scratch/dexuan1/runs/C_spring_official_kfac1000_E100000_replicate2
ROOT=/scratch/dexuan1/runs/C_stagewise_switch_scan_20260807
UPDATES=5000

case "${SLURM_ARRAY_TASK_ID}" in
  0) SOURCE_EPOCH=5000; ARM=switch5k; DENOM=1.5 ;;
  1) SOURCE_EPOCH=20000; ARM=switch20k; DENOM=3.0 ;;
  *) echo "unexpected task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

LR=$(awk -v d="${DENOM}" 'BEGIN {printf "%.17g", 0.04/d}')
DECAY=$(awk -v d="${DENOM}" 'BEGIN {printf "%.17g", 1e-4/d}')
RUN=${ROOT}/train/${ARM}
META=${ROOT}/metadata/${ARM}

test -f "${SOURCE}/checkpoints/${SOURCE_EPOCH}.npz"
test -f "${SNAPSHOT}/SOURCE_SNAPSHOT.txt"
[[ ! -e "${RUN}" && ! -e "${META}" ]] || {
  echo "refusing to overwrite ${RUN} or ${META}" >&2
  exit 2
}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export PYTHONPATH="${SNAPSHOT}"
export GIT_DIR="${REPO}/.git"
export GIT_WORK_TREE="${SNAPSHOT}"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd "${SNAPSHOT}"
cp SOURCE_SNAPSHOT.txt "${META}/"
hostname > "${META}/hostname.txt"
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"
cat > "${META}/protocol.txt" <<EOF
question=when_does_SPRING_to_WSSR_switch_become_beneficial
source=${SOURCE}/checkpoints/${SOURCE_EPOCH}.npz
source_optimizer=spring
source_epoch=${SOURCE_EPOCH}
arm=${ARM}
updates=${UPDATES}
new_optimizer_state=true
reburn=false
nchains=1000
rank=1600
shared_eta=0.3
learning_rate_local=${LR}
learning_decay_rate_local=${DECAY}
equivalent_global_schedule=0.04/(1+1e-4*(${SOURCE_EPOCH}+s))
tikhonov_lambda=0.001
relative_cutoff=0.0003
norm_constraint=0.001
EOF

vmc-molecule \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path="checkpoints/${SOURCE_EPOCH}.npz" \
  --reload.new_optimizer_state=True --reload.reburn=False \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${RUN}" --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nchains=1000 --config.vmc.nburn=0 \
  --config.vmc.nepochs="${UPDATES}" --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.check_for_nans=True \
  --config.vmc.checkpoint_every="${UPDATES}" \
  --config.vmc.best_checkpoint_every="${UPDATES}" \
  --config.eval.nepochs=0 \
  --config.vmc.optimizer_type=wssr_warm_svd_right \
  --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_rate="${LR}" \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate="${DECAY}" \
  --config.vmc.optimizer.wssr_warm_svd_right.eta=0.3 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_S=-1.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_g=-1.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_bias_correction=True \
  --config.vmc.optimizer.wssr_warm_svd_right.adaptive_S_average=False \
  --config.vmc.optimizer.wssr_warm_svd_right.adaptive_g_average=False \
  --config.vmc.optimizer.wssr_warm_svd_right.enable_gradient_transport=False \
  --config.vmc.optimizer.wssr_warm_svd_right.solution_recurrence_mode=none \
  --config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.relative_singular_value_cutoff=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.tikhonov_lambda=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov \
  --config.vmc.optimizer.wssr_warm_svd_right.mixed_precision_solve=True \
  --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True \
  --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=40 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 \
  --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=True \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first=False \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_reference_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.update_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.reliability_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=none \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_beta=0.0 \
  --config.wandb.mode=disabled

test -f "${RUN}/checkpoints/${UPDATES}.npz"
