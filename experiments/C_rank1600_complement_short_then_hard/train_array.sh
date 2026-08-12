#!/bin/bash
#SBATCH --job-name=C-comp-short-hard
#SBATCH --account=def-ortner
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-1%2
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail
case "${SLURM_ARRAY_TASK_ID}" in
  0) CUTOFF=101000; LABEL=on1000 ;;
  1) CUTOFF=102000; LABEL=on2000 ;;
  *) exit 2 ;;
esac

BASE=/scratch/dexuan1/runs/C_wssr_rank1600_eta03_lr004_E100000_resume1
EMA=/scratch/dexuan1/runs/C_wssr_rank1600_beta0001_rho099_${LABEL}_stage1
HARD=/scratch/dexuan1/runs/C_wssr_rank1600_beta0001_rho099_${LABEL}_then_hard_E5000
META=${HARD}_metadata
[[ -f "${BASE}/checkpoints/100000.npz" ]] || exit 2
for p in "${EMA}" "${HARD}" "${META}"; do
  [[ ! -e "${p}" ]] || { echo "refusing to overwrite ${p}" >&2; exit 2; }
done

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd /scratch/dexuan1/vmcnet
hostname > "${META}/hostname.txt"
git rev-parse HEAD > "${META}/git_commit.txt"
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"
cat > "${META}/protocol.txt" <<EOF
source=${BASE}/checkpoints/100000.npz
complement_enabled_through=${CUTOFF}
final_epoch=105000
rank=1600
eta=0.3
learning_rate=0.04
beta_when_enabled=0.001
rho_when_enabled=0.99
state_relative_cap=0.01
stage2=hard_complement_weight_zero_and_no_ema
reburn=false
preserve_core_and_optax_state=true
EOF

# Stage 1: beta-capped complement EMA.
vmc-molecule \
  --reload.logdir="${BASE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/100000.npz \
  --reload.new_optimizer_state=False --reload.reburn=False \
  --reload.append=True --reload.same_logdir=False \
  --config.logdir="${EMA}" --config.base_logdir="${EMA}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nepochs="${CUTOFF}" --config.vmc.check_for_nans=True \
  --config.vmc.checkpoint_every=1000 --config.vmc.best_checkpoint_every=5000 \
  --config.eval.nburn=0 --config.eval.nepochs=0 \
  --config.vmc.optimizer_type=wssr_warm_svd_right \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.04 \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta=0.3 \
  --config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=40 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 \
  --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=adaptive_complement \
  --config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_beta=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0.0001 \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_state_decay=0.99 \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_state_relative_cap=0.01 \
  --config.wandb.mode=disabled
test -f "${EMA}/checkpoints/${CUTOFF}.npz"

# Stage 2: discard EMA state and continue with the genuine hard update.
vmc-molecule \
  --reload.logdir="${EMA}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path="checkpoints/${CUTOFF}.npz" \
  --reload.new_optimizer_state=False --reload.reburn=False \
  --reload.append=True --reload.same_logdir=False \
  --config.logdir="${HARD}" --config.base_logdir="${HARD}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nepochs=105000 --config.vmc.check_for_nans=True \
  --config.vmc.checkpoint_every=1000 --config.vmc.best_checkpoint_every=5000 \
  --config.eval.nburn=0 --config.eval.nepochs=0 \
  --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=none \
  --config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_beta=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_state_decay=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_state_relative_cap=0.0 \
  --config.wandb.mode=disabled

for epoch in $(seq $((CUTOFF + 1000)) 1000 105000); do
  test -f "${HARD}/checkpoints/${epoch}.npz"
done
