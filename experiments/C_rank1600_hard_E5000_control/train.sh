#!/bin/bash
#SBATCH --job-name=C-r1600-hard-ctl
#SBATCH --account=def-ortner
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail
SOURCE=/scratch/dexuan1/runs/C_wssr_rank1600_eta03_lr004_E100000_resume1
RUN=/scratch/dexuan1/runs/C_wssr_rank1600_hard_matched_control_E5000
META=${RUN}_metadata
test -f "${SOURCE}/checkpoints/100000.npz"
for p in "${RUN}" "${META}"; do
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
purpose=matched_hard_only_control_for_multilevel_complement
source_checkpoint=${SOURCE}/checkpoints/100000.npz
source_epoch=100000
target_epoch=105000
rank=1600
eta=0.3
learning_rate=0.04
learning_decay_rate=0.0001
damping=0.0003
svd_maxiter_initial=40
svd_maxiter_warm=2
experimental_mode=none
complement_weight=0.0
adaptive_complement_beta=0.0
reload_new_optimizer_state=false
reload_reburn=false
EOF

vmc-molecule \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/100000.npz \
  --reload.new_optimizer_state=False --reload.reburn=False \
  --reload.append=True --reload.same_logdir=False \
  --config.logdir="${RUN}" --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nepochs=105000 --config.vmc.check_for_nans=True \
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
  --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=none \
  --config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_beta=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_state_decay=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_state_relative_cap=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.multilevel_complement_period=1 \
  --config.vmc.optimizer.wssr_warm_svd_right.multilevel_complement_cosine_threshold=0.0 \
  --config.wandb.mode=disabled

for epoch in 101000 102000 103000 104000 105000; do
  test -f "${RUN}/checkpoints/${epoch}.npz"
done
