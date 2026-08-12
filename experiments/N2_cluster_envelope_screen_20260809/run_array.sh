#!/bin/bash
#SBATCH --job-name=N2-cluster-env
#SBATCH --account=def-ortner
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --array=0-1%2
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

HERE=/scratch/dexuan1/vmcnet/experiments/N2_cluster_envelope_screen_20260809
SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_cluster_envelope_20260809
REPO=/scratch/dexuan1/vmcnet
SOURCE=/scratch/dexuan1/runs/old/N2_R2016/e100000_followup/N2eq/N2eq_kfac_pre5000
ROOT=${ROOT:-/scratch/dexuan1/runs/N2_cluster_envelope_screen_20260809_retry1}
NEPOCHS=${NEPOCHS:-200}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-200}
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

case "${SLURM_ARRAY_TASK_ID}" in
  0) ARM=A_hard_rank200; RANK=200; MODE=none ;;
  1) ARM=B_cluster_envelope; RANK=32; MODE=cluster_envelope ;;
  *) exit 2 ;;
esac

RUN=${ROOT}/${ARM}
META=${ROOT}/metadata/${ARM}
[[ -d "${SNAPSHOT}" ]] || { echo "missing snapshot ${SNAPSHOT}" >&2; exit 2; }
[[ -f "${SOURCE}/checkpoints/5000.npz" ]] || exit 2
[[ ! -e "${RUN}" && ! -e "${META}" ]] || {
  echo "refusing to overwrite ${RUN} or ${META}" >&2
  exit 2
}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export PYTHONPATH="${SNAPSHOT}"
export GIT_DIR="${REPO}/.git"
export GIT_WORK_TREE="${SNAPSHOT}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd "${SNAPSHOT}"

hostname > "${META}/hostname.txt"
nvidia-smi > "${META}/nvidia_smi_start.txt"
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"
cp "${HERE}/manifest.tsv" "${META}/manifest.tsv"
sha256sum vmcnet/updates/wssr.py vmcnet/updates/wssr_experimental.py \
  vmcnet/train/default_config.py > "${META}/source_sha256.txt"

vmc-molecule \
  --reload.logdir="${SOURCE}" \
  --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/5000.npz \
  --reload.new_optimizer_state=True \
  --reload.reburn=False \
  --reload.append=False \
  --reload.same_logdir=False \
  --config.logdir="${RUN}" \
  --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name=NONE \
  --config.problem.ion_pos='((0.,0.,-1.008),(0.,0.,1.008))' \
  --config.problem.ion_charges='(7.,7.)' \
  --config.problem.nelec='(7,7)' \
  --config.vmc.nchains=1000 \
  --config.vmc.nepochs="${NEPOCHS}" \
  --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.clip_threshold=5.0 \
  --config.vmc.clip_center=mean \
  --config.vmc.check_for_nans=True \
  --config.vmc.disable_checkpointing=False \
  --config.vmc.checkpoint_every="${CHECKPOINT_EVERY}" \
  --config.vmc.best_checkpoint_every="${CHECKPOINT_EVERY}" \
  --config.eval.nepochs=0 \
  --config.eval.nburn=0 \
  --config.vmc.optimizer_type=wssr_warm_svd_right \
  --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.002 \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_S=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_g=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.relative_singular_value_cutoff=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.tikhonov_lambda=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0.0 \
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
  --config.vmc.optimizer.wssr_warm_svd_right.mixed_precision_solve=False \
  --config.vmc.optimizer.wssr_warm_svd_right.solution_recurrence_mode=none \
  --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode="${MODE}" \
  --config.wandb.mode=disabled

[[ -f "${RUN}/checkpoints/${NEPOCHS}.npz" ]] || exit 4
nvidia-smi > "${META}/nvidia_smi_end.txt"
