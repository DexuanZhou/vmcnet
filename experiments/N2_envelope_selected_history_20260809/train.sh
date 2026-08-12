#!/bin/bash
#SBATCH --job-name=N2-env-selected-E1000
#SBATCH --account=def-ortner
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail
SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_cluster_envelope_selected_20260809
REPO=/scratch/dexuan1/vmcnet
SOURCE=/scratch/dexuan1/runs/old/N2_R2016/e100000_followup/N2eq/N2eq_kfac_pre5000
ROOT=/scratch/dexuan1/runs/N2_envelope_selected_history_20260809
RUN=${ROOT}/S96_current32_pool160_select64
META=${ROOT}/metadata/S96_current32_pool160_select64
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311
[[ -f "${SOURCE}/checkpoints/5000.npz" ]] || exit 2
[[ -d "${SNAPSHOT}" ]] || exit 2
[[ ! -e "${RUN}" && ! -e "${META}" ]] || exit 2

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
sha256sum vmcnet/updates/wssr.py vmcnet/updates/wssr_experimental.py \
  vmcnet/train/default_config.py vmcnet/train/parse_config_flags.py \
  > "${META}/source_sha256.txt"

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
  --config.vmc.nepochs=1000 \
  --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.clip_threshold=5.0 \
  --config.vmc.clip_center=mean \
  --config.vmc.check_for_nans=True \
  --config.vmc.disable_checkpointing=False \
  --config.vmc.checkpoint_every=1000 \
  --config.vmc.best_checkpoint_every=1000 \
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
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=32 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=32 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank=32 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=32 \
  --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=True \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=40 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first=False \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first_force=False \
  --config.vmc.optimizer.wssr_warm_svd_right.mixed_precision_solve=False \
  --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=cluster_envelope_ritz_selected \
  --config.wandb.mode=disabled

[[ -f "${RUN}/checkpoints/1000.npz" ]] || exit 4
nvidia-smi > "${META}/nvidia_smi_end.txt"
