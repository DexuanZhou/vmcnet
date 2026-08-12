#!/bin/bash
#SBATCH --job-name=N2-current-capacity-100
#SBATCH --account=def-ortner
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --array=0-2%3
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail
SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_cluster_envelope_capacity_audit_v2_20260809
REPO=/scratch/dexuan1/vmcnet
SOURCE=/scratch/dexuan1/runs/old/N2_R2016/e100000_followup/N2eq/N2eq_kfac_pre5000
ROOT=/scratch/dexuan1/runs/N2_current_capacity_timing_tailfix_20260809
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311
RANKS=(200 384 512)
RANK=${RANKS[$SLURM_ARRAY_TASK_ID]}
WORKING_RANK=$((RANK + 1))
ARM=current_rank${RANK}
RUN=${ROOT}/${ARM}
META=${ROOT}/metadata/${ARM}
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
printf 'time_ns\tepochs\tgpu_memory_mib\n' > "${META}/progress.tsv"

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
  --config.vmc.nepochs=100 \
  --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.clip_threshold=5.0 \
  --config.vmc.clip_center=mean \
  --config.vmc.check_for_nans=True \
  --config.vmc.disable_checkpointing=True \
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
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank="${WORKING_RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max="${WORKING_RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank="${WORKING_RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank="${WORKING_RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=True \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=40 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first=False \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first_force=False \
  --config.vmc.optimizer.wssr_warm_svd_right.mixed_precision_solve=False \
  --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=cluster_envelope_ritz \
  --config.vmc.optimizer.wssr_warm_svd_right.cluster_envelope_rank="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.cluster_envelope_history=1 \
  --config.wandb.mode=disabled &
pid=$!

while kill -0 "${pid}" 2>/dev/null; do
  epochs=0
  [[ -f "${RUN}/energy.txt" ]] && epochs=$(wc -l < "${RUN}/energy.txt")
  gpu_memory=$(nvidia-smi --query-compute-apps=used_memory \
    --format=csv,noheader,nounits 2>/dev/null | awk 'BEGIN{m=0} $1+0>m{m=$1+0} END{print m}')
  printf '%s\t%s\t%s\n' "$(date +%s%N)" "${epochs}" "${gpu_memory:-0}" \
    >> "${META}/progress.tsv"
  sleep 1
done
wait "${pid}"
[[ $(wc -l < "${RUN}/energy.txt") -eq 100 ]] || exit 4
nvidia-smi > "${META}/nvidia_smi_end.txt"
