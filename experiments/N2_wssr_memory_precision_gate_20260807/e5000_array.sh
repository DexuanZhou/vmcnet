#!/bin/bash
#SBATCH --job-name=N2-wssr-rec-E5k
#SBATCH --account=def-ortner
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-3
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

set -euo pipefail

SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_N2_WSSR_semimf_gate_20260807
REPO=/scratch/dexuan1/vmcnet
SOURCE=/scratch/dexuan1/runs/old/N2_R2016/e100000_followup/N2eq/N2eq_kfac_pre5000
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

case "${SLURM_ARRAY_TASK_ID}" in
  0) RANK=800; SUBSPACE_ETA=0.0 ;;
  1) RANK=800; SUBSPACE_ETA=0.2 ;;
  2) RANK=1000; SUBSPACE_ETA=0.0 ;;
  3) RANK=1000; SUBSPACE_ETA=0.2 ;;
  *) echo "invalid task" >&2; exit 2 ;;
esac

ETA_TAG=${SUBSPACE_ETA/./p}
RUN=/scratch/dexuan1/runs/N2_R2016_WSSR_fullres_r${RANK}_subeta${ETA_TAG}_E5000_20260807
META=${RUN}_metadata
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
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd "${SNAPSHOT}"

hostname > "${META}/hostname.txt"
nvidia-smi > "${META}/nvidia_smi_start.txt"
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"
cp SOURCE_SNAPSHOT.txt "${META}/"

(
  while true; do
    nvidia-smi --query-compute-apps=timestamp,pid,used_memory \
      --format=csv,noheader,nounits || true
    sleep 2
  done
) > "${META}/gpu_memory_mib.csv" &
MONITOR_PID=$!
trap 'kill ${MONITOR_PID} 2>/dev/null || true' EXIT

cat > "${META}/protocol.txt" <<EOF
purpose=N2 E5000 precision screen after rank1600 memory gate
system=N2
bond_length_bohr=2.016
source_checkpoint=${SOURCE}/checkpoints/5000.npz
training_steps=5000
walkers=1000
mcmc_steps=10
reburn=False
rank=${RANK}
subspace_eta_S=${SUBSPACE_ETA}
eta_S=0.0
eta_g=0.0
solution_recurrence_mode=residual
residual_evaluation=full_current_batch
solution_mu=0.95
learning_rate=0.002
learning_decay_rate=0.0001
tikhonov_lambda=0.001
relative_singular_value_cutoff=0.0003
norm_constraint=0.001
SSI=40/2
store_warm_u=False
semi_matrix_free_augmented=True
frozen_walkers=2000
frozen_burn_in=10000
frozen_measurements=2000
frozen_mcmc_steps=10
EOF

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
  --config.vmc.nepochs=5000 \
  --config.vmc.checkpoint_every=2500 \
  --config.vmc.best_checkpoint_every=2500 \
  --config.eval.use_data_from_training=False \
  --config.eval.nchains=2000 \
  --config.eval.nburn=10000 \
  --config.eval.nepochs=2000 \
  --config.eval.nsteps_per_param_update=10 \
  --config.eval.record_local_energies=True \
  --config.vmc.optimizer_type=wssr_warm_svd_right \
  --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.002 \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_S=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_g=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.subspace_eta_S="${SUBSPACE_ETA}" \
  --config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.relative_singular_value_cutoff=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.tikhonov_lambda=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.mixed_precision_solve=True \
  --config.vmc.optimizer.wssr_warm_svd_right.solution_recurrence_mode=residual \
  --config.vmc.optimizer.wssr_warm_svd_right.solution_recurrence_mu=0.95 \
  --config.vmc.optimizer.wssr_warm_svd_right.residual_evaluation=full_current_batch \
  --config.vmc.optimizer.wssr_warm_svd_right.recurrence_telemetry=True \
  --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True \
  --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=False \
  --config.vmc.optimizer.wssr_warm_svd_right.semi_matrix_free_augmented=True \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=40 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first=False \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first_force=False \
  --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=none \
  --config.wandb.mode=disabled

kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
trap - EXIT

[[ -f "${RUN}/checkpoints/5000.npz" ]] || exit 4
[[ -f "${RUN}/eval/statistics.json" ]] || exit 5
nvidia-smi > "${META}/nvidia_smi_end.txt"
