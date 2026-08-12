#!/bin/bash
#SBATCH --job-name=N2-wssr-r1600-mem
#SBATCH --account=def-ortner
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

REPO=/scratch/dexuan1/vmcnet
SOURCE=/scratch/dexuan1/runs/old/N2_R2016/e100000_followup/N2eq/N2eq_kfac_pre5000
CHECKPOINT=${SOURCE}/checkpoints/5000.npz
RUN=/scratch/dexuan1/runs/N2_R2016_WSSR_r1600_semimf_mem100_20260807
META=${RUN}_metadata
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

[[ -f "${CHECKPOINT}" ]] || { echo "missing ${CHECKPOINT}" >&2; exit 2; }
[[ ! -e "${RUN}" && ! -e "${META}" ]] || {
  echo "refusing to overwrite ${RUN} or ${META}" >&2
  exit 2
}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export PYTHONPATH="${REPO}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd "${REPO}"

# The H100 gate begins with the same regression tests used by the CPU job.
pytest -q tests/units/updates/test_wssr.py \
  -k 'explicit_current_block or default_config_contains_wssr_warm_svd_right or no_persistent_u' \
  | tee "${META}/unit_tests.txt"

hostname > "${META}/hostname.txt"
nvidia-smi > "${META}/nvidia_smi_start.txt"
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"

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
purpose=rank1600 semi-matrix-free augmented-factor memory gate
system=N2
bond_length_bohr=2.016
source_checkpoint=${CHECKPOINT}
start_epoch=5000
training_steps=100
walkers=1000
mcmc_steps=10
reburn=False
optimizer=wssr_warm_svd_right
rank=1600
store_warm_u=False
semi_matrix_free_augmented=True
solution_recurrence_mode=residual
residual_evaluation=full_current_batch
solution_mu=0.95
subspace_eta_S=0.2
eta_S=0.0
eta_g=0.0
learning_rate=0.002
tikhonov_lambda=0.001
relative_singular_value_cutoff=0.0003
ssi=40/2
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
  --config.vmc.nepochs=100 \
  --config.vmc.checkpoint_every=1000 \
  --config.vmc.best_checkpoint_every=1000 \
  --config.eval.nepochs=0 \
  --config.vmc.optimizer_type=wssr_warm_svd_right \
  --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.002 \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_S=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_g=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.subspace_eta_S=0.2 \
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
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=1600 \
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

nvidia-smi > "${META}/nvidia_smi_end.txt"
