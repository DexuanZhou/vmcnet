#!/bin/bash
#SBATCH --job-name=N2-WSSR-r1600-100k
#SBATCH --account=def-ortner
#SBATCH --time=11:55:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_N2_R2016_WSSR_r1600_20260803
REPO=/scratch/dexuan1/vmcnet
RANK=${WSSR_RANK:-1600}
if [[ "${RANK}" == "1600" ]]; then
  SOURCE=/scratch/dexuan1/runs/N2_R2016_WSSR_rank1600_matched_E100000_20260803_stage1_retry1
  RUN=/scratch/dexuan1/runs/N2_R2016_WSSR_rank1600_matched_E100000_20260803
else
  SOURCE=/scratch/dexuan1/runs/N2_R2016_WSSR_rank${RANK}_matched_E100000_20260804_stage1
  RUN=/scratch/dexuan1/runs/N2_R2016_WSSR_rank${RANK}_matched_E100000_20260804
fi
CHECKPOINT=${SOURCE}/checkpoints/50000.npz
META=${RUN}_metadata
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

[[ -f "${CHECKPOINT}" ]] || { echo "missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${SNAPSHOT}/SOURCE_SNAPSHOT.txt" ]] || { echo "missing source snapshot" >&2; exit 2; }
[[ ! -e "${RUN}" && ! -e "${META}" ]] || {
  echo "refusing to overwrite ${RUN} or ${META}" >&2
  exit 2
}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export PYTHONPATH="${SNAPSHOT}"
# See stage1_train.sh: use the source snapshot for execution while making the
# originating repository metadata available to VMCNet's revision logger.
export GIT_DIR="${REPO}/.git"
export GIT_WORK_TREE="${SNAPSHOT}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd "${SNAPSHOT}"

cp SOURCE_SNAPSHOT.txt "${META}/"
hostname > "${META}/hostname.txt"
nvidia-smi > "${META}/nvidia_smi_start.txt"
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"

cat > "${META}/protocol.txt" <<EOF
purpose=matched N2 rank1600 WSSR stage 2 and frozen evaluation
source_run=${SOURCE}
source_checkpoint=${CHECKPOINT}
source_checkpoint_internal_epoch=49999
resume_start_epoch=50000
target_total_epochs=100000
optimizer_state_restored=True
warm_state_restored=True
walkers_amplitudes_prng_restored=True
reload_new_optimizer_state=False
reload_reburn=False
reload_append=True
rank=${RANK}
eta_S=0.2
eta_g=0.0
learning_rate=0.002
tikhonov_lambda=0.001
evaluation_nchains=2000
evaluation_burn_in=10000
evaluation_epochs=20000
evaluation_mcmc_steps_between_measurements=10
EOF

vmc-molecule \
  --reload.logdir="${SOURCE}" \
  --reload.use_config_file=True \
  --reload.config_relative_file_path=config.json \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/50000.npz \
  --reload.new_optimizer_state=False \
  --reload.reburn=False \
  --reload.append=True \
  --reload.same_logdir=False \
  --config.logdir="${RUN}" \
  --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name=NONE \
  --config.vmc.nepochs=100000 \
  --config.vmc.checkpoint_every=5000 \
  --config.vmc.best_checkpoint_every=5000 \
  --config.eval.use_data_from_training=False \
  --config.eval.nchains=2000 \
  --config.eval.nburn=10000 \
  --config.eval.nepochs=20000 \
  --config.eval.nsteps_per_param_update=10 \
  --config.eval.record_local_energies=True \
  --config.wandb.mode=disabled

[[ -f "${RUN}/checkpoints/100000.npz" ]] || {
  echo "missing epoch-100000 checkpoint" >&2
  exit 4
}
[[ -f "${RUN}/eval/statistics.json" ]] || {
  echo "missing frozen statistics" >&2
  exit 5
}
[[ -f "${RUN}/eval/local_energies.txt" ]] || {
  echo "missing frozen local energies" >&2
  exit 5
}
