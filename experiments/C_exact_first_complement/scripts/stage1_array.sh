#!/bin/bash
#SBATCH --job-name=C-exactfirst-10
#SBATCH --account=def-ortner
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err
set -euo pipefail

REPO=/scratch/dexuan1/vmcnet
PKG=${REPO}/experiments/C_exact_first_complement
SOURCE=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1
CHECKPOINT=${SOURCE}/checkpoints/1000.npz
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311
case "${SLURM_ARRAY_TASK_ID}" in
  0) VARIANT=A_warm_c0; EXACT=False; COMPLEMENT=0.0 ;;
  1) VARIANT=B_exact_c0; EXACT=True; COMPLEMENT=0.0 ;;
  2) VARIANT=C_exact_c1e4; EXACT=True; COMPLEMENT=0.0001 ;;
  3) VARIANT=D_exact_c1e3; EXACT=True; COMPLEMENT=0.001 ;;
  *) echo "Invalid task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac
RUN_DIR=/scratch/dexuan1/runs/C_exact_first_complement/stage1_10_retry1/${VARIANT}
META=${RUN_DIR}.metadata
[[ -f "${CHECKPOINT}" ]] || { echo "Missing ${CHECKPOINT}" >&2; exit 2; }
[[ ! -e "${RUN_DIR}" && ! -e "${META}" ]] || { echo "Refusing overwrite ${RUN_DIR}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd "${REPO}"
hostname > "${META}/hostname.txt"
nvidia-smi > "${META}/nvidia_smi_start.txt"
python -c 'import jax; assert any(d.platform == "gpu" for d in jax.devices()), jax.devices(); print(jax.devices())' > "${META}/jax_devices.txt"
git rev-parse HEAD > "${META}/git_commit.txt"
git diff -- vmcnet/updates/wssr.py vmcnet/train/default_config.py tests/units/updates/test_wssr.py > "${META}/local_exact_first.patch"
echo 'timestamp_unix,memory_used_mib,utilization_gpu_percent' > "${META}/gpu_memory_poll.csv"
( while true; do TS=$(date +%s.%N); nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -v t="${TS}" -F', ' '{print t "," $1 "," $2}' >> "${META}/gpu_memory_poll.csv" || true; sleep 1; done ) &
MON=$!
cleanup(){ kill "${MON}" 2>/dev/null || true; wait "${MON}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

START=$(date +%s)
set +e
vmc-molecule \
  --reload.logdir="${SOURCE}" --reload.checkpoint_relative_file_path=checkpoints/1000.npz \
  --reload.use_config_file=True --reload.use_checkpoint_file=True \
  --reload.new_optimizer_state=True --reload.reburn=False \
  --reload.append=False --reload.same_logdir=False \
  --config.logdir="${RUN_DIR}" --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.problem.nelec='(4,2)' \
  --config.vmc.nchains=1000 --config.vmc.nburn=0 --config.vmc.nepochs=10 --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.clip_center=mean --config.vmc.clip_threshold=5.0 --config.vmc.check_for_nans=True \
  --config.vmc.checkpoint_every=10 --config.vmc.best_checkpoint_every=10 \
  --config.eval.nepochs=0 --config.wandb.mode=disabled \
  --config.vmc.optimizer_type=wssr_warm_svd_right \
  --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.02 \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta=0.8 \
  --config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_weight="${COMPLEMENT}" \
  --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True \
  --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=400 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=400 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank=400 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=400 \
  --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=True \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=8 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=1 \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first="${EXACT}" \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_reference_diagnostics=True
STATUS=$?
set -e
END=$(date +%s)
cleanup; trap - EXIT INT TERM
printf 'start_unix,end_unix,elapsed_seconds,exit_status\n%s,%s,%s,%s\n' "${START}" "${END}" "$((END-START))" "${STATUS}" > "${META}/timing.csv"
exit "${STATUS}"
