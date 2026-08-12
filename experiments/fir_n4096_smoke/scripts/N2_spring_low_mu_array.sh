#!/bin/bash
#SBATCH --job-name=N2-spring-lowmu
#SBATCH --account=def-ortner
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512
#SBATCH --array=0-3%2
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err
set -euo pipefail
MUS=(0.0 0.5 0.7 0.8); LABELS=(mu000 mu050 mu070 mu080)
MU=${MUS[${SLURM_ARRAY_TASK_ID}]}; LABEL=${LABELS[${SLURM_ARRAY_TASK_ID}]}
REPO=/scratch/dexuan1/vmcnet; PKG=${REPO}/experiments/fir_n4096_smoke
ROOT=/scratch/dexuan1/runs/fir_n4096_spring_low_mu/N2
SOURCE=/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2/kfac_preliminary
NAME=N2eq_R2068_spring_lr0005_${LABEL}_n4096_e200_diag
RUN=${ROOT}/${NAME}; META=${ROOT}/metadata/${NAME}
[[ -f "${SOURCE}/checkpoints/5000.npz" ]] || { echo 'Missing source checkpoint' >&2; exit 2; }
[[ ! -e "${RUN}" && ! -e "${META}" ]] || { echo 'Refusing overwrite' >&2; exit 2; }
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false VMCNET_DISABLE_CHECKPOINTS=1 VMCNET_PROFILE_TIMING=1
export SMOKE_TIMING_DIR="${META}"
mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd "${REPO}"
hostname > "${META}/hostname.txt"
source "${PKG}/scripts/gpu_health_check.sh" "${META}/nvidia_smi_start.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits > "${META}/gpu_identity.csv"
git rev-parse HEAD > "${META}/git_commit.txt"
git diff -- vmcnet/updates/spring.py vmcnet/train/vmc.py vmcnet/train/parse_config_flags.py vmcnet/train/default_config.py > "${META}/diagnostic_code.diff"
printf '%s\n' "${SOURCE}/checkpoints/5000.npz" > "${META}/source_checkpoint.txt"
set +e
python "${PKG}/scripts/timed_reload_launcher.py" \
 --reload.logdir="${SOURCE}" --reload.checkpoint_relative_file_path=checkpoints/5000.npz --reload.use_config_file=True --reload.use_checkpoint_file=True --reload.new_optimizer_state=True --reload.reburn=False --reload.append=False --reload.same_logdir=False \
 --config.logdir="${RUN}" --config.base_logdir="${RUN}" --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
 --config.vmc.nchains=4096 --config.vmc.nburn=5000 --config.vmc.nepochs=200 --config.vmc.nsteps_per_param_update=10 --config.vmc.clip_center=mean --config.vmc.clip_threshold=5.0 --config.vmc.check_for_nans=True --config.vmc.disable_checkpointing=True \
 --config.eval.nchains=4096 --config.eval.nepochs=0 --config.eval.nburn=0 --config.wandb.mode=disabled \
 --config.vmc.optimizer_type=spring --config.vmc.optimizer.spring.learning_rate=0.0005 --config.vmc.optimizer.spring.mu="${MU}" --config.vmc.optimizer.spring.damping=0.001 --config.vmc.optimizer.spring.constrain_norm=True --config.vmc.optimizer.spring.norm_constraint=0.001 --config.vmc.optimizer.spring.schedule_type=inverse_time --config.vmc.optimizer.spring.learning_decay_rate=0.0001 \
 --config.vmc.optimizer.spring.diagnostics=True --config.vmc.optimizer.spring.diagnostics_spectral=False --config.vmc.optimizer.spring.diagnostics_replay_epochs='()' --config.vmc.optimizer.spring.diagnostics_replay_dir='' &
PY_PID=$!
python "${PKG}/monitor_spring_thresholds.py" --pid "${PY_PID}" --run "${RUN}" --output "${META}/threshold_stop.json" &
MON_PID=$!
wait "${PY_PID}"; STATUS=$?
wait "${MON_PID}" 2>/dev/null || true
set -e
exit "${STATUS}"
