#!/bin/bash
#SBATCH --job-name=wssr-k1k-select
#SBATCH --account=def-ortner
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512
#SBATCH --array=0-2%2
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err
set -euo pipefail; : "${SYSTEM:?}"; LOCAL_TASK_ID=${LOCAL_TASK_ID:-${SLURM_ARRAY_TASK_ID}}
case ${SYSTEM}:${LOCAL_TASK_ID} in
C:0) NAME=rank400_eta08_lr02; RANK=400; ETA=.8; LR=.02; WARM=1;; C:1) NAME=rank800_eta08_lr02; RANK=800; ETA=.8; LR=.02; WARM=2;; C:2) NAME=rank800_eta02_lr0002; RANK=800; ETA=.2; LR=.002; WARM=2;;
N2:0) NAME=rank400_eta02_lr0002; RANK=400; ETA=.2; LR=.002; WARM=1;; N2:1) NAME=rank800_eta02_lr0002; RANK=800; ETA=.2; LR=.002; WARM=2;; N2:2) NAME=rank800_eta03_lr0002; RANK=800; ETA=.3; LR=.002; WARM=2;; *) exit 2;; esac
SOURCE=${SOURCE_OVERRIDE:-/scratch/dexuan1/runs/wssr_${SYSTEM}_4096_kfac1000}; ROOT=${ROOT_OVERRIDE:-/scratch/dexuan1/runs/wssr_${SYSTEM}_4096_kfac1000_selection}; RUN=${ROOT}/${NAME}; META=${ROOT}/metadata/${NAME}
[[ -f ${SOURCE}/checkpoints/${CHECKPOINT_NAME:-1000.npz} && ! -e ${RUN} ]]||exit 2
module --force purge; module load StdEnv/2023 gcc/12.3 python/3.11.5; source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true XLA_PYTHON_CLIENT_PREALLOCATE=false VMCNET_PROFILE_TIMING=1 SMOKE_TIMING_DIR=${META}
mkdir -p ${META}; cd /scratch/dexuan1/vmcnet; source experiments/fir_n4096_smoke/scripts/gpu_health_check.sh ${META}/nvidia_smi.txt
echo 'timestamp,memory_used_mib,utilization_gpu_percent' > ${META}/gpu_memory_poll.csv; (while true;do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits >>${META}/gpu_memory_poll.csv||true;sleep 1;done)& M=$!; trap 'kill $M 2>/dev/null||true' EXIT
set +e
python experiments/fir_n4096_smoke/scripts/timed_reload_launcher.py --reload.logdir=${SOURCE} --reload.checkpoint_relative_file_path=checkpoints/${CHECKPOINT_NAME:-1000.npz} --reload.use_config_file=True --reload.use_checkpoint_file=True --reload.new_optimizer_state=True --reload.reburn=False --reload.append=False --reload.same_logdir=False \
 --config.logdir=${RUN} --config.base_logdir=${RUN} --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE --config.vmc.nchains=4096 --config.eval.nchains=4096 --config.vmc.nepochs=2000 --config.vmc.nsteps_per_param_update=10 --config.vmc.check_for_nans=True --config.vmc.checkpoint_every=2000 --config.vmc.best_checkpoint_every=2000 --config.eval.nepochs=0 --config.wandb.mode=disabled \
 --config.vmc.optimizer_type=wssr_warm_svd_right --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=${LR} --config.vmc.optimizer.wssr_warm_svd_right.eta=${ETA} --config.vmc.optimizer.wssr_warm_svd_right.damping=.0003 --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0 --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=.001 --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=.0001 --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=${RANK} --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=${RANK} --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank=${RANK} --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=${RANK} --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=False --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=8 --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=${WARM} & P=$!
python experiments/fir_wssr4096_protocol/monitor_screen.py --pid $P --run ${RUN} --out ${META}/early_stop.json & Q=$!; wait $P; S=$?; wait $Q||true; exit $S
