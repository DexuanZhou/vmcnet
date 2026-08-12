#!/bin/bash
#SBATCH --job-name=wssr4096-E100k
#SBATCH --account=def-ortner
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
set -euo pipefail; : "${SYSTEM:?}"; : "${RANK:?}"; : "${ETA:?}"; : "${LR:?}"; : "${WARM:?}"
SOURCE=/scratch/dexuan1/runs/wssr_${SYSTEM}_4096_kfac1000; RUN=/scratch/dexuan1/runs/wssr_${SYSTEM}_4096_kfac1000_E100000; META=${RUN}_metadata
[[ -f ${SOURCE}/validation.json && ! -e ${RUN} ]]||exit 2
module --force purge; module load StdEnv/2023 gcc/12.3 python/3.11.5; source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true XLA_PYTHON_CLIENT_PREALLOCATE=false VMCNET_PROFILE_TIMING=1 SMOKE_TIMING_DIR=${META}
mkdir -p ${META}; cd /scratch/dexuan1/vmcnet; source experiments/fir_n4096_smoke/scripts/gpu_health_check.sh ${META}/nvidia_smi.txt
git rev-parse HEAD >${META}/git_commit.txt; printf 'SYSTEM=%s\nRANK=%s\nETA=%s\nLR=%s\nWARM=%s\nSOURCE=%s\n' "$SYSTEM" "$RANK" "$ETA" "$LR" "$WARM" "$SOURCE/checkpoints/1000.npz" >${META}/selected_config.txt
echo 'timestamp,memory_used_mib,utilization_gpu_percent' >${META}/gpu_memory_poll.csv;(while true;do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits >>${META}/gpu_memory_poll.csv||true;sleep 1;done)&M=$!;trap 'kill $M 2>/dev/null||true' EXIT
python experiments/fir_n4096_smoke/scripts/timed_reload_launcher.py --reload.logdir=${SOURCE} --reload.checkpoint_relative_file_path=checkpoints/1000.npz --reload.use_config_file=True --reload.use_checkpoint_file=True --reload.new_optimizer_state=True --reload.reburn=False --reload.append=False --reload.same_logdir=False \
 --config.logdir=${RUN} --config.base_logdir=${RUN} --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE --config.vmc.nchains=4096 --config.eval.nchains=4096 --config.vmc.nepochs=100000 --config.vmc.nsteps_per_param_update=10 --config.vmc.check_for_nans=True --config.vmc.checkpoint_every=5000 --config.vmc.best_checkpoint_every=5000 --config.eval.nepochs=0 --config.wandb.mode=disabled \
 --config.vmc.optimizer_type=wssr_warm_svd_right --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=${LR} --config.vmc.optimizer.wssr_warm_svd_right.eta=${ETA} --config.vmc.optimizer.wssr_warm_svd_right.damping=.0003 --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0 --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=.001 --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=.0001 --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=${RANK} --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=${RANK} --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank=${RANK} --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=${RANK} --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=False --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=8 --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=${WARM}
