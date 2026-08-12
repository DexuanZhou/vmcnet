#!/bin/bash
#SBATCH --job-name=wssr50k-gate
#SBATCH --account=def-ortner
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err
set -euo pipefail
case ${SLURM_ARRAY_TASK_ID} in 0) SYSTEM=C;; 1) SYSTEM=N2;; *) exit 2;; esac
SOURCE=/scratch/dexuan1/runs/wssr_${SYSTEM}_4096_kfac1000
RUN=/scratch/dexuan1/runs/wssr_${SYSTEM}_4096_kfac1000_E50000_grid_gate/rank1600_warm2
META=${RUN}_metadata
[[ -f ${SOURCE}/validation.json && -f ${SOURCE}/checkpoints/1000.npz && ! -e ${RUN} ]] || exit 2
module --force purge; module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true XLA_PYTHON_CLIENT_PREALLOCATE=false VMCNET_DISABLE_CHECKPOINTS=1 VMCNET_PROFILE_TIMING=1 SMOKE_TIMING_DIR=${META}
mkdir -p ${META}; cd /scratch/dexuan1/vmcnet
source experiments/fir_n4096_smoke/scripts/gpu_health_check.sh ${META}/nvidia_smi.txt
echo 'timestamp,memory_used_mib,utilization_gpu_percent' >${META}/gpu_memory_poll.csv
(while true;do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits >>${META}/gpu_memory_poll.csv||true;sleep 1;done)& M=$!;trap 'kill $M 2>/dev/null||true' EXIT
python experiments/fir_n4096_smoke/scripts/timed_reload_launcher.py --reload.logdir=${SOURCE} --reload.checkpoint_relative_file_path=checkpoints/1000.npz --reload.use_config_file=True --reload.use_checkpoint_file=True --reload.new_optimizer_state=True --reload.reburn=False --reload.append=False --reload.same_logdir=False \
 --config.logdir=${RUN} --config.base_logdir=${RUN} --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE --config.vmc.nchains=4096 --config.eval.nchains=4096 --config.vmc.nepochs=5 --config.vmc.nsteps_per_param_update=10 --config.vmc.check_for_nans=True --config.vmc.disable_checkpointing=True --config.eval.nepochs=0 --config.wandb.mode=disabled \
 --config.vmc.optimizer_type=wssr_warm_svd_right --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=.002 --config.vmc.optimizer.wssr_warm_svd_right.eta=.8 --config.vmc.optimizer.wssr_warm_svd_right.damping=.0003 --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0 --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=.001 --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=.0001 --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=1600 --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=1600 --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank=1600 --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=1600 --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=False --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=8 --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2
