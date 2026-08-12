#!/bin/bash
#SBATCH --job-name=C-spring-eval100k
#SBATCH --account=def-ortner
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
set -euo pipefail
SOURCE=/scratch/dexuan1/runs/C_spring_mu099_after_kfac5000_E50000
RUN=/scratch/dexuan1/runs/C_spring_mu099_epoch30000_frozen_eval_burn100k_n8192
META=${RUN}_metadata
CHECKPOINT=${SOURCE}/checkpoints/30000.npz
[[ -f ${CHECKPOINT} ]] || { echo "missing ${CHECKPOINT}" >&2; exit 2; }
[[ ! -e ${RUN} && ! -e ${META} ]] || { echo "refusing overwrite" >&2; exit 2; }
module --force purge;module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true XLA_PYTHON_CLIENT_PREALLOCATE=false
# Same independent stream tag as the matched WSSR frozen evaluation.
export EVAL_PRNG_TAG=2026072101
mkdir -p ${META} /scratch/dexuan1/runs/logs;cd /scratch/dexuan1/vmcnet
hostname >${META}/hostname.txt
source experiments/fir_n4096_smoke/scripts/gpu_health_check.sh ${META}/nvidia_smi.txt
git rev-parse HEAD >${META}/git_commit.txt
printf 'mode=frozen_independent_inference\noptimizer=spring\nmu=0.99\nsource_checkpoint=%s\ncheckpoint_training_epoch=30000\ntraining_updates=0\neval_nchains=8192\neval_nburn=100000\neval_nepochs=20000\neval_steps_between_samples=10\nraw_local_energy_samples=163840000\neval_prng_tag=%s\n' "${CHECKPOINT}" "${EVAL_PRNG_TAG}" >${META}/protocol.txt
echo 'timestamp,memory_used_mib,utilization_gpu_percent' >${META}/gpu_memory_poll.csv
(while true;do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits >>${META}/gpu_memory_poll.csv||true;sleep 1;done)& M=$!
cleanup(){ kill $M 2>/dev/null||true; };trap cleanup EXIT INT TERM
python experiments/fir_wssr4096_protocol/eval_seeded_launcher.py \
 --reload.logdir=${SOURCE} --reload.checkpoint_relative_file_path=checkpoints/30000.npz \
 --reload.use_config_file=True --reload.use_checkpoint_file=True --reload.new_optimizer_state=True \
 --reload.reburn=False --reload.append=False --reload.same_logdir=False \
 --config.logdir=${RUN} --config.base_logdir=${RUN} --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
 --config.vmc.nepochs=0 --config.vmc.disable_checkpointing=True \
 --config.eval.use_data_from_training=False --config.eval.nchains=8192 --config.eval.nburn=100000 \
 --config.eval.nepochs=20000 --config.eval.nsteps_per_param_update=10 \
 --config.eval.nmoves_per_width_update=100 --config.eval.std_move=0.25 \
 --config.eval.record_local_energies=True --config.eval.nan_safe=False --config.wandb.mode=disabled
