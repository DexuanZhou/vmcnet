#!/bin/bash
#SBATCH --job-name=C-wssr-frozen-eval
#SBATCH --account=def-ortner
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err
set -euo pipefail
case ${SLURM_ARRAY_TASK_ID} in
  0) NAME=C_rank800_eta02_lr0002; EVAL_PRNG_TAG=2026071901;;
  1) NAME=C_rank800_eta08_lr02; EVAL_PRNG_TAG=2026071902;;
  *) exit 2;;
esac
export EVAL_PRNG_TAG
SOURCE=/scratch/dexuan1/runs/wssr4096_top2/C/${NAME}
RUN=/scratch/dexuan1/runs/wssr4096_frozen_eval/C/${NAME}
META=/scratch/dexuan1/runs/wssr4096_frozen_eval/C/metadata/${NAME}
[[ -f ${SOURCE}/checkpoints/2000.npz ]] || { echo missing checkpoint >&2; exit 2; }
[[ ! -e ${RUN} ]] || { echo refusing overwrite >&2; exit 2; }
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p ${META} /scratch/dexuan1/runs/logs
cd /scratch/dexuan1/vmcnet
hostname > ${META}/hostname.txt
source experiments/fir_n4096_smoke/scripts/gpu_health_check.sh ${META}/nvidia_smi_start.txt
git rev-parse HEAD > ${META}/git_commit.txt
printf 'source_checkpoint=%s\neval_prng_tag=%s\n' "${SOURCE}/checkpoints/2000.npz" "${EVAL_PRNG_TAG}" > ${META}/protocol.txt
python experiments/fir_wssr4096_protocol/eval_seeded_launcher.py \
 --reload.logdir=${SOURCE} --reload.checkpoint_relative_file_path=checkpoints/2000.npz \
 --reload.use_config_file=True --reload.use_checkpoint_file=True --reload.new_optimizer_state=True \
 --reload.reburn=False --reload.append=False --reload.same_logdir=False \
 --config.logdir=${RUN} --config.base_logdir=${RUN} --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
 --config.vmc.nepochs=0 --config.vmc.disable_checkpointing=True --config.eval.use_data_from_training=False \
 --config.eval.nchains=4096 --config.eval.nburn=5000 --config.eval.nepochs=1000 \
 --config.eval.nsteps_per_param_update=10 --config.eval.nmoves_per_width_update=100 \
 --config.eval.std_move=0.25 --config.eval.record_local_energies=True --config.eval.nan_safe=False \
 --config.wandb.mode=disabled
