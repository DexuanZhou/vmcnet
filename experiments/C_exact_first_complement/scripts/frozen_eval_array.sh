#!/bin/bash
#SBATCH --job-name=C-exactfirst-eval
#SBATCH --account=def-ortner
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512
#SBATCH --array=0-2%3
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err
set -euo pipefail
VARIANTS=(A_warm_c0 B_exact_c0 C_exact_c1e4 D_exact_c1e3)
V=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
SOURCE=/scratch/dexuan1/runs/C_exact_first_complement/stage2_500_retry1/${V}
RUN=/scratch/dexuan1/runs/C_exact_first_complement/frozen_eval_retry1/${V}
[[ -f ${SOURCE}/checkpoints/500.npz ]]||exit 2
[[ ! -e ${RUN} ]]||exit 2
module --force purge;module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true XLA_PYTHON_CLIENT_PREALLOCATE=false EVAL_PRNG_TAG=2026072102
cd /scratch/dexuan1/vmcnet
python experiments/fir_wssr4096_protocol/eval_seeded_launcher.py \
 --reload.logdir="${SOURCE}" --reload.checkpoint_relative_file_path=checkpoints/500.npz --reload.use_config_file=True --reload.use_checkpoint_file=True --reload.new_optimizer_state=True --reload.reburn=False --reload.append=False --reload.same_logdir=False \
 --config.logdir="${RUN}" --config.base_logdir="${RUN}" --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
 --config.vmc.nepochs=0 --config.vmc.disable_checkpointing=True \
 --config.eval.use_data_from_training=False --config.eval.nchains=4096 --config.eval.nburn=10000 --config.eval.nepochs=5000 --config.eval.nsteps_per_param_update=10 --config.eval.nmoves_per_width_update=100 --config.eval.std_move=0.25 --config.eval.record_local_energies=True --config.eval.nan_safe=False --config.wandb.mode=disabled
