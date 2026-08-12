#!/bin/bash
#SBATCH --job-name=N2-adcomp-eval
#SBATCH --account=def-ortner
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10405,fc10420,fc10512,fc10603
#SBATCH --array=0-11%4
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err
set -euo pipefail
TASK=$SLURM_ARRAY_TASK_ID; METHOD=$((TASK/3)); SEED=$((TASK%3))
case $METHOD in 0)VAR=A_rank400_hard;;1)VAR=B_rank400_beta01;;2)VAR=C_rank400_beta02;;3)VAR=D_rank800_hard;;*)exit 2;;esac
SOURCE=/scratch/dexuan1/runs/N2_adaptive_complement_E500/$VAR/seed$SEED
OUTPUT_ROOT=${FROZEN_OUTPUT_ROOT:-/scratch/dexuan1/runs/N2_adaptive_complement_frozen_retry1}
RUN=$OUTPUT_ROOT/$VAR/seed$SEED
META=${RUN}.metadata
[[ -f $SOURCE/checkpoints/500.npz && ! -e $RUN && ! -e $META ]] || exit 2
mkdir -p "$META"
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_DISABLED=true WANDB_MODE=disabled XLA_PYTHON_CLIENT_PREALLOCATE=false EVAL_PRNG_TAG=$((2026072200+TASK))
cd /scratch/dexuan1/vmcnet
python -c "import jax; d=jax.devices(); print(d); assert any(x.platform=='gpu' for x in d)" > "$META/jax_devices.txt"
printf 'source_checkpoint=%s\neval_prng_tag=%s\nnchains=4096\nnburn=10000\nnepochs=1000\nnsteps_per_param_update=10\n' "$SOURCE/checkpoints/500.npz" "$EVAL_PRNG_TAG" > "$META/evaluation_protocol.txt"
python experiments/fir_wssr4096_protocol/eval_seeded_launcher.py \
 --reload.logdir=$SOURCE --reload.checkpoint_relative_file_path=checkpoints/500.npz --reload.use_config_file=True --reload.use_checkpoint_file=True --reload.new_optimizer_state=True --reload.reburn=False --reload.append=False --reload.same_logdir=False \
 --config.logdir=$RUN --config.base_logdir=$RUN --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE --config.vmc.nepochs=0 --config.vmc.disable_checkpointing=True \
 --config.vmc.optimizer_type=adam \
 --config.eval.use_data_from_training=False --config.eval.nchains=4096 --config.eval.nburn=10000 --config.eval.nepochs=1000 --config.eval.nsteps_per_param_update=10 --config.eval.nmoves_per_width_update=100 --config.eval.std_move=.25 --config.eval.record_local_energies=True --config.eval.nan_safe=False --config.wandb.mode=disabled
