#!/bin/bash
#SBATCH --job-name=N2-ltp-eval
#SBATCH --account=def-ortner
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10603
#SBATCH --array=0-5%4
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err
set -euo pipefail
case "$SLURM_ARRAY_TASK_ID" in
0)V=A_beta02_control;CK=30500.npz;;1)V=B_beta01_fixed;CK=30500.npz;;2)V=C_beta_decay;CK=30500.npz;;3)V=D_residual_optimal;CK=30500.npz;;4)V=E_capped_near_tail;CK=30500.npz;;5)V=F_rank600_hard;CK=30500.npz;;*)exit 2;;esac
SRC=/scratch/dexuan1/runs/N2_rank400_longtime_pilot_E500/$V;RUN=/scratch/dexuan1/runs/N2_rank400_longtime_pilot_frozen/$V;META=${RUN}.metadata
[[ -f $SRC/checkpoints/$CK && ! -e $RUN && ! -e $META ]]||exit 2;mkdir -p "$META"
module --force purge;module load StdEnv/2023 gcc/12.3 python/3.11.5;source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_DISABLED=true WANDB_MODE=disabled XLA_PYTHON_CLIENT_PREALLOCATE=false EVAL_PRNG_TAG=$((2026072400+SLURM_ARRAY_TASK_ID))
cd /scratch/dexuan1/vmcnet;python -c "import jax;assert any(x.platform=='gpu' for x in jax.devices())" > "$META/jax_devices.txt"
printf 'source_checkpoint=%s\neval_prng_tag=%s\nnchains=4096\nnburn=5000\nnepochs=500\nnsteps_per_param_update=10\n' "$SRC/checkpoints/$CK" "$EVAL_PRNG_TAG" > "$META/evaluation_protocol.txt"
python experiments/fir_wssr4096_protocol/eval_seeded_launcher.py \
 --reload.logdir=$SRC --reload.checkpoint_relative_file_path=checkpoints/$CK --reload.use_config_file=True --reload.use_checkpoint_file=True --reload.new_optimizer_state=True --reload.reburn=False --reload.append=False --reload.same_logdir=False \
 --config.logdir=$RUN --config.base_logdir=$RUN --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE --config.vmc.nepochs=0 --config.vmc.disable_checkpointing=True --config.vmc.optimizer_type=adam \
 --config.eval.use_data_from_training=False --config.eval.nchains=4096 --config.eval.nburn=5000 --config.eval.nepochs=500 --config.eval.nsteps_per_param_update=10 --config.eval.nmoves_per_width_update=100 --config.eval.std_move=.25 --config.eval.record_local_energies=True --config.eval.nan_safe=False --config.wandb.mode=disabled
