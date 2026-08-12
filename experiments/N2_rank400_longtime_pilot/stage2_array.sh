#!/bin/bash
#SBATCH --job-name=N2-ltp-E500
#SBATCH --account=def-ortner
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10603
#SBATCH --array=0-5%4
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err
set -euo pipefail
SRC=/scratch/dexuan1/runs/N2_adcomp_E50000_compare/B_rank400_beta02
case "$SLURM_ARRAY_TASK_ID" in
0) V=A_beta02_control;R=400;W=400;MODE=adaptive_complement;B=.2;BF=.2;DS=0;EX=False;NEW=False;COMP=.0001;;
1) V=B_beta01_fixed;R=400;W=400;MODE=adaptive_complement;B=.1;BF=.1;DS=0;EX=False;NEW=False;COMP=.0001;;
2) V=C_beta_decay;R=400;W=400;MODE=adaptive_complement;B=.2;BF=.05;DS=500;EX=False;NEW=False;COMP=.0001;;
3) V=D_residual_optimal;R=400;W=400;MODE=residual_optimal_complement;B=.2;BF=.2;DS=0;EX=False;NEW=False;COMP=0;;
4) V=E_capped_near_tail;R=600;W=600;MODE=capped_near_tail;B=.2;BF=.2;DS=0;EX=True;NEW=False;COMP=0;;
5) V=F_rank600_hard;R=600;W=600;MODE=none;B=0;BF=0;DS=0;EX=True;NEW=False;COMP=0;;
*)exit 2;;esac
RUN=/scratch/dexuan1/runs/N2_rank400_longtime_pilot_E500/$V;META=${RUN}.metadata
[[ -f $SRC/checkpoints/30000.npz && ! -e $RUN && ! -e $META ]]||exit 2
mkdir -p "$META" /scratch/dexuan1/runs/logs
module --force purge;module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true XLA_PYTHON_CLIENT_PREALLOCATE=false PILOT_META=$META
NE=30500;CE=30500;export ADVANCE_RELOAD_EPOCH=1
if [[ $R == 600 ]];then export EXPAND_WSSR_RANK=600;fi
cd /scratch/dexuan1/vmcnet
hostname > "$META/hostname.txt";nvidia-smi > "$META/nvidia_smi.txt";python -c "import jax;assert any(x.platform=='gpu' for x in jax.devices())" > "$META/jax_devices.txt"
printf 'variant=%s\nsource_checkpoint=%s\nrank=%s\nworking_rank=%s\nmode=%s\nbeta=%s\nbeta_final=%s\ndecay_steps=%s\nexact_first=%s\nnew_optimizer_state=%s\noptax_schedule_start=30000\n' "$V" "$SRC/checkpoints/30000.npz" "$R" "$W" "$MODE" "$B" "$BF" "$DS" "$EX" "$NEW" > "$META/protocol.txt"
echo 'timestamp_unix,memory_used_mib' > "$META/gpu_memory_poll.csv";(while true;do T=$(date +%s.%N);nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk -v t="$T" '{print t","$1}' >> "$META/gpu_memory_poll.csv"||true;sleep 1;done)&MON=$!;trap 'kill $MON 2>/dev/null||true' EXIT
python experiments/N2_rank400_longtime_pilot/training_launcher.py \
 --reload.logdir=$SRC --reload.checkpoint_relative_file_path=checkpoints/30000.npz --reload.use_config_file=True --reload.use_checkpoint_file=True --reload.new_optimizer_state=$NEW --reload.reburn=False --reload.append=False --reload.same_logdir=False \
 --config.logdir=$RUN --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE --config.vmc.nchains=4096 --config.vmc.nepochs=$NE --config.vmc.nsteps_per_param_update=10 --config.vmc.check_for_nans=True --config.vmc.checkpoint_every=$CE --config.vmc.best_checkpoint_every=$CE --config.eval.nepochs=0 --config.wandb.mode=disabled \
 --config.vmc.optimizer_type=wssr_warm_svd_right --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=.002 --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=.0001 --config.vmc.optimizer.wssr_warm_svd_right.eta=.2 --config.vmc.optimizer.wssr_warm_svd_right.damping=.0003 --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=$COMP --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=.001 \
 --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=$R --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=$R --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank=$R --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=$W --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=False --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=8 --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 --config.vmc.optimizer.wssr_warm_svd_right.exact_first=$EX --config.vmc.optimizer.wssr_warm_svd_right.exact_first_force=$EX --config.vmc.optimizer.wssr_warm_svd_right.update_diagnostics=True \
 --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=$MODE --config.vmc.optimizer.wssr_warm_svd_right.experimental_target_rank=400 --config.vmc.optimizer.wssr_warm_svd_right.near_tail_modes=200 --config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_beta=$B --config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_beta_final=$BF --config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_decay_steps=$DS --config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_decay_start=30000 --config.vmc.optimizer.wssr_warm_svd_right.smooth_transition_start=-1 --config.vmc.optimizer.wssr_warm_svd_right.smooth_transition_end=-1 --config.vmc.optimizer.wssr_warm_svd_right.force_aware_krylov_vectors=0 --config.vmc.optimizer.wssr_warm_svd_right.iterative_complement_iterations=0
