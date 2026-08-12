#!/bin/bash
#SBATCH --job-name=N2-adcomp-50k
#SBATCH --account=def-ortner
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10603
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err
set -euo pipefail
SOURCE=/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2/kfac_preliminary
case "$SLURM_ARRAY_TASK_ID" in
  0) VAR=A_rank800_hard; RANK=800; MODE=none; BETA=0; COMP=0;;
  1) VAR=B_rank400_beta02; RANK=400; MODE=adaptive_complement; BETA=.2; COMP=.0001;;
  *) exit 2;;
esac
RUN=/scratch/dexuan1/runs/N2_adcomp_E50000_compare/$VAR
META=${RUN}.metadata
[[ -f $SOURCE/checkpoints/5000.npz && ! -e $RUN && ! -e $META ]] || { echo "missing checkpoint or output exists" >&2; exit 2; }
mkdir -p "$META" /scratch/dexuan1/runs/logs
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export WSSR_STAGE3_SEED=0 SMOKE_TIMING_DIR=$META
cd /scratch/dexuan1/vmcnet
hostname > "$META/hostname.txt"
nvidia-smi > "$META/nvidia_smi.txt"
python -c "import jax; d=jax.devices(); print(d); assert any(x.platform=='gpu' for x in d)" > "$META/jax_devices.txt"
git rev-parse HEAD > "$META/git_hash.txt"; git status --short > "$META/git_status.txt"
printf 'variant=%s\nrank=%s\nexperimental_mode=%s\nadaptive_beta=%s\ncomplement_weight=%s\nsource_checkpoint=%s\n' "$VAR" "$RANK" "$MODE" "$BETA" "$COMP" "$SOURCE/checkpoints/5000.npz" > "$META/protocol.txt"
echo 'timestamp_unix,memory_used_mib,utilization_gpu_percent' > "$META/gpu_memory_poll.csv"
(while true; do T=$(date +%s.%N); nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -v t="$T" '{gsub(/ /,""); print t "," $1 "," $2}' >> "$META/gpu_memory_poll.csv" || true; sleep 1; done) & MON=$!
cleanup(){ kill "$MON" 2>/dev/null || true; wait "$MON" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
python experiments/C_wssr_six_improvements/seeded_training_launcher.py \
 --reload.logdir=$SOURCE --reload.checkpoint_relative_file_path=checkpoints/5000.npz --reload.use_config_file=True --reload.use_checkpoint_file=True --reload.new_optimizer_state=True --reload.reburn=True --reload.append=False --reload.same_logdir=False \
 --config.logdir=$RUN --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
 --config.vmc.nchains=4096 --config.vmc.nburn=5000 --config.vmc.nepochs=50000 --config.vmc.nsteps_per_param_update=10 --config.vmc.clip_center=mean --config.vmc.clip_threshold=5 --config.vmc.check_for_nans=True --config.vmc.checkpoint_every=10000 --config.vmc.best_checkpoint_every=10000 --config.eval.nepochs=0 --config.wandb.mode=disabled \
 --config.vmc.optimizer_type=wssr_warm_svd_right --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=.002 --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=.0001 --config.vmc.optimizer.wssr_warm_svd_right.eta=.2 --config.vmc.optimizer.wssr_warm_svd_right.damping=.0003 --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=$COMP --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=.001 \
 --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=$RANK --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=$RANK --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank=$RANK --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=$RANK --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=False --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=8 --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 --config.vmc.optimizer.wssr_warm_svd_right.exact_first=False --config.vmc.optimizer.wssr_warm_svd_right.update_diagnostics=True \
 --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=$MODE --config.vmc.optimizer.wssr_warm_svd_right.experimental_target_rank=$RANK --config.vmc.optimizer.wssr_warm_svd_right.cluster_gap_threshold=.002 --config.vmc.optimizer.wssr_warm_svd_right.near_tail_modes=0 --config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_beta=$BETA --config.vmc.optimizer.wssr_warm_svd_right.smooth_transition_start=-1 --config.vmc.optimizer.wssr_warm_svd_right.smooth_transition_end=-1 --config.vmc.optimizer.wssr_warm_svd_right.force_aware_krylov_vectors=0 --config.vmc.optimizer.wssr_warm_svd_right.iterative_complement_iterations=0
