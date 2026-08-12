#!/bin/bash
#SBATCH --job-name=H-vmc-e2e-audit
#SBATCH --account=def-ortner
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail
RUN=/scratch/dexuan1/runs/H_energy_bias_audit_kfac2000_eval20000
META=${RUN}_metadata
[[ ! -e "$RUN" && ! -e "$META" ]] || {
  echo "Refusing to overwrite $RUN or $META" >&2
  exit 2
}
mkdir -p "$META" /scratch/dexuan1/runs/logs

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_DISABLED=true WANDB_MODE=disabled XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /scratch/dexuan1/vmcnet

hostname > "$META/hostname.txt"
nvidia-smi > "$META/nvidia_smi.txt"
git rev-parse HEAD > "$META/git_hash.txt"
git status --short > "$META/git_status.txt"
cat > "$META/protocol.txt" <<EOF
purpose=H end-to-end raw-estimator audit
problem=H at origin, charges=(1,), nelec=(1,0)
training_optimizer=KFAC
training_nchains=1000
training_nburn=5000
training_nepochs=2000
eval_nchains=4096
eval_nburn=10000
eval_nepochs=20000
eval_nsteps_per_param_update=10
exact_energy=-0.5 Ha
EOF

python experiments/C_energy_bias_audit/run_h_end_to_end.py \
  --config.logdir="$RUN" \
  --config.base_logdir="$RUN" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name=NONE \
  --config.problem.ion_pos='((0.,0.,0.),)' \
  --config.problem.ion_charges='(1.,)' \
  --config.problem.nelec='(1,0)' \
  --config.vmc.optimizer_type=kfac \
  --config.vmc.optimizer.kfac.learning_rate=0.05 \
  --config.vmc.optimizer.kfac.schedule_type=inverse_time \
  --config.vmc.optimizer.kfac.learning_decay_rate=0.0001 \
  --config.vmc.nchains=1000 \
  --config.vmc.nburn=5000 \
  --config.vmc.nepochs=2000 \
  --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.clip_threshold=5.0 \
  --config.vmc.clip_center=mean \
  --config.vmc.check_for_nans=True \
  --config.vmc.checkpoint_every=2000 \
  --config.vmc.best_checkpoint_every=2000 \
  --config.eval.use_data_from_training=False \
  --config.eval.nchains=4096 \
  --config.eval.nburn=10000 \
  --config.eval.nepochs=20000 \
  --config.eval.nsteps_per_param_update=10 \
  --config.eval.nmoves_per_width_update=100 \
  --config.eval.std_move=0.25 \
  --config.eval.record_local_energies=True \
  --config.eval.nan_safe=False \
  --config.distribute=False \
  --config.wandb.mode=disabled
