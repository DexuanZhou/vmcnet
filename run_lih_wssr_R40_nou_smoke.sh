#!/bin/bash
#SBATCH --job-name=lih-wssr-R40-smoke
#SBATCH --account=def-ortner
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load python/3.11.5

source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate

export TMPDIR="${SLURM_TMPDIR:-/home/dexuan1/projects/rrg-ortner/dexuan1/tmp}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$TMPDIR"

cd /scratch/dexuan1/vmcnet

nvidia-smi

python - <<'PY'
import jax
print("JAX:", jax.__version__)
print("JAXLIB:", jax.lib.__version__)
print("devices:", jax.devices())
print("backend:", jax.default_backend())
PY

vmc-molecule \
  --reload.logdir=/home/dexuan1/projects/rrg-ortner/dexuan1/vmcnet_runs/lih_spring_gpu/spring500/vmc \
  --reload.checkpoint_relative_file_path=checkpoints/500.npz \
  --reload.use_config_file=True \
  --reload.new_optimizer_state=True \
  --reload.append=False \
  --reload.reburn=True \
  --config.problem.ion_pos="((0.0, 0.0, -1.5069621), (0.0, 0.0, 1.5069621))" \
  --config.problem.ion_charges="(1.0, 3.0)" \
  --config.problem.nelec="(2, 2)" \
  --config.distribute=False \
  --config.eval.nepochs=0 \
  --config.wandb.mode=disabled \
  --config.initial_seed=0 \
  --config.vmc.nchains=200 \
  --config.vmc.nburn=500 \
  --config.vmc.nsteps_per_param_update=1 \
  --config.vmc.nmoves_per_width_update=20 \
  --config.vmc.checkpoint_every=100 \
  --config.vmc.best_checkpoint_every=50 \
  --config.vmc.nepochs=200 \
  --config.vmc.optimizer_type=wssr_warm_svd_right \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.02 \
  --config.vmc.optimizer.wssr_warm_svd_right.damping=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=8 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=40 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=800 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=80 \
  --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=False \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name=NONE \
  --config.logdir=/scratch/dexuan1/runs/lih_wssr_nou_smoke

nvidia-smi
