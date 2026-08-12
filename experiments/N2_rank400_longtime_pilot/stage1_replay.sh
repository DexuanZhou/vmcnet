#!/bin/bash
#SBATCH --job-name=N2-ltp-replay
#SBATCH --account=def-ortner
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
set -euo pipefail
module --force purge;module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_DISABLED=true
cd /scratch/dexuan1/vmcnet
nvidia-smi;python -c "import jax; assert any(x.platform=='gpu' for x in jax.devices())"
python experiments/N2_rank400_longtime_pilot/stage1_replay.py
