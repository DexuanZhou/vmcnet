#!/bin/bash
#SBATCH --job-name=N2-env-E5k
#SBATCH --account=def-ortner
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --array=0-1%2
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err

export ROOT=/scratch/dexuan1/runs/N2_cluster_envelope_E5000_20260809
export NEPOCHS=5000
export CHECKPOINT_EVERY=500
source /scratch/dexuan1/vmcnet/experiments/N2_cluster_envelope_screen_20260809/run_array.sh
