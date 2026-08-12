#!/bin/bash
#SBATCH --job-name=N2-R2016-SPRING-lr001-r1
#SBATCH --account=def-ortner
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407,fc10405
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

export ROOT_OVERRIDE=/scratch/dexuan1/runs/N2_R2016_spring_lr001_mu099_after_kfac5000_E100000_retry1

exec bash \
  /scratch/dexuan1/vmcnet/experiments/N2_R2016_spring_lr001_control/run.sh
