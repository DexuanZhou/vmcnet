#!/bin/bash
#SBATCH --job-name=C-wssr6-stage1
#SBATCH --account=def-ortner
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10405,fc10420,fc10512
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err
set -euo pipefail
S=(initial A500 B500 C500);N=${S[$SLURM_ARRAY_TASK_ID]};O=/scratch/dexuan1/vmcnet/experiments/C_wssr_six_improvements/results/stage1/$N
[[ ! -e $O ]]||exit 2;mkdir -p "$O" /scratch/dexuan1/runs/logs
module --force purge;module load StdEnv/2023 gcc/12.3 python/3.11.5;source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_DISABLED=true
cd /scratch/dexuan1/vmcnet;hostname > "$O/hostname.txt";nvidia-smi > "$O/nvidia_smi.txt"
echo 'timestamp_unix,memory_used_mib' > "$O/gpu_memory_poll.csv"
(while true;do T=$(date +%s.%N);nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk -v t="$T" '{print t "," $1}' >> "$O/gpu_memory_poll.csv"||true;sleep 1;done)& MON=$!
cleanup(){ kill "$MON" 2>/dev/null||true;wait "$MON" 2>/dev/null||true;};trap cleanup EXIT INT TERM
python experiments/C_wssr_six_improvements/stage1_replay.py "$N" --out "$O"
