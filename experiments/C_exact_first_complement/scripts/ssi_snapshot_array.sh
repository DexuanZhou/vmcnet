#!/bin/bash
#SBATCH --job-name=C-ssi-snapshot
#SBATCH --account=def-ortner
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10420,fc10512
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err
set -euo pipefail
SNAPS=(initial A500 B500 C500); SNAP=${SNAPS[$SLURM_ARRAY_TASK_ID]}
OUT=/scratch/dexuan1/vmcnet/experiments/C_exact_first_complement/results/ssi_convergence/${SNAP}
[[ ! -e ${OUT} ]] || { echo "Refusing overwrite ${OUT}" >&2; exit 2; }
mkdir -p "${OUT}" /scratch/dexuan1/runs/logs
module --force purge; module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /scratch/dexuan1/vmcnet
hostname > "${OUT}/hostname.txt"; nvidia-smi > "${OUT}/nvidia_smi.txt"
echo 'timestamp_unix,memory_used_mib' > "${OUT}/gpu_memory_poll.csv"
(while true; do TS=$(date +%s.%N); nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk -v t="$TS" '{print t "," $1}' >> "${OUT}/gpu_memory_poll.csv" || true; sleep 1; done) & MON=$!
cleanup(){ kill "$MON" 2>/dev/null || true; wait "$MON" 2>/dev/null || true; }; trap cleanup EXIT INT TERM
python experiments/C_exact_first_complement/ssi_snapshot_replay.py "$SNAP" --output "$OUT"
