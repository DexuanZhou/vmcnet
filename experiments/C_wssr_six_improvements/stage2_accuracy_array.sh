#!/bin/bash
#SBATCH --job-name=C-wssr6-s2acc
#SBATCH --account=def-ortner
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10405,fc10420,fc10512
#SBATCH --array=0-5%4
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err
set -euo pipefail
ROOT=/scratch/dexuan1/vmcnet/experiments/C_wssr_six_improvements;SEL=$ROOT/results/stage1/selection.json;MI=$SLURM_ARRAY_TASK_ID;M=$(python -c "import json;print(json.load(open('$ROOT/results/stage1/stage2_methods.json'))[$MI])");if [[ $M == baseline ]];then P=warm2;else P=$(python -c "import json;print(json.load(open('$SEL'))['$M']['parameter'])");fi;V=${M}_${P//+/_};RUN=/scratch/dexuan1/runs/C_wssr6_stage2_E200/$V;mkdir -p "$ROOT/results/stage2_accuracy"
module --force purge;module load StdEnv/2023 gcc/12.3 python/3.11.5;source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate;export XLA_PYTHON_CLIENT_PREALLOCATE=false;cd /scratch/dexuan1/vmcnet
for E in 50 100 200;do
 OUT=$ROOT/results/stage2_accuracy/${V}_e${E}.json
 [[ -f $RUN/checkpoints/$E.npz && ! -e $OUT ]]||exit 2
 python experiments/C_wssr_six_improvements/stage2_accuracy.py --run "$RUN" --epoch "$E" --out "$OUT"
done
