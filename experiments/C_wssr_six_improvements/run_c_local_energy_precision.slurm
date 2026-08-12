#!/bin/bash
#SBATCH --job-name=C-EL-fp64-audit
#SBATCH --account=def-ortner
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10407,fc10420,fc10512
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
set -euo pipefail
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 scipy-stack/2024b
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_DISABLED=true WANDB_MODE=disabled XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /scratch/dexuan1/vmcnet
nvidia-smi
python - <<'PY'
import jax
print("JAX devices:", jax.devices())
assert any(d.platform == "gpu" for d in jax.devices())
PY
rm -f \
  experiments/C_wssr_six_improvements/results/energy_bias_audit/p2_c_fp64/phase1_f32.npz \
  experiments/C_wssr_six_improvements/results/energy_bias_audit/p2_c_fp64/phase2_f64.npz
JAX_ENABLE_X64=false python experiments/C_wssr_six_improvements/c_el_precision_phase1_f32.py
JAX_ENABLE_X64=true python experiments/C_wssr_six_improvements/c_el_precision_phase2_f64.py
module load scipy-stack
python experiments/C_wssr_six_improvements/c_el_precision_phase3_analyze.py
