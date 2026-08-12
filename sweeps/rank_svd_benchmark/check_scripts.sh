#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

bash -n "${HERE}"/*.sh
test -f /scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1/checkpoints/1000.npz
test -f /scratch/dexuan1/runs/kfac_pre/N2eq_R2068_kfac_pre5000/checkpoints/5000.npz
grep -q '#SBATCH --array=0-5%2' "${HERE}/C_rank_svd_array_E100000.sh"
grep -q '#SBATCH --array=0-5%2' "${HERE}/N2eq_R2068_rank_svd_array_E100000.sh"
grep -q 'COMBOS=(200:2 400:1 400:2 400:4 800:2 1600:2)' "${HERE}/C_rank_svd_array_E100000.sh"
grep -q 'COMBOS=(200:2 400:1 400:2 400:4 800:2 1600:2)' "${HERE}/N2eq_R2068_rank_svd_array_E100000.sh"
grep -q 'CHECKPOINT_EVERY=100000' "${HERE}/run_benchmark.sh"
grep -q 'reload.new_optimizer_state=True --reload.reburn=True' "${HERE}/run_benchmark.sh"
echo 'OK: syntax, checkpoints, matrix, fresh reload, concurrency cap, and final-only checkpointing'
echo "sbatch ${HERE}/C_rank_svd_array_E100000.sh"
echo "sbatch ${HERE}/N2eq_R2068_rank_svd_array_E100000.sh"
