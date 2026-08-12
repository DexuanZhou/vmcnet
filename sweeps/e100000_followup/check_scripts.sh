#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
C_PRE=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1/checkpoints/1000.npz

bash -n "${HERE}"/*.sh
[[ -f "${C_PRE}" ]] || { echo "FAIL: missing C checkpoint ${C_PRE}" >&2; exit 1; }
echo "OK: C checkpoint ${C_PRE}"

for script in "${HERE}"/C_*fresh_E100000.sh; do
  grep -q 'SOURCE_EPOCH=1000 RESUME_OPTIMIZER=False' "${script}"
  grep -q 'SOURCE_DIR=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1' "${script}"
  grep -q "NELEC='(4,2)'" "${script}"
done
echo 'OK: all C wrappers are fresh from KFAC-1000 with nelec=(4,2)'

grep -q 'CHECKPOINT_EVERY=100000' "${HERE}/run_spring_100k.sh"
grep -q 'CHECKPOINT_EVERY=100000' "${HERE}/run_wssr_100k.sh"
echo 'OK: E100000 saves only final 100000'

if grep -R -n -E '^[[:space:]]*sbatch[[:space:]]' "${HERE}" --include='*.sh'; then
  echo 'FAIL: executable sbatch command found' >&2; exit 1
fi
echo 'OK: no script executes sbatch'

echo
echo 'Exact commands (PRINTED ONLY; NOT EXECUTED):'
echo "sbatch ${HERE}/C_spring_lr02_mu099_nc001_fresh_E100000.sh"
echo "sbatch ${HERE}/C_wssr_eta08_lr02_tikhonov_svd8_5_fresh_E100000.sh"
echo "sbatch ${HERE}/C_wssr_eta02_lr0002_tikhonov_svd8_2_fresh_E100000.sh  # optional"
echo "sbatch ${HERE}/C_wssr_eta08_lr02_tikhonov_svd8_2_fresh_E100000.sh  # optional"
