#!/bin/bash
set -euo pipefail
cd /scratch/dexuan1/vmcnet/experiments/fir_n4096_smoke
[[ ! -e /scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2/spring_mu_stability ]] || exit 2
A=$(sbatch --parsable scripts/N2_spring_mu_stability_array.sh)
C=$(sbatch --parsable --dependency="afterany:${A}" scripts/N2_collect_spring_mu.sh)
printf 'ARRAY_JOB_ID=%s\nCOLLECTOR_JOB_ID=%s\n' "$A" "$C" | tee n2_spring_mu_job_ids.txt
