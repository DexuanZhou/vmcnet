#!/bin/bash
set -euo pipefail
cd /scratch/dexuan1/vmcnet/experiments/fir_n4096_smoke
RUN=/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2/spring_mu_stability/N2eq_R2068_spring_lr0005_minsr_mu0_n4096_e200
[[ ! -e "${RUN}" ]] || { echo 'Refusing overwrite' >&2; exit 2; }
J=$(sbatch --parsable --array=0 --export=ALL,MU_OVERRIDE=0.0,LABEL_OVERRIDE=minsr_mu0 scripts/N2_spring_mu_stability_array.sh)
C=$(sbatch --parsable --dependency="afterany:${J}" scripts/N2_collect_minsr_mu0.sh)
printf 'MINSR_JOB_ID=%s\nCOLLECTOR_JOB_ID=%s\n' "$J" "$C" | tee n2_minsr_mu0_job_ids.txt
