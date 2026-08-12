#!/bin/bash
set -euo pipefail
cd /scratch/dexuan1/vmcnet
C_KFAC=$(sbatch --parsable --export=ALL,SYSTEM=C experiments/fir_wssr4096_kfac1000/scripts/kfac1000.sh)
N_KFAC=$(sbatch --parsable --export=ALL,SYSTEM=N2 experiments/fir_wssr4096_kfac1000/scripts/kfac1000.sh)
C_VAL=$(sbatch --parsable --dependency=afterok:${C_KFAC} --export=ALL,SYSTEM=C experiments/fir_wssr4096_kfac1000/scripts/validate.sh)
N_VAL=$(sbatch --parsable --dependency=afterok:${N_KFAC} --export=ALL,SYSTEM=N2 experiments/fir_wssr4096_kfac1000/scripts/validate.sh)
C_SEL=$(sbatch --parsable --dependency=afterok:${C_VAL} --export=ALL,SYSTEM=C experiments/fir_wssr4096_kfac1000/scripts/select_array.sh)
N_SEL=$(sbatch --parsable --dependency=afterok:${N_VAL} --export=ALL,SYSTEM=N2 experiments/fir_wssr4096_kfac1000/scripts/select_array.sh)
COLLECT=$(sbatch --parsable --dependency=afterok:${C_SEL}:${N_SEL} experiments/fir_wssr4096_kfac1000/scripts/collector.sh)
printf 'C_KFAC=%s\nN2_KFAC=%s\nC_VALIDATE=%s\nN2_VALIDATE=%s\nC_SELECT=%s\nN2_SELECT=%s\nCOLLECT=%s\n' "$C_KFAC" "$N_KFAC" "$C_VAL" "$N_VAL" "$C_SEL" "$N_SEL" "$COLLECT" | tee experiments/fir_wssr4096_kfac1000/results/job_ids.txt
