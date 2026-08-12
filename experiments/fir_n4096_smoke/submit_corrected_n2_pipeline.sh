#!/bin/bash
set -euo pipefail
cd /scratch/dexuan1/vmcnet/experiments/fir_n4096_smoke
mkdir -p /scratch/dexuan1/runs/logs

KFAC=$(sbatch --parsable scripts/N2_kfac_preliminary_n4096.sh)
VALIDATION=$(sbatch --parsable --dependency="afterok:${KFAC}" scripts/N2_validate_kfac_n4096.sh)
WSSR=$(sbatch --parsable --dependency="afterok:${VALIDATION}" scripts/N2_corrected_wssr_array.sh)
COLLECTOR=$(sbatch --parsable --dependency="afterany:${WSSR}" scripts/N2_collect_corrected.sh)

cat > corrected_pipeline_job_ids.txt <<EOF
KFAC_JOB_ID=${KFAC}
VALIDATION_JOB_ID=${VALIDATION}
WSSR_ARRAY_JOB_ID=${WSSR}
COLLECTOR_JOB_ID=${COLLECTOR}
EOF
cat > corrected_pipeline_commands.txt <<EOF
sbatch --parsable scripts/N2_kfac_preliminary_n4096.sh
sbatch --parsable --dependency=afterok:${KFAC} scripts/N2_validate_kfac_n4096.sh
sbatch --parsable --dependency=afterok:${VALIDATION} scripts/N2_corrected_wssr_array.sh
sbatch --parsable --dependency=afterany:${WSSR} scripts/N2_collect_corrected.sh
EOF
cat corrected_pipeline_job_ids.txt
