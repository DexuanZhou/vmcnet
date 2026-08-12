#!/bin/bash
set -euo pipefail
cd /scratch/dexuan1/vmcnet/experiments/fir_n4096_smoke
ROOT=/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2
[[ ! -e "${ROOT}" ]] || { echo "Refusing to overwrite ${ROOT}" >&2; exit 2; }
mkdir -p /scratch/dexuan1/runs/logs
EXPORT="ALL,CORRECTED_ROOT=${ROOT},CORRECTED_TAG=_retry1"

KFAC=$(sbatch --parsable --export="${EXPORT}" scripts/N2_kfac_preliminary_n4096.sh)
VALIDATION=$(sbatch --parsable --export="${EXPORT}" --dependency="afterok:${KFAC}" scripts/N2_validate_kfac_n4096.sh)
WSSR=$(sbatch --parsable --export="${EXPORT}" --dependency="afterok:${VALIDATION}" scripts/N2_corrected_wssr_array.sh)
COLLECTOR=$(sbatch --parsable --export="${EXPORT}" --dependency="afterany:${WSSR}" scripts/N2_collect_corrected.sh)

cat > corrected_pipeline_retry1_job_ids.txt <<EOF
KFAC_JOB_ID=${KFAC}
VALIDATION_JOB_ID=${VALIDATION}
WSSR_ARRAY_JOB_ID=${WSSR}
COLLECTOR_JOB_ID=${COLLECTOR}
RETRY_ROOT=${ROOT}
EOF
cat > corrected_pipeline_retry1_commands.txt <<EOF
sbatch --parsable --export=${EXPORT} scripts/N2_kfac_preliminary_n4096.sh
sbatch --parsable --export=${EXPORT} --dependency=afterok:${KFAC} scripts/N2_validate_kfac_n4096.sh
sbatch --parsable --export=${EXPORT} --dependency=afterok:${VALIDATION} scripts/N2_corrected_wssr_array.sh
sbatch --parsable --export=${EXPORT} --dependency=afterany:${WSSR} scripts/N2_collect_corrected.sh
EOF
cat corrected_pipeline_retry1_job_ids.txt
