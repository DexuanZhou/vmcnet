#!/bin/bash
set -euo pipefail
cd /scratch/dexuan1/vmcnet/experiments/fir_n4096_smoke
ROOT=/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2
CHECKPOINT=${ROOT}/kfac_preliminary/checkpoints/5000.npz
[[ -f "${CHECKPOINT}" ]] || { echo "Missing ${CHECKPOINT}" >&2; exit 2; }
[[ ! -e "${ROOT}/validation_eval" ]] || { echo 'Refusing to overwrite validation output' >&2; exit 2; }
EXPORT="ALL,CORRECTED_ROOT=${ROOT},CORRECTED_TAG=_retry1"

VALIDATION=$(sbatch --parsable --export="${EXPORT}" scripts/N2_validate_kfac_n4096.sh)
WSSR=$(sbatch --parsable --export="${EXPORT}" --dependency="afterok:${VALIDATION}" scripts/N2_corrected_wssr_array.sh)
COLLECTOR=$(sbatch --parsable --export="${EXPORT}" --dependency="afterany:${WSSR}" scripts/N2_collect_corrected.sh)

cat > corrected_downstream_retry1_job_ids.txt <<EOF
KFAC_JOB_ID=49593613
VALIDATION_JOB_ID=${VALIDATION}
WSSR_ARRAY_JOB_ID=${WSSR}
COLLECTOR_JOB_ID=${COLLECTOR}
EOF
cat > corrected_downstream_retry1_commands.txt <<EOF
sbatch --parsable --export=${EXPORT} scripts/N2_validate_kfac_n4096.sh
sbatch --parsable --export=${EXPORT} --dependency=afterok:${VALIDATION} scripts/N2_corrected_wssr_array.sh
sbatch --parsable --export=${EXPORT} --dependency=afterany:${WSSR} scripts/N2_collect_corrected.sh
EOF
cat corrected_downstream_retry1_job_ids.txt
