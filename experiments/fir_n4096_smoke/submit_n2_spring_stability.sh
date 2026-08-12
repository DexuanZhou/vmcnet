#!/bin/bash
set -euo pipefail
cd /scratch/dexuan1/vmcnet/experiments/fir_n4096_smoke
ROOT=/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2/spring_stability
[[ ! -e "${ROOT}" ]] || { echo "Refusing overwrite ${ROOT}" >&2; exit 2; }
ARRAY=$(sbatch --parsable scripts/N2_spring_stability_array.sh)
COLLECTOR=$(sbatch --parsable --dependency="afterany:${ARRAY}" scripts/N2_collect_spring_stability.sh)
cat > n2_spring_stability_job_ids.txt <<EOF
ARRAY_JOB_ID=${ARRAY}
COLLECTOR_JOB_ID=${COLLECTOR}
EOF
cat > n2_spring_stability_commands.txt <<EOF
sbatch --parsable scripts/N2_spring_stability_array.sh
sbatch --parsable --dependency=afterany:${ARRAY} scripts/N2_collect_spring_stability.sh
EOF
cat n2_spring_stability_job_ids.txt
