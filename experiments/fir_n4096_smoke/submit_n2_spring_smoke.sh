#!/bin/bash
set -euo pipefail
cd /scratch/dexuan1/vmcnet/experiments/fir_n4096_smoke
RUN=/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2/spring_smoke/N2eq_R2068_spring_lr001_n4096_corrected_e50
[[ ! -e "${RUN}" ]] || { echo "Refusing to overwrite ${RUN}" >&2; exit 2; }
SPRING=$(sbatch --parsable scripts/N2_corrected_spring_smoke.sh)
COLLECTOR=$(sbatch --parsable --dependency="afterany:${SPRING}" scripts/N2_collect_spring_smoke.sh)
cat > n2_spring_smoke_job_ids.txt <<EOF
SPRING_JOB_ID=${SPRING}
COLLECTOR_JOB_ID=${COLLECTOR}
EOF
cat > n2_spring_smoke_commands.txt <<EOF
sbatch --parsable scripts/N2_corrected_spring_smoke.sh
sbatch --parsable --dependency=afterany:${SPRING} scripts/N2_collect_spring_smoke.sh
EOF
cat n2_spring_smoke_job_ids.txt
