#!/bin/bash
set -euo pipefail
cd /scratch/dexuan1/vmcnet
ARRAY=$(sbatch --parsable experiments/C_exact_first_complement/scripts/stage1_array.sh)
COLLECT=$(sbatch --parsable --dependency=afterany:${ARRAY} experiments/C_exact_first_complement/scripts/collect_stage1.sh)
printf 'stage1_array=%s\nstage1_collector=%s\n' "${ARRAY}" "${COLLECT}"
