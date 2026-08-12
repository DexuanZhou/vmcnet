#!/bin/bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PRE=$(sbatch --parsable "${HERE}/kfac_pre5000.sh")
MAIN=$(sbatch --parsable --dependency="afterok:${PRE}" "${HERE}/main_100k_array.sh")

printf 'kfac_pre5000_job=%s\nmain_array_job=%s\n' "${PRE}" "${MAIN}" \
  | tee "${HERE}/job_ids.txt"
