#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
"${HERE}/validate.py"
job=$(sbatch --parsable "${HERE}/fir_n4096_array.sh")
printf '%s\n' "${job}" | tee "${HERE}/submitted_job_id.txt"
