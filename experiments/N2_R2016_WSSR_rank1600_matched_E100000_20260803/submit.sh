#!/bin/bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
rank=${WSSR_RANK:-1600}
JOB_FILE=${HERE}/job_ids_rank${rank}.txt

stage1=$(sbatch --parsable --job-name="N2-WSSR-r${rank}-50k" \
  --export="ALL,WSSR_RANK=${rank}" "${HERE}/stage1_train.sh")
stage2=$(sbatch --parsable --job-name="N2-WSSR-r${rank}-100k" \
  --export="ALL,WSSR_RANK=${rank}" --dependency="afterok:${stage1}" \
  "${HERE}/stage2_resume_eval.sh")

printf 'stage1=%s\nstage2=%s\n' "${stage1}" "${stage2}" | tee "${JOB_FILE}"
