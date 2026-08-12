#!/bin/bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${HERE}/job_ids.txt
[[ ! -e "${OUT}" ]] || { echo "refusing to overwrite ${OUT}" >&2; exit 2; }

n2_smoke=$(sbatch --parsable --job-name=N2-WSSR-r1600-smoke \
  --export=ALL,SYSTEM=N2 "${HERE}/smoke.sh")
n2_stage1=$(sbatch --parsable --job-name=N2-WSSR-r1600-50k \
  --export=ALL,SYSTEM=N2 --dependency=afterok:${n2_smoke} \
  "${HERE}/stage1_train.sh")
n2_stage2=$(sbatch --parsable --job-name=N2-WSSR-r1600-100k \
  --export=ALL,SYSTEM=N2 --dependency=afterok:${n2_stage1} \
  "${HERE}/stage2_resume_eval.sh")

h2o_smoke=$(sbatch --parsable --job-name=H2O-WSSR-r1600-smoke \
  --export=ALL,SYSTEM=H2O "${HERE}/smoke.sh")
h2o_stage1=$(sbatch --parsable --job-name=H2O-WSSR-r1600-50k \
  --export=ALL,SYSTEM=H2O --dependency=afterok:${h2o_smoke} \
  "${HERE}/stage1_train.sh")
h2o_stage2=$(sbatch --parsable --job-name=H2O-WSSR-r1600-100k \
  --export=ALL,SYSTEM=H2O --dependency=afterok:${h2o_stage1} \
  "${HERE}/stage2_resume_eval.sh")

printf 'n2_smoke=%s\nn2_stage1=%s\nn2_stage2=%s\n' \
  "${n2_smoke}" "${n2_stage1}" "${n2_stage2}" > "${OUT}"
printf 'h2o_smoke=%s\nh2o_stage1=%s\nh2o_stage2=%s\n' \
  "${h2o_smoke}" "${h2o_stage1}" "${h2o_stage2}" >> "${OUT}"
cat "${OUT}"
