#!/bin/bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${HERE}/job_ids_etaS08_etaG08.txt
TAG=etaS08_etaG08_retry1
[[ ! -e "${OUT}" ]] || { echo "refusing to overwrite ${OUT}" >&2; exit 2; }

common_export="ALL,ETA_S=0.8,ETA_G=0.8,RUN_TAG=${TAG}"

n2_smoke=$(sbatch --parsable --job-name=N2-WSSR-r1600-e08-smoke \
  --export="${common_export},SYSTEM=N2" "${HERE}/smoke.sh")
n2_stage1=$(sbatch --parsable --job-name=N2-WSSR-r1600-e08-50k \
  --export="${common_export},SYSTEM=N2" --dependency=afterok:${n2_smoke} \
  "${HERE}/stage1_train.sh")
n2_stage2=$(sbatch --parsable --job-name=N2-WSSR-r1600-e08-100k \
  --export="${common_export},SYSTEM=N2" --dependency=afterok:${n2_stage1} \
  "${HERE}/stage2_resume_eval.sh")

h2o_smoke=$(sbatch --parsable --job-name=H2O-WSSR-r1600-e08-smoke \
  --export="${common_export},SYSTEM=H2O" "${HERE}/smoke.sh")
h2o_stage1=$(sbatch --parsable --job-name=H2O-WSSR-r1600-e08-50k \
  --export="${common_export},SYSTEM=H2O" --dependency=afterok:${h2o_smoke} \
  "${HERE}/stage1_train.sh")
h2o_stage2=$(sbatch --parsable --job-name=H2O-WSSR-r1600-e08-100k \
  --export="${common_export},SYSTEM=H2O" --dependency=afterok:${h2o_stage1} \
  "${HERE}/stage2_resume_eval.sh")

printf 'eta_S=0.8\neta_g=0.8\nrun_tag=%s\n' "${TAG}" > "${OUT}"
printf 'n2_smoke=%s\nn2_stage1=%s\nn2_stage2=%s\n' \
  "${n2_smoke}" "${n2_stage1}" "${n2_stage2}" >> "${OUT}"
printf 'h2o_smoke=%s\nh2o_stage1=%s\nh2o_stage2=%s\n' \
  "${h2o_smoke}" "${h2o_stage1}" "${h2o_stage2}" >> "${OUT}"
cat "${OUT}"
