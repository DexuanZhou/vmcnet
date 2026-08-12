#!/bin/bash
#SBATCH --job-name=N2-kfac-monitor-submit
#SBATCH --account=def-ortner
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
# Poll a submitted N2 KFAC job every 30 minutes and submit primary N2 follow-ups
# only after a successful job state and a real epoch-5000 checkpoint.
set -euo pipefail
: "${KFAC_JOB_ID:?KFAC_JOB_ID is required}"
HERE=$(cd "$(dirname "$0")" && pwd)
CHECKPOINT=/scratch/dexuan1/runs/e100000_followup/N2eq/N2eq_kfac_pre5000/checkpoints/5000.npz
LOG="${HERE}/monitor_n2_kfac_${KFAC_JOB_ID}.log"
IDS="${HERE}/n2_phase2_job_ids_${KFAC_JOB_ID}.csv"
INTERVAL_SECONDS=1800

log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "${LOG}"; }
log "monitor started for KFAC job ${KFAC_JOB_ID}; interval=${INTERVAL_SECONDS}s"

while true; do
  state=$(sacct -X -n -j "${KFAC_JOB_ID}" --format=State --parsable2 2>/dev/null | sed '/^[[:space:]]*$/d' | head -1 | cut -d'|' -f1 | cut -d' ' -f1)
  state=${state:-UNKNOWN}
  log "KFAC state=${state}"
  case "${state}" in
    COMPLETED)
      if [[ ! -f "${CHECKPOINT}" ]]; then
        log "ERROR: job completed but checkpoint is missing: ${CHECKPOINT}; no downstream submission"
        exit 2
      fi
      break
      ;;
    FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)
      log "ERROR: terminal unsuccessful state ${state}; no downstream submission"
      exit 2
      ;;
  esac
  sleep "${INTERVAL_SECONDS}"
done

log "checkpoint verified: ${CHECKPOINT}"
printf 'run_name,job_id\n' > "${IDS}"
submit() {
  local run=$1 script=$2 job
  job=$(sbatch --parsable "${HERE}/${script}")
  printf '%s,%s\n' "${run}" "${job}" >> "${IDS}"
  log "submitted ${run}: ${job}"
}
submit N2eq_spring_lr002_mu099_nc001_E100000 N2eq_spring_lr002_mu099_nc001_E100000.sh
submit N2eq_wssr_eta03_lr0002_tikhonov_E100000 N2eq_wssr_eta03_lr0002_tikhonov_E100000.sh
submit N2eq_wssr_eta02_lr0002_tikhonov_E100000 N2eq_wssr_eta02_lr0002_tikhonov_E100000.sh
submit N2eq_wssr_eta03_lr0003_tikhonov_E100000 N2eq_wssr_eta03_lr0003_tikhonov_E100000.sh
submit N2eq_wssr_eta03_lr0002_tikhonov_C002_E100000 N2eq_wssr_eta03_lr0002_tikhonov_C002_E100000.sh
log "phase 2 primary submission complete; rescue lr=0.001 intentionally not submitted"
