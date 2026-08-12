#!/bin/bash
set -euo pipefail
cd /scratch/dexuan1/vmcnet
JV=$(sbatch --parsable experiments/N2_adaptive_complement/validate_checkpoint.sh)
JR=$(sbatch --parsable --dependency=afterok:$JV experiments/N2_adaptive_complement/stage1_replay.sh)
JT=$(sbatch --parsable --dependency=afterok:$JR experiments/N2_adaptive_complement/stage2_array.sh)
JE=$(sbatch --parsable --dependency=afterok:$JT experiments/N2_adaptive_complement/frozen_eval_array.sh)
JC=$(sbatch --parsable --dependency=afterok:$JE experiments/N2_adaptive_complement/collect_results.sh)
printf 'validation=%s\nstage1_replay=%s\nstage2_E500=%s\nfrozen_eval=%s\ncollector=%s\n' "$JV" "$JR" "$JT" "$JE" "$JC" | tee experiments/N2_adaptive_complement/job_ids.txt
