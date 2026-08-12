#!/bin/bash
#SBATCH --job-name=C-wssr-rel-E500
#SBATCH --account=def-ortner
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420,fc10407
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

SOURCE=/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage2_E2000
RUN=/scratch/dexuan1/runs/C_wssr_rank1600_reliability_indicator_replay_E500
META=${RUN}_metadata
CHECKPOINT=${SOURCE}/checkpoints/101500.npz

test -f "${CHECKPOINT}"
[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite ${RUN}" >&2; exit 2; }
[[ ! -e "${META}" ]] || { echo "refusing to overwrite ${META}" >&2; exit 2; }

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd /scratch/dexuan1/vmcnet
hostname > "${META}/hostname.txt"
git rev-parse HEAD > "${META}/git_commit.txt"
git status --short > "${META}/git_status_short.txt"
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"

python - "${SOURCE}/config.json" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
vmc = config["vmc"]
wssr = vmc["optimizer"]["wssr_warm_svd_right"]
expected = {
    "nchains": (vmc["nchains"], 1000),
    "mcmc_steps": (vmc["nsteps_per_param_update"], 10),
    "rank": (wssr["sr_rank"], 1600),
    "rank_max": (wssr["sr_rank_max"], 1600),
    "storage_rank": (wssr["sr_storage_rank"], 1600),
    "working_rank": (wssr["svd_working_rank"], 1600),
    "eta": (wssr["eta"], 0.3),
    "learning_rate": (wssr["learning_rate"], 0.04),
    "learning_decay_rate": (wssr["learning_decay_rate"], 1e-4),
    "damping": (wssr["damping"], 3e-4),
    "relative_cutoff": (wssr["relative_singular_value_cutoff"], 3e-4),
    "tikhonov_lambda": (wssr["tikhonov_lambda"], 1e-3),
    "warm_iterations": (wssr["svd_maxiter_warm"], 2),
    "complement_weight": (wssr["complement_weight"], 0.0),
}
bad = {key: values for key, values in expected.items() if values[0] != values[1]}
assert not bad, bad
assert config["problem"]["ion_charges"] == [6.0]
assert config["problem"]["nelec"] == [4, 2]
assert wssr["experimental_mode"] == "none"
assert wssr["adaptive_complement_beta"] == 0.0
print("verified source configuration", expected)
PY

printf '%s\n' \
  "scientific_question=can_read_only_WSSR_reliability_indicators_predict_raw_local_energy_tails" \
  "source_checkpoint=${CHECKPOINT}" \
  "source_epoch=101500" \
  "target_epoch=102000" \
  "additional_training_steps=500" \
  "update_mathematics_changed=false" \
  "rank=1600" \
  "eta=0.3" \
  "learning_rate=0.04" \
  "learning_decay_rate=0.0001" \
  "damping=0.0003" \
  "relative_singular_value_cutoff=0.0003" \
  "tikhonov_lambda=0.001" \
  "svd_maxiter_warm=2" \
  "ssi_diagnostic_comparison=warm1_vs_warm2" \
  "complement_weight=0.0" \
  "reload_new_optimizer_state=false" \
  "reload_reburn=false" \
  > "${META}/protocol.txt"

vmc-molecule \
  --reload.logdir="${SOURCE}" --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/101500.npz \
  --reload.new_optimizer_state=False --reload.reburn=False \
  --reload.append=True --reload.same_logdir=False \
  --config.logdir="${RUN}" --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
  --config.vmc.nepochs=102000 --config.vmc.check_for_nans=True \
  --config.vmc.checkpoint_every=500 --config.vmc.best_checkpoint_every=500 \
  --config.eval.nburn=0 --config.eval.nepochs=0 \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_reference_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.update_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.reliability_diagnostics=True \
  --config.vmc.optimizer.wssr_warm_svd_right.burst_diagnostics_payload=False \
  --config.wandb.mode=disabled

test -f "${RUN}/checkpoints/102000.npz"
test -f "${RUN}/training_metrics.csv"
test -f "${RUN}/wssr_reliability_update_weighted_ritz_residual.txt"
# A checkpoint named 101500 stores the state immediately before the logged
# epoch-101500 update, so the inclusive replay through 102000 has 501 rows.
test "$(wc -l < "${RUN}/wssr_reliability_update_weighted_ritz_residual.txt")" -eq 501
