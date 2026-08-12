#!/bin/bash
#SBATCH --job-name=N2-WSSR-r1600-50k
#SBATCH --account=def-ortner
#SBATCH --time=11:55:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_N2_R2016_WSSR_r1600_20260803
REPO=/scratch/dexuan1/vmcnet
RANK=${WSSR_RANK:-1600}
SOURCE=/scratch/dexuan1/runs/old/N2_R2016/e100000_followup/N2eq/N2eq_kfac_pre5000
CHECKPOINT=${SOURCE}/checkpoints/5000.npz
if [[ "${RANK}" == "1600" ]]; then
  RUN=/scratch/dexuan1/runs/N2_R2016_WSSR_rank1600_matched_E100000_20260803_stage1_retry1
else
  RUN=/scratch/dexuan1/runs/N2_R2016_WSSR_rank${RANK}_matched_E100000_20260804_stage1
fi
META=${RUN}_metadata
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

[[ -f "${CHECKPOINT}" ]] || { echo "missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${SNAPSHOT}/SOURCE_SNAPSHOT.txt" ]] || { echo "missing source snapshot" >&2; exit 2; }
[[ ! -e "${RUN}" && ! -e "${META}" ]] || {
  echo "refusing to overwrite ${RUN} or ${META}" >&2
  exit 2
}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export PYTHONPATH="${SNAPSHOT}"
# VMCNet records the source revision at startup.  The immutable source snapshot
# deliberately has no .git directory, so expose the originating repository's
# metadata while retaining the snapshot as both the import path and work tree.
export GIT_DIR="${REPO}/.git"
export GIT_WORK_TREE="${SNAPSHOT}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${META}" /scratch/dexuan1/runs/logs
cd "${SNAPSHOT}"

cp SOURCE_SNAPSHOT.txt "${META}/"
hostname > "${META}/hostname.txt"
nvidia-smi > "${META}/nvidia_smi_start.txt"
scontrol show job "${SLURM_JOB_ID}" > "${META}/slurm_job.txt"
python - <<'PY' > "${META}/environment_and_checkpoint.txt"
import json
import os
import numpy as np
import jax
import vmcnet

snapshot = "/scratch/dexuan1/vmcnet_snapshot_N2_R2016_WSSR_r1600_20260803"
source = "/scratch/dexuan1/runs/old/N2_R2016/e100000_followup/N2eq/N2eq_kfac_pre5000"
checkpoint = source + "/checkpoints/5000.npz"
print("vmcnet_file", vmcnet.__file__)
print("jax_version", jax.__version__)
print("jax_backend", jax.default_backend())
print("jax_devices", jax.devices())
assert os.path.realpath(vmcnet.__file__).startswith(snapshot + "/")
assert jax.default_backend() == "gpu"
with open(source + "/config.json") as f:
    config = json.load(f)
assert config["problem"]["ion_pos"] == [[0.0, 0.0, -1.008], [0.0, 0.0, 1.008]]
assert config["problem"]["ion_charges"] == [7.0, 7.0]
assert config["problem"]["nelec"] == [7, 7]
assert config["vmc"]["nchains"] == 1000
assert config["vmc"]["nsteps_per_param_update"] == 10
state = np.load(checkpoint, allow_pickle=True)
assert int(state["e"]) == 4999
for key in ("d", "p", "o", "k"):
    assert key in state.files
print("checkpoint_internal_epoch", int(state["e"]))
print("checkpoint_validation PASS")
PY

cat > "${META}/protocol.txt" <<EOF
purpose=matched N2 rank1600 WSSR stage 1 of E100000
system=N2
bond_length_bohr=2.016
electron_configuration=(7,7)
source_checkpoint=${CHECKPOINT}
source_checkpoint_internal_epoch=4999
training_nchains=1000
stage_target_epoch=50000
nsteps_per_param_update=10
clip_threshold=5
clip_center=mean
optimizer=wssr_warm_svd_right
rank=${RANK}
ssi_initial=40
ssi_warm=2
eta_S=0.2
eta_g=0.0
learning_rate=0.002
learning_decay_rate=0.0001
relative_singular_value_cutoff=0.0003
tikhonov_lambda=0.001
norm_constraint=0.001
complement_weight=0.0
solution_recurrence_mode=none
reload_new_optimizer_state=True
reload_reburn=False
EOF

vmc-molecule \
  --reload.logdir="${SOURCE}" \
  --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/5000.npz \
  --reload.new_optimizer_state=True \
  --reload.reburn=False \
  --reload.append=False \
  --reload.same_logdir=False \
  --config.logdir="${RUN}" \
  --config.base_logdir="${RUN}" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name=NONE \
  --config.problem.ion_pos='((0.,0.,-1.008),(0.,0.,1.008))' \
  --config.problem.ion_charges='(7.,7.)' \
  --config.problem.nelec='(7,7)' \
  --config.vmc.nchains=1000 \
  --config.vmc.nburn=5000 \
  --config.vmc.nepochs=50000 \
  --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.clip_center=mean \
  --config.vmc.clip_threshold=5.0 \
  --config.vmc.check_for_nans=True \
  --config.vmc.checkpoint_every=5000 \
  --config.vmc.best_checkpoint_every=5000 \
  --config.eval.nepochs=0 \
  --config.vmc.optimizer_type=wssr_warm_svd_right \
  --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.002 \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta=0.2 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_S=0.2 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_g=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.enable_gradient_transport=False \
  --config.vmc.optimizer.wssr_warm_svd_right.adaptive_S_average=False \
  --config.vmc.optimizer.wssr_warm_svd_right.adaptive_g_average=False \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_bias_correction=False \
  --config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.relative_singular_value_cutoff=0.0003 \
  --config.vmc.optimizer.wssr_warm_svd_right.tikhonov_lambda=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov \
  --config.vmc.optimizer.wssr_warm_svd_right.mixed_precision_solve=False \
  --config.vmc.optimizer.wssr_warm_svd_right.constrain_norm=True \
  --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank="${RANK}" \
  --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=True \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=40 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first=False \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first_force=False \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_reference_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.update_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.reliability_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.burst_diagnostics_payload=False \
  --config.vmc.optimizer.wssr_warm_svd_right.solution_recurrence_mode=none \
  --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=none \
  --config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_beta=0.0 \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0.0 \
  --config.wandb.mode=disabled

[[ -f "${RUN}/checkpoints/50000.npz" ]] || {
  echo "missing epoch-50000 checkpoint" >&2
  exit 4
}
