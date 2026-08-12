#!/bin/bash
#SBATCH --account=def-ortner
#SBATCH --time=11:55:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10405,fc10407,fc10420,fc10512,fc10515,fc10603
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err

set -euo pipefail

SYSTEM=${SYSTEM:?submit with SYSTEM=N2 or SYSTEM=H2O}
ETA_S=${ETA_S:-0.3}
ETA_G=${ETA_G:-0.3}
RUN_TAG=${RUN_TAG:-}
SNAPSHOT=/scratch/dexuan1/vmcnet_snapshot_N2_H2O_WSSR_dualcap_20260810
REPO=/scratch/dexuan1/vmcnet
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311

case "${SYSTEM}" in
  N2)
    SOURCE=/scratch/dexuan1/runs/old/N2_R2016/e100000_followup/N2eq/N2eq_kfac_pre5000
    RUN=/scratch/dexuan1/runs/N2_R2016_WSSR_rank1600_dualcap_E100000_20260810_stage1
    RELOAD_USE_CONFIG=True
    MOLECULE_ARGS=()
    ;;
  H2O)
    SOURCE=/scratch/dexuan1/runs/H2O_kfac_pre5000_N2matched_20260807
    RUN=/scratch/dexuan1/runs/H2O_WSSR_rank1600_dualcap_E100000_20260810_stage1
    RELOAD_USE_CONFIG=False
    MOLECULE_ARGS=(
      --config.problem.ion_pos='((0.,0.,0.),(1.43233673,0.,1.10715266),(-1.43233673,0.,1.10715266))'
      --config.problem.ion_charges='(8.,1.,1.)'
      --config.problem.nelec='(5,5)'
      --config.model.ferminet.ndeterminants=16
      --config.model.ferminet.full_det=True
    )
    ;;
  *) echo "unsupported SYSTEM=${SYSTEM}" >&2; exit 2 ;;
esac
if [[ -n "${RUN_TAG}" ]]; then
  RUN=${RUN}_${RUN_TAG}
fi
CHECKPOINT=${SOURCE}/checkpoints/5000.npz
META=${RUN}_metadata

[[ -f "${CHECKPOINT}" ]] || { echo "missing ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${SNAPSHOT}/SOURCE_SNAPSHOT.txt" ]] || { echo "missing snapshot" >&2; exit 2; }
[[ ! -e "${RUN}" && ! -e "${META}" ]] || {
  echo "refusing to overwrite ${RUN} or ${META}" >&2
  exit 2
}

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
source "${VENV}/bin/activate"
export PYTHONPATH="${SNAPSHOT}"
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

cat > "${META}/protocol.txt" <<EOF
purpose=matched ${SYSTEM} metric-aware dual-cap tail WSSR rank1600 stage 1
system=${SYSTEM}
source_checkpoint=${CHECKPOINT}
preliminary_optimizer=KFAC
preliminary_epochs=5000
training_nchains=1000
stage_target_epoch=50000
nsteps_per_param_update=10
clip_threshold=5
clip_center=mean
optimizer=wssr_warm_svd_right
rank=1600
ssi_initial=40
ssi_warm=2
eta_S=${ETA_S}
eta_g=${ETA_G}
learning_rate=0.002
learning_decay_rate=0.0001
tikhonov_lambda=0.001
norm_constraint=0.001
norm_constraint_mode=function_space
function_norm_constraint=0.0008
euclidean_safety_constraint=0.001
experimental_mode=adaptive_complement
complement_weight=0.0001
adaptive_complement_beta=0.1
adaptive_complement_beta_function=0.1
store_warm_u=False
semi_matrix_free_augmented=True
solution_recurrence_mode=none
checkpoint_format=pytree_leaves_v1
reload_new_optimizer_state=True
reload_reburn=False
EOF

START=$(date +%s)
vmc-molecule \
  --reload.logdir="${SOURCE}" \
  --reload.use_config_file="${RELOAD_USE_CONFIG}" \
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
  "${MOLECULE_ARGS[@]}" \
  --config.vmc.nchains=1000 \
  --config.vmc.nepochs=50000 \
  --config.vmc.nsteps_per_param_update=10 \
  --config.vmc.clip_center=mean \
  --config.vmc.clip_threshold=5.0 \
  --config.vmc.check_for_nans=True \
  --config.vmc.disable_checkpointing=False \
  --config.vmc.checkpoint_every=50000 \
  --config.vmc.best_checkpoint_every=50000 \
  --config.eval.nepochs=0 \
  --config.vmc.optimizer_type=wssr_warm_svd_right \
  --config.vmc.optimizer.wssr_warm_svd_right.schedule_type=inverse_time \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.002 \
  --config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001 \
  --config.vmc.optimizer.wssr_warm_svd_right.eta="${ETA_S}" \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_S="${ETA_S}" \
  --config.vmc.optimizer.wssr_warm_svd_right.eta_g="${ETA_G}" \
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
  --config.vmc.optimizer.wssr_warm_svd_right.norm_constraint_mode=function_space \
  --config.vmc.optimizer.wssr_warm_svd_right.function_norm_constraint=0.0008 \
  --config.vmc.optimizer.wssr_warm_svd_right.euclidean_safety_constraint=0.001 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank=1600 \
  --config.vmc.optimizer.wssr_warm_svd_right.store_warm_u=False \
  --config.vmc.optimizer.wssr_warm_svd_right.semi_matrix_free_augmented=True \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial=40 \
  --config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2 \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first=False \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_first_force=False \
  --config.vmc.optimizer.wssr_warm_svd_right.exact_reference_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.update_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.reliability_diagnostics=False \
  --config.vmc.optimizer.wssr_warm_svd_right.burst_diagnostics_payload=False \
  --config.vmc.optimizer.wssr_warm_svd_right.solution_recurrence_mode=none \
  --config.vmc.optimizer.wssr_warm_svd_right.experimental_mode=adaptive_complement \
  --config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_beta=0.1 \
  --config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_beta_function=0.1 \
  --config.vmc.optimizer.wssr_warm_svd_right.complement_weight=0.0001 \
  --config.wandb.mode=disabled
END=$(date +%s)
printf 'start_unix,end_unix,elapsed_seconds\n%s,%s,%s\n' \
  "${START}" "${END}" "$((END - START))" > "${META}/run_timing.csv"

CHECKPOINT="${RUN}/checkpoints/50000.npz" python - <<'PY' \
  > "${META}/checkpoint_validation.txt"
import os
import numpy as np

path = os.environ["CHECKPOINT"]
with np.load(path, allow_pickle=True) as state:
    assert int(state["e"]) == 49999
    assert state["o_format"].item() == "pytree_leaves_v1"
    nleaves = int(state["o_num_leaves"].item())
    assert nleaves > 0
    assert f"o_leaf_{nleaves - 1:06d}" in state.files
    print("checkpoint", path)
    print("optimizer_format", state["o_format"].item())
    print("optimizer_leaves", nleaves)
    print("validation PASS")
PY
