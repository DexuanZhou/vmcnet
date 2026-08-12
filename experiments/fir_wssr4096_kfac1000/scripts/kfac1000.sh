#!/bin/bash
#SBATCH --job-name=kfac4096-p1000
#SBATCH --account=def-ortner
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
set -euo pipefail
: "${SYSTEM:?SYSTEM=C or N2 required}"
if [[ $SYSTEM == C ]]; then SOURCE=/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1; ROOT=/scratch/dexuan1/runs/wssr_C_4096_kfac1000; else SOURCE=/scratch/dexuan1/runs/kfac_pre/N2eq_R2068_kfac_pre5000; ROOT=/scratch/dexuan1/runs/wssr_N2_4096_kfac1000; fi
META=${ROOT}_metadata; REPO=/scratch/dexuan1/vmcnet
[[ -f ${SOURCE}/config.json && ! -e ${ROOT} ]] || { echo missing-source-or-output-exists >&2; exit 2; }
module --force purge; module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p ${META} /scratch/dexuan1/runs/logs; cd ${REPO}
source experiments/fir_n4096_smoke/scripts/gpu_health_check.sh ${META}/nvidia_smi_start.txt
python - <<'PY' > ${META}/jax_backend.txt
import jax
print(jax.default_backend(),jax.devices()); assert jax.default_backend()=='gpu'
PY
git rev-parse HEAD > ${META}/git_commit.txt; scontrol show job ${SLURM_JOB_ID} > ${META}/slurm_job.txt
echo 'timestamp,memory_used_mib,utilization_gpu_percent' > ${META}/gpu_memory_poll.csv
(while true; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits >> ${META}/gpu_memory_poll.csv||true; sleep 1; done)& MON=$!; trap 'kill $MON 2>/dev/null||true' EXIT
vmc-molecule --reload.logdir=${SOURCE} --reload.use_config_file=True --reload.use_checkpoint_file=False --reload.append=False --reload.same_logdir=False \
 --config.logdir=${ROOT} --config.base_logdir=${ROOT} --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
 --config.vmc.optimizer_type=kfac --config.vmc.nchains=4096 --config.eval.nchains=4096 --config.vmc.nburn=5000 \
 --config.vmc.nepochs=1000 --config.vmc.nsteps_per_param_update=10 --config.vmc.check_for_nans=True \
 --config.vmc.checkpoint_every=250 --config.vmc.best_checkpoint_every=250 --config.vmc.disable_checkpointing=False \
 --config.eval.nepochs=0 --config.wandb.mode=disabled
[[ -f ${ROOT}/checkpoints/1000.npz ]] || exit 4
