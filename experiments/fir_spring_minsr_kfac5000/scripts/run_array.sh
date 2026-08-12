#!/bin/bash
#SBATCH --job-name=spring-minsr-k5k
#SBATCH --account=def-ortner
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%A_%a.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%A_%a.err
set -euo pipefail
: "${PHASE:?PHASE must be smoke or long}"
case ${SLURM_ARRAY_TASK_ID} in
 0) SYSTEM=C; LABEL=minsr; MU=0.0; LR=0.02; SOURCE=/scratch/dexuan1/runs/wssr_C_4096_protocol/C_KFAC4096_pre5000; ROOT=/scratch/dexuan1/runs/C_minsr_after_kfac5000_E50000;;
 1) SYSTEM=C; LABEL=spring_mu099; MU=0.99; LR=0.02; SOURCE=/scratch/dexuan1/runs/wssr_C_4096_protocol/C_KFAC4096_pre5000; ROOT=/scratch/dexuan1/runs/C_spring_mu099_after_kfac5000_E50000;;
 2) SYSTEM=N2; LABEL=minsr; MU=0.0; LR=0.001; SOURCE=/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2/kfac_preliminary; ROOT=/scratch/dexuan1/runs/N2_minsr_after_kfac5000_E50000;;
 3) SYSTEM=N2; LABEL=spring_mu099; MU=0.99; LR=0.001; SOURCE=/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2/kfac_preliminary; ROOT=/scratch/dexuan1/runs/N2_spring_mu099_after_kfac5000_E50000;;
 *) exit 2;;
esac
if [[ ${PHASE} == smoke ]]; then EPOCHS=200; RUN=${ROOT}_smoke200; DISABLE=True; CKPT=10000; else EPOCHS=50000; RUN=${ROOT}; DISABLE=False; CKPT=10000; fi
# Keep metadata outside RUN: creating RUN before vmcnet starts makes its
# collision-avoidance logic silently redirect scientific output to RUN_1.
META=${RUN}_metadata
[[ -f ${SOURCE}/checkpoints/5000.npz ]] || { echo "missing ${SOURCE}/checkpoints/5000.npz" >&2; exit 2; }
[[ ! -e ${RUN} ]] || { echo "refusing overwrite ${RUN}" >&2; exit 2; }
module --force purge; module load StdEnv/2023 gcc/12.3 python/3.11.5
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true XLA_PYTHON_CLIENT_PREALLOCATE=false VMCNET_PROFILE_TIMING=1 SMOKE_TIMING_DIR=${META}
mkdir -p ${META} /scratch/dexuan1/runs/logs; cd /scratch/dexuan1/vmcnet
source experiments/fir_n4096_smoke/scripts/gpu_health_check.sh ${META}/nvidia_smi.txt
git rev-parse HEAD >${META}/git_commit.txt
printf 'pretraining_epochs=5000\nsource_checkpoint=%s\nfirst_optimizer_epoch=1\nsystem=%s\noptimizer=%s\nmu=%s\nlearning_rate=%s\ndamping=0.001\nnorm_constraint=0.001\nprecision=float32\n' "${SOURCE}/checkpoints/5000.npz" "$SYSTEM" "$LABEL" "$MU" "$LR" >${META}/protocol.txt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits >${META}/gpu_identity.csv
echo 'timestamp,memory_used_mib,utilization_gpu_percent' >${META}/gpu_memory_poll.csv
(while true;do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits >>${META}/gpu_memory_poll.csv||true;sleep 1;done)& M=$!
cleanup(){ kill $M 2>/dev/null||true; }; trap cleanup EXIT INT TERM
set +e
python experiments/fir_n4096_smoke/scripts/timed_reload_launcher.py \
 --reload.logdir=${SOURCE} --reload.checkpoint_relative_file_path=checkpoints/5000.npz --reload.use_config_file=True --reload.use_checkpoint_file=True --reload.new_optimizer_state=True --reload.reburn=False --reload.append=False --reload.same_logdir=False \
 --config.logdir=${RUN} --config.base_logdir=${RUN} --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
 --config.vmc.nchains=4096 --config.eval.nchains=4096 --config.vmc.nepochs=${EPOCHS} --config.vmc.nsteps_per_param_update=10 --config.vmc.check_for_nans=True --config.vmc.disable_checkpointing=${DISABLE} --config.vmc.checkpoint_every=${CKPT} --config.vmc.best_checkpoint_every=${CKPT} --config.eval.nepochs=0 --config.wandb.mode=disabled \
 --config.vmc.optimizer_type=spring --config.vmc.optimizer.spring.learning_rate=${LR} --config.vmc.optimizer.spring.mu=${MU} --config.vmc.optimizer.spring.damping=0.001 --config.vmc.optimizer.spring.constrain_norm=True --config.vmc.optimizer.spring.norm_constraint=0.001 --config.vmc.optimizer.spring.schedule_type=inverse_time --config.vmc.optimizer.spring.learning_decay_rate=0.0001 --config.vmc.optimizer.spring.mixed_precision_solve=False --config.vmc.optimizer.spring.diagnostics=True --config.vmc.optimizer.spring.diagnostics_spectral=False --config.vmc.optimizer.spring.diagnostics_decomposition=True & P=$!
python experiments/fir_spring_minsr_kfac5000/monitor.py --pid $P --run ${RUN} --out ${META}/early_stop.json & Q=$!
wait $P; S=$?; wait $Q||true; cleanup; trap - EXIT INT TERM
exit $S
