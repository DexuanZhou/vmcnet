#!/bin/bash
#SBATCH --job-name=N2eq-kfac-p1000-n1000
#SBATCH --account=def-ortner
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gpus-per-node=h100:1
#SBATCH --exclude=fc10512,fc10420
#SBATCH --output=/scratch/dexuan1/runs/logs/%x-%j.out
#SBATCH --error=/scratch/dexuan1/runs/logs/%x-%j.err
set -euo pipefail
REPO=/scratch/dexuan1/vmcnet
RUN_DIR=/scratch/dexuan1/runs/kfac_pre/N2eq_R2068_kfac_pre1000_n1000
VENV=/home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311
[[ ! -e ${RUN_DIR} ]] || { echo "Refusing to overwrite ${RUN_DIR}" >&2; exit 2; }
module --force purge;module load StdEnv/2023 gcc/12.3 python/3.11.5
source ${VENV}/bin/activate
export TMPDIR=${SLURM_TMPDIR:-/scratch/dexuan1/tmp} WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p ${TMPDIR} $(dirname ${RUN_DIR}) /scratch/dexuan1/runs/logs;cd ${REPO}
source experiments/fir_n4096_smoke/scripts/gpu_health_check.sh /tmp/N2eq_kfac_p1000_n1000_nvidia_smi_${SLURM_JOB_ID}.txt
META_TMP=${RUN_DIR}.run_metadata.json.tmp;GPU_TMP=${RUN_DIR}.gpumem.csv.tmp
cat >${META_TMP} <<'EOF'
{"system":"N2eq","bond_length_bohr":2.068,"run_name":"N2eq_R2068_kfac_pre1000_n1000","family":"kfac_pre","nepochs":1000,"nchains":1000,"nburn":5000,"nsteps_per_param_update":10,"clip_center":"mean","clip_threshold":5.0,"eval_nepochs":0,"optimizer_type":"kfac","learning_rate":0.05,"schedule_type":"inverse_time","learning_decay_rate":0.0001,"checkpoint_every":1000,"ion_pos":"((0.0,0.0,-1.034),(0.0,0.0,1.034))","ion_charges":"(7.0,7.0)","nelec":"(7,7)","fresh_initialization":true}
EOF
echo 'timestamp,index,memory_used_mib' >${GPU_TMP}
(while true;do nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits >>${GPU_TMP} 2>/dev/null||true;sleep 5;done)& M=$!
cleanup(){ kill $M 2>/dev/null||true;wait $M 2>/dev/null||true;};trap cleanup EXIT INT TERM
START=$(date +%s);set +e
vmc-molecule --config.logdir=${RUN_DIR} --config.save_to_current_datetime_subfolder=False --config.subfolder_name=NONE \
 --config.problem.ion_pos='((0.0,0.0,-1.034),(0.0,0.0,1.034))' --config.problem.ion_charges='(7.0,7.0)' --config.problem.nelec='(7,7)' \
 --config.distribute=False --config.eval.nepochs=0 --config.wandb.mode=disabled \
 --config.vmc.nchains=1000 --config.vmc.nburn=5000 --config.vmc.nsteps_per_param_update=10 \
 --config.vmc.clip_threshold=5.0 --config.vmc.clip_center=mean --config.vmc.nepochs=1000 \
 --config.vmc.checkpoint_every=1000 --config.vmc.best_checkpoint_every=1000 \
 --config.vmc.optimizer_type=kfac --config.vmc.optimizer.kfac.schedule_type=inverse_time \
 --config.vmc.optimizer.kfac.learning_rate=0.05 --config.vmc.optimizer.kfac.learning_decay_rate=0.0001
STATUS=$?;set -e;END=$(date +%s);cleanup;trap - EXIT INT TERM
mkdir -p ${RUN_DIR};mv ${META_TMP} ${RUN_DIR}/run_metadata.json;mv ${GPU_TMP} ${RUN_DIR}/gpumem.csv
mv /tmp/N2eq_kfac_p1000_n1000_nvidia_smi_${SLURM_JOB_ID}.txt ${RUN_DIR}/nvidia_smi.txt 2>/dev/null||true
printf 'start_unix,end_unix,elapsed_seconds,nepochs,status\n%s,%s,%s,1000,%s\n' "$START" "$END" "$((END-START))" "$STATUS" >${RUN_DIR}/run_timing.csv
exit $STATUS
