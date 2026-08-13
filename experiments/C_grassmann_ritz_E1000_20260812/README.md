# C top-200 Procrustes--Grassmann WSSR screen

This is a low-cost causal test of subspace-direction smoothing.  It starts
from the common C KFAC-pre1000 checkpoint and changes only the Grassmann
interpolation weight.  Every arm uses the current batch for the score matrix,
RHS, Rayleigh--Ritz matrix, and solve; no S or gradient averaging is enabled.

The stored state is one smoothed top-200 basis.  At step t, with
`M = U_prev.T @ U_cur = L diag(s) R.T`, the current basis is aligned as
`U_cur @ R @ L.T`, interpolated with the previous smoothed basis, and QR
retracted.  Alpha=1 is the matched current-subspace control.

Protocol: C KFAC-pre1000, 1000 walkers, 10 MCMC moves/update, fixed
Tikhonov lambda 1e-3, lr 0.04, Euclidean norm constraint 1e-3, SSI 40/2,
rank 200, 1000 updates, seed 0.  Arms are alpha in {1.0, 0.75, 0.5, 0.25}.
The scripts request one generic GPU by default.  On a cluster with H100 MIG
slices, specify the slice resource on the `sbatch` command line.

## Run on another machine

Fetch the `wssr` branch and install the repository in an environment with a
working GPU JAX build.  The unit test can be run directly:

```bash
git fetch origin wssr
git checkout wssr
git pull --ff-only origin wssr
python -m pip install -e .
python -m pytest -q tests/units/updates/test_wssr.py \
  -k 'procrustes_grassmann or grassmann_ritz'
```

The committed C KFAC-pre1000 initializer is used by default.  To use a
different copy, point `C_KFAC_PRE1000` at the run directory containing
`checkpoints/1000.npz`.  Set a unique output directory before submission:

```bash
export VMCNET_REPOSITORY="$PWD"
export C_KFAC_PRE1000="$PWD/reproducibility/kfac_initializers/C_kfac_pre1000"
export C_GRASSMANN_ROOT="${SCRATCH:-/tmp/$USER}/runs/C_grassmann_ritz_E1000"
export VMCNET_ENV_ACTIVATE="$VIRTUAL_ENV/bin/activate"  # optional
```

If environment modules are required, provide their setup as a single command,
for example `export VMCNET_MODULE_SETUP='module purge; module load python/3.11'`.
Submit the four training arms and make frozen evaluation depend on them:

```bash
TRAIN_JOB=$(sbatch --parsable --array=0-3%2 \
  experiments/C_grassmann_ritz_E1000_20260812/train_array.sh)
EVAL_JOB=$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}" --array=0-3%2 \
  experiments/C_grassmann_ritz_E1000_20260812/eval_array.sh)
echo "training=${TRAIN_JOB} evaluation=${EVAL_JOB}"
```

Cluster-specific account, partition, and GPU type can be supplied to both
`sbatch` calls.  For example, add
`--account=ACCOUNT --partition=PARTITION --gpus-per-node=GPU_TYPE:1` for an
H100 MIG slice.  Once evaluation is complete, collect the result table with:

```bash
python experiments/C_grassmann_ritz_E1000_20260812/collect.py
```

The collector writes `summary.csv` and `summary.json` under
`$C_GRASSMANN_ROOT`.  Task IDs map to alpha values as follows: 0 -> 1.0,
1 -> 0.75, 2 -> 0.5, and 3 -> 0.25.
