# C WSSR matched E100000 benchmark

Three independent WSSR trajectories restart from:

`/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1/checkpoints/1000.npz`

They match the completed C SPRING/KFAC benchmark in system, architecture,
starting model/walkers/PRNG state, 1,000 training walkers, 100,000 optimization
epochs, and the 2,000-walker × 20,000-epoch frozen evaluation.

All variants use one exact first truncated-SVD initialization. Starting at the
second WSSR update, the normal right-warm path uses two SSI iterations.

The validation job must pass before the three-task GPU array is released.
No result should be interpreted from training energy alone; use
`eval/statistics.json` and raw frozen local energies.
