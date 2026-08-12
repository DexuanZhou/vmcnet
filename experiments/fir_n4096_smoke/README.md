# Fir n=4096 WSSR smoke/timing test

Six 50-epoch feasibility runs use 4096 walkers and the common production KFAC checkpoint for each system. All tasks reburn for 5000 steps, create a new optimizer state, use seed 0, disable checkpointing, and run on a full H100 80 GiB GPU. Array concurrency is capped at two.

The explicit implementation is selected by `optimizer_type=wssr_warm_svd_right`; this code revision has no separate `sr_block_operator` or `svd_method` config keys. The manifest records the resolved interpretation `sr_block_operator=False, svd_method=explicit`. The distinct `wssr_warm_svd_right_matfree` optimizer is not used.

The timing launcher wraps only burn-in entry/exit and metrics-file writes. It does not modify WSSR or any optimizer calculation. Steady timing uses epoch-completion intervals for epochs 11–50. `VMCNET_PROFILE_TIMING` is not enabled because this code revision does not implement it.

The stock checkpoint reload restores walker data as well as parameters. The original smoke launcher retained freshly initialized 4096-walker data while loading only the n=1000 checkpoint parameters. That protocol produced invalid N2 sampling and is retained only as historical data under `/scratch/dexuan1/runs/fir_n4096_smoke`; it must not be reused for N2 conclusions.

N2 uses R=2.068 Bohr and eta 0.2 as explicitly requested for this smoke test. Checkpoint output is disabled both by `VMCNET_DISABLE_CHECKPOINTS=1` and `config.vmc.disable_checkpointing=True`; successful collection requires `npz_count=0`.

## Corrected N2 pipeline

`submit_corrected_n2_pipeline.sh` creates a dependency chain consisting of a fresh R=2.068 KFAC preliminary run with 4096 chains, frozen checkpoint validation, the three corrected WSSR smoke tasks, and collection. The WSSR tasks use the full saved n=4096 model and equilibrated walker/sampler state, set `reload.new_optimizer_state=True`, and set `reload.reburn=False`; they do not restore the KFAC optimizer state into WSSR.

The validation job requires 4096 distinct stored walkers, matching model shapes, finite frozen-evaluation results, energy within 2 Ha of -109.48 Ha, variance below 100 Ha², and acceptance between 0.2 and 0.8. Failure exits nonzero and the Slurm `afterok` dependency prevents the WSSR array from starting.

Corrected output is isolated under `/scratch/dexuan1/runs/fir_n4096_corrected/N2`. No WSSR smoke checkpoints are written. The KFAC preliminary run writes only its normal final epoch-5000 checkpoint.
