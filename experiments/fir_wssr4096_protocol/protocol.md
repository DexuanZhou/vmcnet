# Fir C/N2 4096-walker WSSR protocol

Commit at protocol creation: `24093f2de61b4dd9d3bab4b1ee75b47a78799af2`.

## Authoritative implementation

The completed C and corrected N2 n=4096 smoke runs resolve to
`optimizer_type=wssr_warm_svd_right`. The code path is
`initialize_wssr_warm_svd_right` ->
`construct_wssr_warm_svd_right_update_param_fn` ->
`compute_wssr_warm_svd_right_core_update` ->
`wssr_warm_svd_right_core_update` -> `right_warm_start_svd`.

It explicitly constructs the augmented WSSR operator and uses a truncated,
right-subspace warm-start SVD. It is neither a full SVD nor the separate
matrix-free implementation. All formal runs use `store_warm_u=False`,
`svd_maxiter_initial=8`, Tikhonov regularization, and zero complement weight.

| system | lr | history eta | damping | rank 400 warm | rank 800 warm |
|---|---:|---:|---:|---:|---:|
| C | 0.02 | 0.8 | 0.0003 | 1 | 2 |
| N2 R=2.068 | 0.002 | 0.2 | 0.0003 | 1 | 2 |

For each rank, `sr_rank`, `sr_rank_max`, `sr_storage_rank`, and
`svd_working_rank` equal the requested rank. `norm_constraint=0.001`, inverse
time decay `1e-4`, and `complement_weight=0.0` remain fixed.

## Existing n=4096 timing evidence

| system | rank | median s/epoch | max memory MiB |
|---|---:|---:|---:|
| C | 400 | 0.140 | 22219 |
| C | 800 | 0.240 | 42699 |
| C | 1600 | 0.457 | 42701 |
| N2 | 400 | 0.404 | 43195 |
| N2 | 800 | 0.520 | 43195 |
| N2 | 1600 | 0.770 | 51387 |

Rank therefore changes runtime and, at some thresholds, memory. New matched
five-epoch rank-400 and rank-800 gates are submitted before screening.

## Checkpoints

- C: fresh `(4,2)` C KFAC n=4096 pre5000, job 49621891.
- N2: validated `/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2/kfac_preliminary/checkpoints/5000.npz`, 4096 unique walkers, R=2.068 Bohr.

## Proposed 500-epoch screening

C candidates:

1. rank400/warm1, eta=0.8, lr=0.02 (established baseline);
2. rank800/warm2, eta=0.8, lr=0.02 (best rank/cost production baseline);
3. rank800/warm2, eta=0.2, lr=0.002 (previous strong conservative control).

N2 candidates:

1. rank400/warm1, eta=0.2, lr=0.002 (corrected n=4096 baseline);
2. rank800/warm2, eta=0.2, lr=0.002 (corrected n=4096 baseline);
3. rank800/warm2, eta=0.3, lr=0.002 (best established history-eta alternative from the completed n=1000 E100000 sweep).

All candidates within a system restore the same model, walker state, and PRNG
state from the same KFAC checkpoint and create a new WSSR optimizer state.
Only existing metrics are used for safety and selection. No optimizer behavior
or implementation is modified.
