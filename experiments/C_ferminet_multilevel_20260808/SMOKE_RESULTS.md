# Strict multilevel smoke result

Numerical gate passed for both SPRING and WSSR on job `53771343`.

| invariant | SPRING | WSSR |
|---|---:|---:|
| coarse parameters | 11,048 | 11,048 |
| fine parameters | 27,240 | 27,240 |
| sign mismatches | 0 | 0 |
| max log-amplitude error | 0.0 | 0.0 |
| schedule count before conversion | 1 | 1 |
| schedule count after conversion | 1 | 1 |
| history norm relative error | 0.0 | 0.0 (`sr_o`, `u`) |
| successful fine metrics | 11 | 11 |

The Slurm array elements have exit status 1 only because the shell assertion
expected ten metric rows.  VMCNet stores the zero-based epoch in a checkpoint,
so a checkpoint named `2.npz` reloads with `start_epoch=1`; targeting epoch 12
therefore executes eleven rows.  Both runs completed through checkpoint 12
without NaNs before that bookkeeping assertion.  The maintained script is
corrected to target checkpoint 11 for exactly ten rows.

The proposed production hierarchy was also audited independently:

| level | backflow widths | parameters |
|---|---|---:|
| L0 | `((64,8),(64,))` | 27,624 |
| L1 | `((128,16),(128,16),(128,))` | 134,944 |
| L2 | `((256,16),(256,16),(256,16),(256,))` | 670,896 |

The L0→L1 and L1→L2 transitions had zero sign mismatches.  Their maximum fp32
log-amplitude discrepancies were respectively 0 and `5.72e-6`, consistent with
different GEMM accumulation order around an algebraically exact embedding.
