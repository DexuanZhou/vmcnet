# Results

Both jobs completed successfully. Training job `52732990` took 00:41:29 on
one full H100. Frozen evaluation job `52738089` took 00:09:27 on one full
H100. No divergence monitor fired, and all 2,000,000 frozen local energies
were finite.

## Frozen endpoint

- Energy: `-37.83287885668468 Ha`
- Standard error: `0.0009056190012943644 Ha`
- Raw variance: `1.209367358517589 Ha^2`
- Integrated autocorrelation: `1.3563219971650557`
- Variance after winsorizing the most extreme 0.1% in each tail:
  `0.566376969107 Ha^2`

## Matched comparisons

| method | checkpoint | frozen energy (Ha) | frozen variance (Ha^2) | new / reference variance |
|---|---:|---:|---:|---:|
| rank-400 exponential-S, current gradient | 20k | -37.8328788567 | 1.2093673585 | 1.000 |
| rank-1600 WSSR, eta=0.3 | 20k | -37.8437625652 | 0.0637433596 | 18.972 |
| SPRING replicate 2 | 20k | -37.8447209257 | 0.0086429275 | 139.926 |
| shared KFAC-pre1000 starting point | 1k | -37.8119962704 | 1.9856857394 | 0.609 |

The candidate improves over the KFAC-pre1000 starting point but is a strong
negative relative to either optimized 20k comparator. The negative result is
not caused only by rare tails: its 0.1%-winsorized variance is 22.85 times the
corresponding rank-1600 WSSR value (`0.0247817835 Ha^2`).

The clipped training variance also worsens as the S-history weight approaches
0.95: the last-100-step variance is 0.2913 at epoch 1000, 0.3222 at epoch
5000, 0.3577 at epoch 10000, 0.3583 at epoch 15000, and 0.3460 at epoch
20000. This rejects the tested combination of rank 400, long exponential S
history, and an unaveraged current-gradient RHS.
