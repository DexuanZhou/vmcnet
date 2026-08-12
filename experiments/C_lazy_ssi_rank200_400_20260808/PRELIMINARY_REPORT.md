# C low-rank fixed-basis Lazy SSI: preliminary result

Unit/regression job `53748765` completed with 3 tests passing.  Steady timing
excludes the first 20 of 200 updates.  Online variance is the median of the
last 20 training updates and is used only as an instability screen, not as a
replacement for frozen evaluation.

| arm | mean s/step | ratio to SPRING | last-20 online variance | variance ratio to SPRING | decision |
|---|---:|---:|---:|---:|---|
| SPRING | 0.034452 | 1.000x | 0.118836 | 1.000x | calibration |
| rank200, K=1 | 0.053499 | 1.553x | 0.828241 | 6.970x | reject |
| rank200, K=10 | 0.033732 | 0.979x | 7.126564 | 59.970x | reject: unstable |
| rank200, K=20 | 0.032588 | 0.946x | 8.821060 | 74.229x | reject: unstable |
| rank400, K=1 | 0.072629 | 2.108x | 0.299873 | 2.523x | reject: too slow |
| rank400, K=10 | 0.042796 | 1.242x | 3.880227 | 32.652x | reject: slow and unstable |

The rank400, K=20 arm was resubmitted as job `53749545_6` after the first
node stalled during JAX initialization.  It is not expected to alter the
training decision: K=10 is already slower than SPRING and has a 32.7x online
variance penalty, while both rank200 arms show that increasing K worsens the
staleness instability.  No E5000 run has been admitted by the screen.

The result separates engineering and scientific effects cleanly.  A true
fixed-U lazy path can cross the SPRING timing line at rank200, so the intended
speed mechanism is real.  However, the reused low-dimensional correction
space becomes stale within 10--20 parameter updates; the full residual detects
the conflict but cannot correct it outside the frozen span.  This causes a
large variance increase long before an E5000 endpoint.
