# SPRING versus MinSR after KFAC-5000

Both members of each system pair reload the identical full-state 4096-walker
KFAC checkpoint and create a new SPRING optimizer state.  The pair differs only
in `mu` (0 versus 0.99).  The numerical path is the original float32 baseline.

The 200-epoch array is followed by an `afterany` collector.  Only tasks that
complete 200/200 with finite metrics and no monitor threshold are released as
fresh 50,000-epoch jobs from the original KFAC-5000 checkpoint.
