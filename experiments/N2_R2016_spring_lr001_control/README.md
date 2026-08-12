# N2 R=2.016 SPRING lr=0.001 stability control

This job independently reloads the archived, completed 1000-walker
KFAC-pre5000 checkpoint at R=2.016 Bohr. It restores the checkpoint walkers
and PRNG state, creates a new SPRING optimizer state, does not reburn, and
runs up to 100000 epochs with NaN checking enabled.

This is an explicitly requested comparison run. It does not replace the
current R=2.068 production geometry.
