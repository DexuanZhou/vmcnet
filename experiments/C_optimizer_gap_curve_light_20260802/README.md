# C optimizer gap-curve audit

This experiment performs frozen inference only. It evaluates the shared
KFAC-pre1000 checkpoint and matched SPRING-replicate2/WSSR-rank1600 checkpoints
at epochs 5000, 20000, 50000 and 100000 under one lightweight protocol.

No parameter training, optimizer-state mutation, or checkpoint write is
performed. The result determines the checkpoint used for the next offline
mechanism audit.
