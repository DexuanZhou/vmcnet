#!/usr/bin/env python3
"""Static pre-submission validation of the matched three-run protocol."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = (ROOT / "train_array.sh").read_text()
ROWS = list(csv.DictReader((ROOT / "manifest.csv").open()))

assert [int(row["task_id"]) for row in ROWS] == [0, 1, 2]
assert [int(row["rank"]) for row in ROWS] == [400, 400, 800]
assert all(int(row["warm_iterations"]) == 2 for row in ROWS)
assert all(row["exact_first"] == "True" for row in ROWS)
assert ROWS[0]["experimental_mode"] == "none"
assert ROWS[0]["complement_weight"] == "0.0"
assert ROWS[1]["experimental_mode"] == "adaptive_complement"
assert ROWS[1]["adaptive_complement_beta"] == "0.2"
assert ROWS[1]["complement_weight"] == "0.0001"
assert ROWS[2]["experimental_mode"] == "none"
assert ROWS[2]["complement_weight"] == "0.0"

required_tokens = [
    "--reload.new_optimizer_state=True",
    "--reload.reburn=False",
    "--config.problem.nelec='(4,2)'",
    "--config.vmc.nchains=1000",
    "--config.vmc.nepochs=100000",
    "--config.vmc.nsteps_per_param_update=10",
    "--config.vmc.clip_center=mean",
    "--config.vmc.clip_threshold=5.0",
    "--config.eval.nchains=2000",
    "--config.eval.nburn=5000",
    "--config.eval.nepochs=20000",
    "--config.vmc.optimizer_type=wssr_warm_svd_right",
    "--config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.02",
    "--config.vmc.optimizer.wssr_warm_svd_right.learning_decay_rate=0.0001",
    "--config.vmc.optimizer.wssr_warm_svd_right.eta=0.8",
    "--config.vmc.optimizer.wssr_warm_svd_right.damping=0.0003",
    "--config.vmc.optimizer.wssr_warm_svd_right.norm_constraint=0.001",
    "--config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm=2",
    "--config.vmc.optimizer.wssr_warm_svd_right.exact_first=True",
]
missing = [token for token in required_tokens if token not in SCRIPT]
assert not missing, missing
print("PASS: three matched WSSR E100000 configurations validated statically")
