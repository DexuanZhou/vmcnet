#!/usr/bin/env python3
"""Collect the three independently gated history-refresh audits."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/scratch/dexuan1/runs/C_wssr_rank1600_history_refresh_audit_20260802")
OUT = ROOT / "summary.json"
if OUT.exists():
    raise FileExistsError(f"refusing to overwrite {OUT}")

AUDITS = (
    Path(
        "/scratch/dexuan1/runs/"
        "C_wssr_rank1600_history_refresh_audit_20260802_retry1/"
        "legacy_epoch100000/audit.json"
    ),
    Path(
        "/scratch/dexuan1/runs/"
        "C_wssr_rank1600_history_refresh_audit_20260802_retry1/"
        "fixedlambda_epoch101000/audit.json"
    ),
    ROOT / "fixedlambda_epoch102000" / "audit.json",
)

rows = []
for path in AUDITS:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows.append(
        {
            "label": path.parent.name,
            "source_checkpoint": payload["source_checkpoint"],
            "replicates": payload["replicates"],
            "staleness_gate": payload["staleness_gate"],
            "fixed_theta_multibatch_gate": payload[
                "fixed_theta_multibatch_gate"
            ],
            "summaries": payload["summaries"],
            "elapsed_seconds": payload["elapsed_seconds"],
        }
    )
if len(rows) != 3:
    raise RuntimeError(f"expected 3 complete audits, found {len(rows)}")

stale_count = sum(row["staleness_gate"]["passed"] for row in rows)
multibatch_count = sum(
    row["fixed_theta_multibatch_gate"]["passed"] for row in rows
)
payload = {
    "experiment": "C rank1600 history refresh audit collection",
    "checkpoint_count": len(rows),
    "staleness_gate_pass_count": stale_count,
    "fixed_theta_multibatch_gate_pass_count": multibatch_count,
    "cross_checkpoint_stale_history_supported": stale_count >= 2,
    "cross_checkpoint_fixed_theta_multibatch_supported": multibatch_count >= 2,
    "short_training_authorized": stale_count >= 2 and multibatch_count >= 2,
    "decision_rule": (
        "Require both pre-registered gates at at least two of three checkpoints "
        "before any short training."
    ),
    "per_checkpoint": rows,
}
OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
