"""Collect paired-coordinate extreme-event precision audits."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(
    "/scratch/dexuan1/runs/"
    "C_wssr_rank1600_fixed_lambda_stage4_extreme_precision_retry2"
)
LABELS = ("legacy_epoch102000", "fixed_lambda_1e-3_epoch102000")


def main() -> None:
    summary_path = ROOT / "summary.json"
    report_path = ROOT / "report.md"
    if summary_path.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite collector output in {ROOT}")
    runs = {}
    for label in LABELS:
        data = json.loads((ROOT / label / "summary.json").read_text(encoding="utf-8"))
        if not data["finite"] or not data["read_only"]:
            raise ValueError(f"invalid audit {label}")
        runs[label] = data
    payload = {
        "experiment": "C paired-coordinate extreme-event fp32/fp64 audit",
        "read_only": True,
        "runs": runs,
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# C extreme local-energy paired fp32/fp64 audit",
        "",
        "| checkpoint | sampled variance | sampled range | max |ΔE(fp64-fp32)| | fp32 >3 Ha resolved in fp64 |",
        "|---|---:|---|---:|---:|",
    ]
    for label in LABELS:
        data = runs[label]
        sampling = data["sampling_protocol"]
        paired = data["paired_precision"]
        resolved = paired["event_thresholds"]["3.0"][
            "fp32_events_resolved_in_fp64"
        ]
        lines.append(
            f"| {label} | {sampling['sample_raw_variance']:.8f} | "
            f"[{sampling['sample_minimum']:.4f}, {sampling['sample_maximum']:.4f}] | "
            f"{paired['absolute_local_energy_precision_delta']['max']:.6g} | "
            f"{resolved} |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
