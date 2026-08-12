"""Collect endpoint accuracy and trajectory metadata for the four-arm pilot."""

import csv
import json
from pathlib import Path


ROOT = Path("/scratch/dexuan1/runs/C_ferminet_multilevel_pilot_20260808")
ARMS = (
    ("spring", "direct_L2", "train_L2"),
    ("spring", "multilevel", "train_L2_to2000"),
    ("wssr_warm_svd_right", "direct_L2", "train_L2"),
    ("wssr_warm_svd_right", "multilevel", "train_L2_to2000"),
)


def _metrics(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    last = rows[-1]
    return {
        "rows": len(rows),
        "last_epoch": int(last["epoch"]),
        "last_energy": float(last["energy"]),
        "last_variance": float(last["variance"]),
    }


def main():
    summary = {}
    markdown = [
        "# C strict-FermiNet multilevel pilot",
        "",
        "| optimizer | capacity schedule | frozen energy | frozen variance | "
        "endpoint train energy | endpoint train variance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, mode, final_name in ARMS:
        train_root = ROOT / "pilot" / method / mode
        final_root = train_root / final_name
        frozen_path = ROOT / "frozen" / method / mode / "eval/statistics.json"
        frozen = json.loads(frozen_path.read_text())
        final_metrics = _metrics(final_root / "training_metrics.csv")
        transitions = []
        for report_path in sorted(train_root.glob("converted_*/multilevel_conversion_report.json")):
            transitions.append(json.loads(report_path.read_text()))
        key = f"{method}/{mode}"
        summary[key] = {
            "frozen": frozen,
            "final_training_metrics": final_metrics,
            "conversion_reports": transitions,
        }
        markdown.append(
            f"| {method} | {mode} | {frozen['average']:.9f} | "
            f"{frozen['variance']:.9f} | {final_metrics['last_energy']:.9f} | "
            f"{final_metrics['last_variance']:.9f} |"
        )

    for method in ("spring", "wssr_warm_svd_right"):
        direct = summary[f"{method}/direct_L2"]["frozen"]["variance"]
        multilevel_value = summary[f"{method}/multilevel"]["frozen"]["variance"]
        summary[f"{method}/multilevel_vs_direct_variance_ratio"] = (
            multilevel_value / direct
        )
        markdown.extend(
            (
                "",
                f"- {method}: multilevel/direct frozen-variance ratio = "
                f"{multilevel_value / direct:.6f}.",
            )
        )
    (ROOT / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    (ROOT / "PILOT_RESULTS.md").write_text("\n".join(markdown) + "\n")
    print("\n".join(markdown))


if __name__ == "__main__":
    main()
