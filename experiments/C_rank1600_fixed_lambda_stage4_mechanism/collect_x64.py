"""Collect the four same-coordinate fixed-lambda counterfactual audits."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(
    "/scratch/dexuan1/runs/"
    "C_wssr_rank1600_fixed_lambda_stage4_x64_counterfactual"
)
FILES = {
    "101500": {
        "legacy_checkpoint": ROOT / "legacy_epoch101500.json",
        "fixed_checkpoint": ROOT / "fixed_lambda_1e-3_epoch101500.json",
    },
    "102000": {
        "legacy_checkpoint": ROOT / "legacy_epoch102000.json",
        "fixed_checkpoint": ROOT / "fixed_lambda_1e-3_epoch102000.json",
    },
}


def select(data: dict[str, object]) -> dict[str, object]:
    if not data["finite"] or data["update_applied"] or data["walkers_advanced"]:
        raise ValueError(f"invalid read-only audit: {data['source_checkpoint']}")
    return {
        "source_checkpoint": data["source_checkpoint"],
        "legacy_effective_lambda": data["legacy_effective_lambda"],
        "fixed_tikhonov_lambda": data["fixed_tikhonov_lambda"],
        "leading_singular_value": data["leading_singular_value"],
        "full_retained_rank": data["full_retained_rank"],
        "approx_active_rank": data["approx_active_rank"],
        "ssi_x64_subspace_min_cosine": data["ssi_x64_subspace_min_cosine"],
        "ssi_legacy_vs_x64_rank1600": data["ssi_legacy_vs_x64_rank1600"],
        "ssi_fixed_vs_x64_rank1600": data["ssi_fixed_vs_x64_rank1600"],
        "ssi_fixed_vs_legacy": data["ssi_fixed_vs_legacy"],
        "x64_rank1600_fixed_vs_legacy": data["x64_rank1600_fixed_vs_legacy"],
        "x64_full_fixed_vs_legacy": data["x64_full_fixed_vs_legacy"],
        "spectral_band_update_mass": data["spectral_band_update_mass"],
        "force_decomposition": data["force_decomposition"],
        "force_dot_update": data["force_dot_update"],
        "current_residuals": {
            "ssi_legacy": data["ssi_legacy_current_residual"],
            "ssi_fixed_lambda_1e-3": data["ssi_fixed_current_residual"],
            "x64_rank1600_legacy": data["x64_rank1600_legacy_current_residual"],
            "x64_rank1600_fixed_lambda_1e-3": data[
                "x64_rank1600_fixed_current_residual"
            ],
        },
    }


def main() -> None:
    summary_path = ROOT / "summary.json"
    report_path = ROOT / "report.md"
    if summary_path.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite collector output in {ROOT}")
    results = {}
    for epoch, branches in FILES.items():
        results[epoch] = {}
        for branch, path in branches.items():
            results[epoch][branch] = select(
                json.loads(path.read_text(encoding="utf-8"))
            )
    payload = {
        "experiment": "C rank1600 fixed-lambda x64 counterfactual update audit",
        "read_only": True,
        "results": results,
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# C rank1600 fixed-lambda x64 counterfactual update audit",
        "",
        "No updates were applied and no walkers were advanced.",
        "",
        "| epoch | checkpoint | legacy effective lambda | exact fixed/legacy cosine | SSI fixed exact error | SSI legacy exact error |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for epoch in sorted(results):
        for branch, row in results[epoch].items():
            lines.append(
                f"| {epoch} | {branch} | {row['legacy_effective_lambda']:.8g} | "
                f"{row['x64_rank1600_fixed_vs_legacy']['cosine']:.6f} | "
                f"{row['ssi_fixed_vs_x64_rank1600']['relative_error']:.6f} | "
                f"{row['ssi_legacy_vs_x64_rank1600']['relative_error']:.6f} |"
            )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
