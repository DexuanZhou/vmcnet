"""Collect strict-multilevel smoke-test invariants into one JSON report."""

import json
from pathlib import Path

import jax
import numpy as np

from vmcnet.utils import io


ROOT = Path("/scratch/dexuan1/runs/C_ferminet_multilevel_smoke_20260808")


def _parameter_count(params):
    return int(sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(params)))


def _norm(value):
    leaves = jax.tree_util.tree_leaves(value)
    return float(np.sqrt(sum(np.sum(np.asarray(leaf, dtype=np.float64) ** 2) for leaf in leaves)))


def _optax_state(state):
    return state.optax_state if hasattr(state, "optax_state") else state


def _schedule_count(state):
    for part in _optax_state(state):
        if hasattr(part, "_fields") and "count" in part._fields:
            return int(np.asarray(part.count))
    raise ValueError("optimizer state has no schedule count")


def _history_norms(method, state):
    if method == "spring":
        return {"trace": _norm(_optax_state(state)[0].trace)}
    result = {"sr_o": _norm(state.core_state.sr_o)}
    if hasattr(state.core_state, "u"):
        result["u"] = _norm(state.core_state.u)
    if hasattr(state, "solution_state"):
        result["solution_state"] = _norm(state.solution_state)
    return result


def main():
    summary = {}
    for method in ("spring", "wssr_warm_svd_right"):
        root = ROOT / method
        coarse = io.reload_vmc_state(str(root / "coarse/checkpoints"), "2.npz")
        converted = io.reload_vmc_state(str(root / "converted"), "multilevel.npz")
        continued = io.reload_vmc_state(
            str(root / "fine_continue/checkpoints"), "12.npz"
        )
        conversion_report = json.loads(
            (root / "converted/multilevel_conversion_report.json").read_text()
        )
        coarse_history = _history_norms(method, coarse[3])
        converted_history = _history_norms(method, converted[3])
        summary[method] = {
            "conversion": conversion_report,
            "coarse_epoch": int(coarse[0]),
            "converted_epoch": int(converted[0]),
            "continued_epoch": int(continued[0]),
            "coarse_parameter_count": _parameter_count(coarse[2]),
            "converted_parameter_count": _parameter_count(converted[2]),
            "coarse_schedule_count": _schedule_count(coarse[3]),
            "converted_schedule_count": _schedule_count(converted[3]),
            "continued_schedule_count": _schedule_count(continued[3]),
            "coarse_history_norms": coarse_history,
            "converted_history_norms": converted_history,
            "history_norm_relative_errors": {
                key: abs(converted_history[key] - value) / max(abs(value), 1e-30)
                for key, value in coarse_history.items()
            },
            "fine_metric_rows": sum(
                1 for _ in open(root / "fine_continue/training_metrics.csv")
            )
            - 1,
        }
    output = ROOT / "smoke_summary.json"
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
