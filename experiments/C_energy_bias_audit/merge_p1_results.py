#!/usr/bin/env python3
"""Merge independent-process float32/float64 P1 audit results."""

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--float32", type=Path, required=True)
parser.add_argument("--float64", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

f32 = json.loads(args.float32.read_text())
f64 = json.loads(args.float64.read_text())
assert f32["dtype"] == "float32"
assert f64["dtype"] == "float64"
assert f32["point_generation"] == f64["point_generation"]

result = {
    "test_protocol": "independent JAX process per dtype",
    "point_generation": f32["point_generation"],
    "backends": {"float32": f32["jax_backend"], "float64": f64["jax_backend"]},
    "jax_versions": {"float32": f32["jax_version"], "float64": f64["jax_version"]},
    "results": {},
}
for z in ("Z=1", "Z=2"):
    result["results"][z] = {
        "float32": f32["results"][z],
        "float64": f64["results"][z],
    }

args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
