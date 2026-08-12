#!/bin/bash
set -euo pipefail
GPU_HEALTH_FILE=${1:?health output path required}
mkdir -p "$(dirname "${GPU_HEALTH_FILE}")"
if ! nvidia-smi > "${GPU_HEALTH_FILE}" 2>&1; then
  echo 'GPU_HEALTH_FAIL: nvidia-smi failed' >&2
  exit 90
fi
cat "${GPU_HEALTH_FILE}"
if grep -qi 'GPU requires reset' "${GPU_HEALTH_FILE}"; then
  echo 'GPU_HEALTH_FAIL: allocated GPU requires reset' >&2
  exit 91
fi
python - <<'PY'
import jax
devices = [d for d in jax.devices() if d.platform in ('cuda', 'gpu')]
print('JAX CUDA devices:', devices)
if not devices:
    raise SystemExit('GPU_HEALTH_FAIL: JAX sees no CUDA device')
PY
