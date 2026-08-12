#!/bin/bash
set -euo pipefail
cd /scratch/dexuan1/vmcnet
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
python -m pytest -q tests/units/updates/test_wssr.py \
  -k 'current_subspace or cluster_adaptive or default_config_contains_wssr_warm_svd_right'
