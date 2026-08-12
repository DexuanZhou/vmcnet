#!/bin/bash
set -euo pipefail
cd /scratch/dexuan1/vmcnet
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
python -m pytest -q tests/units/updates/test_wssr.py \
  -k 'anisotropic_matrix_history or anisotropic_matrix_augmentation'
