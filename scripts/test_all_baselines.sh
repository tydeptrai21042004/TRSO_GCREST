#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

python -m pytest -q -p no:cacheprovider \
  tests/test_baseline_modules.py \
  tests/test_baseline_integration.py \
  tests/test_corrected_baseline_fidelity.py
