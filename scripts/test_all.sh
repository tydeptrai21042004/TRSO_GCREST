#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
python -m pytest -q -p no:cacheprovider
python -m tools.audit_universal_release --output test_reports/universal_release_audit.json
