# Proposal removal manifest

The following proposal implementations are intentionally absent:

- `models/task_response_unified.py`
- `models/task_response_biaxial.py`
- `models/task_response_adapter.py`
- `models/tuning_modules/tangent_core.py`
- `models/tuning_modules/information_spectral.py`
- `KAGGLE_TRSO_V5_V6_ONE_CELL.py`
- `KAGGLE_INFORMATION_SPECTRAL_ONE_CELL.py`

Removed proposal controls include rank, maximum rank, parameter budget,
calibration-batch count target, gain, manually selected layers,
core-mode choice, and task-head policy.

The replacement is:

```text
models/tuning_modules/mdl_tangent_core.py
```
