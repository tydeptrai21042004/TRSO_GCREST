# G-CREST repository audit

## Active proposal

The repository contains one active TRSO proposal: G-CREST-TRSO. Its model-wide mode count and layer allocation are derived from a single global evidence distribution. No manual adapter rank, total budget, layer list, rescue route or Linear fallback is active.

## Corrected scientific issues

1. Per-layer independent rank selection was replaced by one model-wide evidence allocation.
2. A user-specified global budget is not required.
3. The loss/readiness gate and dense/sparse rescue routes are absent.
4. Only trained epochs are eligible for proposal checkpoint selection.
5. BatchNorm state and module modes are restored after calibration.
6. Core, head, frozen-basis and deployed parameter counts are reported separately.
7. Learned cores are exactly merged for zero-extra-parameter deployment.

## Validation boundary

The full repository test suite passes. Controlled diagnostics show a clear TinyViT improvement and a smaller hard-CNN mean gain. These diagnostics do not establish DTD or VTAB superiority; the real six-session benchmark must be rerun.
