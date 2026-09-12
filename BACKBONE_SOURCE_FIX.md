# Reviewer Scheduler Backbone Source Fix

## Root cause
The reviewer-sensitivity protocol stores explicit provenance tokens such as
`resnet18@torchvision` and `vit_b_16@torchvision`. `tools/run_reviewer_revision.py`
previously forwarded the whole token as `--backbone` while leaving
`--model_source auto`. `main.py` therefore failed torchvision lookup and then
fell through to timm, where the qualified token is not a valid timm model name.

## Fix
- Normalize `backbone@source` into separate `backbone` and `model_source` fields
  in `tools/run_reviewer_revision.py`.
- Reject conflicting explicit sources.
- Add a reviewer-session preflight guard that rejects unresolved qualified
  backbone names before dataset/model downloads.
- Extend scheduler validation to assert all qualified protocol tokens resolve
  to the expected source.

## Verified
- Required scheduler validation: 39 groups / 105 runs / 9 sessions.
- Reviewer revision unit tests: 5 passed.
- torchvision constructor smoke tests: `resnet18` and `vit_b_16` both load.
