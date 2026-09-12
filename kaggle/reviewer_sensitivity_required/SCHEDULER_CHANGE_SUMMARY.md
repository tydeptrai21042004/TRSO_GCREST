# Scheduler change summary

## Problem fixed

The previous merge helper used `Path(__file__)`. Kaggle/Jupyter does not define
`__file__` when the source is pasted directly into a notebook cell, producing:

`NameError: name '__file__' is not defined`.

The new merger is notebook-safe: it has no `__file__` dependency and no local
protocol import. It derives exact expected job IDs from each archived
`session_protocol.json`.

## Schedule reduction

Old sensitivity schedule: 72 groups / 180 runs / 14 sessions.

New required-only schedule: 39 groups / 105 new runs / 9 sessions.

The reduction comes from reusing identical default Proposal runs already in the
204-run `kaggle/reviewer_matrix`:

- full/default Proposal at seeds 0/1/2;
- 100% alternating calibration;
- default DTD/ResNet-18 calibration batch size 32;
- geometric D0/D1 mode-count rule;
- R scale 1.0.

## New runs retained

- Reliability components: 27 Proposal-only runs.
- Calibration amount/partition: 18 Proposal-only runs.
- Calibration batch: 9 Proposal-only runs.
- D0/D1 alternatives: 9 Proposal-only runs.
- R scaling: 6 Proposal-only runs.
- LoRA rank sweep: 36 baseline-control runs.

## Additional merge robustness

- validates protocol identifier and session number;
- validates exactly 105 expected jobs;
- rejects duplicate expected job IDs;
- validates one Git commit across sessions;
- safely extracts ZIP members;
- if multiple resume ZIPs exist, picks the ZIP with the most completed jobs,
  then newest timestamp;
- writes `merge_validation.json`, `session_summary.csv`, and
  `all_run_status.csv`.
