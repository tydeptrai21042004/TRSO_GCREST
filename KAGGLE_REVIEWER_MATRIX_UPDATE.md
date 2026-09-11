# Kaggle reviewer-matrix update — 11 September 2026

The active Kaggle execution layer now covers the complete requested main-table matrix:

- 68 dataset/backbone/method experiment groups;
- seeds 0, 1 and 2 for every experiment;
- 204 independent training runs in total;
- controlled fresh-head comparison (`head_init_policy=random`, `peft_head_lr_scale=1.0`);
- source-audited method-internal defaults from `baseline_recipes.py`;
- 18 runtime-balanced Kaggle sessions;
- fastest-to-slowest ordering inside every session;
- T4 planning estimates plus detected-GPU and measured-runtime updates;
- an 11h50m hard session budget with a 12-minute archive reserve;
- per-seed execution so already completed work survives a later timeout/failure;
- resumable partial session ZIPs;
- merged-result utility for checking completion against all 204 expected runs.

Runtime estimates are scheduling estimates only and are not used as scientific performance claims. Each result archive records the actual wall-clock runtime of each completed run.

The active files are under `kaggle/reviewer_matrix/`. Older Kaggle files remain historical reproduction material and are not the recommended reviewer-ready launcher.
