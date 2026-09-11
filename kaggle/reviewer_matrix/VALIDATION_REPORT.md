# Validation report

- Full repository test suite: **184 passed**.
- Requested matrix: **68/68 experiment groups** present.
- Seeds: **0, 1, 2** for every group → **204 training runs**.
- Dry-run manifest validation: **11/11 dataset/backbone settings** generated exactly the requested methods.
- Session split: **18 sessions**, 7.45–7.95 estimated T4 GPU-hours each.
- All session groups are ordered fastest → slowest.
- Hard one-cell budget: **11h50m**, including clone/install time; **12 min** reserved for ZIP creation.
- Timeout handling kills the full child process group before archiving.
- Result archives are resumable and preserve completed seeds, logs, manifests, protocol, commit, GPU name, ETAs and actual runtimes.
- Every experiment is explicitly expanded to **seeds 0, 1, 2**; seed completeness is exported and machine-checked.
- The session ZIP is refreshed after every fully completed three-seed experiment group and again when the cell exits.
- Every JSON report is retained in `all_reported_metrics.json`; scalar final/report metrics are flattened to `all_final_metrics.csv`; every epoch/history field is exported to `all_epoch_metrics.csv`.

Runtime estimates are planning values only, not benchmark claims.
