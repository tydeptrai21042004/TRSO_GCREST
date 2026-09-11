# Active Kaggle reviewer matrix (204 training runs)

This is the active Kaggle execution plan for the reviewer-ready main comparison. It contains **68 dataset/backbone/method experiments × seeds 0,1,2 = 204 independent training runs**. The outer protocol is controlled and fair: same split, 30 epochs, fresh task heads, AdamW/cosine policy, and source-audited method-internal defaults from `baseline_recipes.py`.

## Safety and runtime behavior

- Sessions are balanced to about **7.45–7.95 h on an NVIDIA T4 planning estimate**, leaving substantial margin below a 12 h Kaggle session.
- Inside each session, experiment groups are sorted **fastest → slowest**. Each group runs seeds `0`, `1`, `2` one at a time.
- The runner detects the GPU, rescales the ETA, records actual wall-clock time, and updates its estimate from completed runs.
- A hard watchdog uses **11 h 50 min** as the absolute session budget. It reserves **12 minutes** for archiving and will not launch a new run if the conservative projected finish would cross that boundary.
- If a run reaches the watchdog timeout, it is stopped, all already completed runs/logs/manifests are preserved, and a session ZIP is produced.
- After every experiment group finishes all **three seeds (0,1,2)**, the current session ZIP is refreshed immediately. The final ZIP is refreshed once more when the cell exits.
- Previous partial session ZIPs can be attached to a later Kaggle notebook and are automatically resumed/skipped at completed run granularity.
- Checkpoints are removed from successful runs by default to keep result ZIPs manageable. Set `TRSO_KEEP_CHECKPOINTS=1` if you explicitly want them retained.

## Session plan

| Session | Experiment groups | Seed-runs | T4 estimate | Fastest → slowest groups |
|---:|---:|---:|---:|---|
| 01 | 3 | 9 | 7.80 h | 8m: DTD / ResNet-18 — Linear<br>28m: DTD / ResNet-18 — Full<br>120m: Pet segmentation / LR-ASPP MobileNetV3-L — Full |
| 02 | 3 | 9 | 7.80 h | 22m: DTD / ResNet-18 — Proposal<br>39m: Flowers / ViT-B16 — ARC<br>95m: DTD / ViT-B16 — Full |
| 03 | 3 | 9 | 7.50 h | 25m: Pet segmentation / LR-ASPP MobileNetV3-L — Linear<br>40m: Flowers / ViT-B16 — VPT-Deep<br>85m: DTD / Swin-T — Full |
| 04 | 3 | 9 | 7.45 h | 24m: Pet / ResNet-18 — Proposal<br>40m: VOC2007 / MobileNetV3-S — Proposal<br>85m: Pet / Swin-T — Full |
| 05 | 4 | 12 | 7.75 h | 7m: Flowers / ResNet-18 — Linear<br>28m: DTD / ResNet-50 — Visual Prompting<br>40m: Flowers / Swin-T — Proposal<br>80m: Pet segmentation / LR-ASPP MobileNetV3-L — Proposal |
| 06 | 4 | 12 | 7.95 h | 12m: VOC2007 / MobileNetV3-S — Linear<br>30m: Flowers / ViT-B16 — VPT-Shallow<br>42m: Flowers / ViT-B16 — ConvPass<br>75m: Flowers / ViT-B16 — Full |
| 07 | 4 | 12 | 7.80 h | 9m: Pet / ResNet-18 — Linear<br>30m: Pet / ResNet-18 — Full<br>42m: Flowers / ViT-B16 — Proposal<br>75m: Pet segmentation / LR-ASPP MobileNetV3-L — SegAdapter |
| 08 | 4 | 12 | 7.95 h | 14m: Flowers / Swin-T — Linear<br>33m: DTD / ResNet-50 — Proposal<br>42m: DTD / ViT-B16 — SSF<br>70m: DTD / ViT-B16 — SPT-Adapter |
| 09 | 4 | 12 | 7.85 h | 12m: DTD / ResNet-50 — Linear<br>34m: Flowers / ViT-B16 — SSF<br>43m: DTD / ViT-B16 — VQT<br>68m: DTD / ViT-B16 — SPT-LoRA |
| 10 | 4 | 12 | 7.85 h | 14m: Flowers / ResNet-18 — Visual Prompting<br>34m: DTD / ResNet-50 — Conv-Adapter<br>44m: DTD / ViT-B16 — FacT-TT<br>65m: Flowers / Swin-T — Full |
| 11 | 4 | 12 | 7.60 h | 15m: Flowers / ViT-B16 — Linear<br>35m: Flowers / ViT-B16 — VQT<br>45m: Pet / Swin-T — SSF<br>57m: Flowers / ViT-B16 — SPT-Adapter |
| 12 | 4 | 12 | 7.70 h | 18m: DTD / ViT-B16 — Linear<br>36m: Flowers / Swin-T — SSF<br>45m: DTD / ViT-B16 — AdaptFormer<br>55m: Flowers / ViT-B16 — SPT-LoRA |
| 13 | 4 | 12 | 7.70 h | 18m: Flowers / ResNet-18 — Proposal<br>36m: Flowers / ViT-B16 — AdaptFormer<br>45m: DTD / ViT-B16 — FacT-TK<br>55m: VOC2007 / MobileNetV3-S — Full |
| 14 | 4 | 12 | 7.75 h | 20m: Pet / ResNet-18 — Visual Prompting<br>37m: Flowers / ViT-B16 — FacT-TK<br>46m: DTD / ViT-B16 — RepAdapter<br>52m: DTD / ResNet-50 — Full |
| 15 | 4 | 12 | 7.80 h | 22m: Flowers / ResNet-18 — Full<br>37m: DTD / ViT-B16 — VPT-Shallow<br>45m: DTD / Swin-T — SSF<br>52m: DTD / ViT-B16 — Proposal |
| 16 | 4 | 12 | 7.65 h | 17m: DTD / Swin-T — Linear<br>37m: Flowers / ViT-B16 — RepAdapter<br>48m: DTD / ViT-B16 — ARC<br>51m: DTD / ViT-B16 — ConvPass |
| 17 | 4 | 12 | 7.65 h | 17m: Pet / Swin-T — Linear<br>36m: Flowers / ViT-B16 — FacT-TT<br>50m: DTD / Swin-T — Proposal<br>50m: VOC2007 / MobileNetV3-S — ML-Decoder |
| 18 | 4 | 12 | 7.65 h | 18m: DTD / ResNet-18 — Visual Prompting<br>36m: DTD / ResNet-50 — Piggyback<br>49m: DTD / ViT-B16 — VPT-Deep<br>50m: Pet / Swin-T — Proposal |

The detailed machine-readable plan is in `reviewer_matrix_plan.csv` and `reviewer_matrix_plan.json`. Runtime values are scheduling estimates, **not measured benchmark claims**. Use the actual times recorded in each result archive for reporting.

## How to run on Kaggle

1. Enable a GPU. Enable Internet if you want the cell to clone GitHub.
2. Open the matching `TRSO_Reviewer_Matrix_Session_XX_OneCell.py` and paste/run it as one cell, or execute the file directly.
3. The bootstrap searches in this order: `TRSO_REPO_DIR` → the local uploaded repo → `TRSO_PROJECT_ZIP`/an attached updated project ZIP → GitHub. This means the corrected project can run even before you push it to GitHub.
4. For strict reproducibility when cloning GitHub, set `TRSO_GITHUB_COMMIT` to the commit you used for all sessions.
5. The cell creates/refreshes `trso_reviewer_matrix_session_XX_results.zip` after each fully completed three-seed experiment group and again at session exit. If the watchdog stops a session early, attach that ZIP to a new Kaggle notebook and run the **same session cell** again; completed run IDs are skipped.
6. After all sessions, attach the result ZIPs and run `TRSO_Reviewer_Matrix_Merge_Results.py`. It verifies all **204 seed-runs = 68 experiments × seeds {0,1,2}**, merges every aggregate metric, and creates `trso_reviewer_matrix_merged.zip`.


## Metrics guaranteed in every session ZIP

The archive keeps the original result directory for every completed seed-run and also materializes aggregate files for reviewer analysis:

- `all_reported_metrics.json`: exact parsed contents of **every JSON report** generated by every completed run.
- `all_final_metrics.csv`: one wide row per seed-run, including final/test metrics, validation summaries, parameter counts, convergence, timing/GPU-hours, memory, efficiency/latency, and method-specific calibration metrics when reported.
- `all_epoch_metrics.csv`: **every field from every epoch** in `history.json`, preserving train/validation learning curves and runtime fields.
- `metrics_inventory.csv`: every collected JSON report with byte size, SHA-256, and parse status.
- `seed_completeness.csv`: explicit status for seeds `0`, `1`, and `2` of every experiment in that session.
- `metrics_coverage.json`: machine-readable proof of how many three-seed experiment groups are complete and whether all required seeds are present.
- `run_status.csv`, `session_state.json`, `session_protocol.json`, `manifests/`, `logs/`, and the original `runs/` directories are retained.

The underlying trainer currently emits reports such as `dataset_protocol.json`, `run_manifest.json`, `args.json`, `environment.json`, `resolved_protocol.json`, `parameter_summary.json`, `initial_validation_summary.json`, `history.json`, `convergence_summary.json`, `test_summary.json`, `timing_summary.json`, and `efficiency_profile*.json`, plus method-specific files including TRSO calibration, Visual Prompting mapping, and SPT sensitivity reports where applicable. Because collection scans every JSON report recursively, newly added metric reports are automatically included without changing the Kaggle exporter.

## Requested experiment matrix

- **DTD / ResNet-50:** Full, Linear, Visual Prompting, Conv-Adapter, Piggyback, Proposal
- **DTD / ViT-B16:** Full, Linear, SSF, AdaptFormer, RepAdapter, ARC, VPT-Shallow, VPT-Deep, ConvPass, FacT-TT, FacT-TK, VQT, SPT-LoRA, SPT-Adapter, Proposal
- **DTD / ResNet-18:** Full, Linear, Visual Prompting, Proposal
- **DTD / Swin-T:** Full, Linear, SSF, Proposal
- **Flowers / ResNet-18:** Full, Linear, Visual Prompting, Proposal
- **Flowers / ViT-B16:** Full, Linear, SSF, AdaptFormer, RepAdapter, ARC, VPT-Shallow, VPT-Deep, ConvPass, FacT-TT, FacT-TK, VQT, SPT-LoRA, SPT-Adapter, Proposal
- **Flowers / Swin-T:** Full, Linear, SSF, Proposal
- **Pet / ResNet-18:** Full, Linear, Visual Prompting, Proposal
- **Pet / Swin-T:** Full, Linear, SSF, Proposal
- **VOC2007 / MobileNetV3-S:** Full, Linear, ML-Decoder, Proposal
- **Pet segmentation / LR-ASPP MobileNetV3-L:** Full, Linear, SegAdapter, Proposal

## Environment controls

- `TRSO_GITHUB_REPO`, `TRSO_GITHUB_REF`, `TRSO_GITHUB_COMMIT`: Git source/pinning.
- `TRSO_PROJECT_ZIP`: explicit path to the corrected project ZIP attached to Kaggle.
- `TRSO_RESUME_ZIP`: explicit prior result ZIP for resuming a partial session.
- `TRSO_KEEP_CHECKPOINTS=1`: keep checkpoint files in result archives.
- `TRSO_STOP_ON_ERROR=1`: stop after the first failed experiment instead of continuing to later experiments.
- `TRSO_NUM_WORKERS`: dataloader workers, default `4`.
- `TRSO_ALLOW_COMMIT_MISMATCH=1`: bypass resume commit consistency checking; not recommended for paper results.

