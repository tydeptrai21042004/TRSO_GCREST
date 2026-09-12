# Reviewer-required sensitivity scheduler (compact v2)

This is the **new recommended scheduler** for reviewer sensitivity experiments.
It complements `kaggle/reviewer_matrix`; it does not replace the 204-run main matrix.

## Why this folder exists

The older `kaggle/reviewer_sensitivity` plan contains 180 runs. It is valid but
conservative and duplicates several default Proposal conditions already present
in the 204-run main matrix. This folder removes those duplicate default runs and
keeps only new evidence needed to answer the reviewers.

## New training work

- 39 logical groups
- **105 new training runs**
- 9 runtime-balanced Kaggle sessions
- planned session ceiling: <= 480 T4 minutes

| Study | New runs | Scope |
|---|---:|---|
| Reliability components | 27 | Proposal only; 3 settings; default/full reused from main matrix |
| Calibration amount + partition | 18 | Proposal only; CNN + Transformer; 100% alternating reused |
| Calibration batch size | 9 | Proposal only; DTD/ResNet-18; default batch 32 reused |
| D0/D1 rule alternatives | 9 | Proposal only; DTD/ResNet-18; geometric reused |
| R scaling | 6 | Proposal only; DTD/ResNet-18; 1x reused |
| LoRA validation rank sweep | 36 | Baseline control; DTD/ViT-B16 + Flowers/ViT-B16 |
| **Total** | **105** | |

The main `reviewer_matrix` remains responsible for three-seed mean±SD main
results for Proposal and all paper baselines.

## Reviewer mapping

- R2-C2: reliability ablation on DTD/R18, DTD/R50, Flowers/ViT.
- R1-C3 + R2-C1/C5: calibration amount, partition, and batching sensitivity.
- R2-C3 + R3-C1/C2/C3: alternative D0/D1 rules and R stress testing.
- R1-C5 + R2-C3 + R3-C4: matched-budget / validation-oracle LoRA comparison.

## Run on Kaggle

1. Add this folder to the repository and push it to GitHub.
2. For archival runs, set `TRSO_GITHUB_COMMIT=<exact SHA>`.
3. Run `TRSO_Reviewer_Required_Session_01_OneCell.py` through Session 09 with GPU + Internet enabled.
4. Download each `trso_reviewer_required_session_XX_results.zip`.
5. Attach all 9 ZIPs to a merge notebook.
6. Paste/run `TRSO_Reviewer_Required_Merge_Results.py`.

## Notebook-safe merger

The new merger intentionally **does not use `__file__`** and does not import a
local protocol module. It reads the exact expected job IDs from each archived
`session_protocol.json`, so it works when pasted directly into Kaggle/Jupyter.
It also chooses the most complete ZIP if multiple resume ZIPs exist.

## Important reuse rule

Do not treat reused main-matrix results as missing sensitivity runs. The compact
plan intentionally reuses:

- full/default Proposal seeds 0/1/2;
- calibration fraction 100% + alternating partition;
- default DTD/R18 calibration batch size 32;
- geometric mode-count rule;
- R scale 1.0.

This avoids spending compute on identical training configurations.
