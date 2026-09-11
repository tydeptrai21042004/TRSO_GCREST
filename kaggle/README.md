# Kaggle execution workflows

For the reviewer revision, use two public GitHub-clone workflows:

- `reviewer_matrix/`: 68 principal experiment groups × seeds 0/1/2 = **204 main-table runs** across 18 sessions.
- `reviewer_sensitivity/`: reviewer-requested reliability, calibration, mode-rule, R-scale and matched-budget controls = **180 runs** across 14 sessions.

Both workflows clone `https://github.com/tydeptrai21042004/TRSO_GCREST.git` by default, support `TRSO_GITHUB_COMMIT` pinning, write source provenance, and enforce resumable runtime guards. The older `TRSO_Revision_Session_18-21` files are retained only for backward compatibility; do not use them for the final reviewer run because they bundle too many runs into one session.

# Active Kaggle protocol — reviewer-ready 204-run matrix

The active Kaggle cells are now under **`kaggle/reviewer_matrix/`**. They implement the complete requested matrix (68 dataset/backbone/method experiment groups × seeds 0/1/2 = **204 training runs**) and split it into **18 time-balanced sessions**. Each session is ordered fastest → slowest, records per-run ETA and actual time, stops safely before **11h50m**, and always produces a ZIP containing all completed runs.

Use `kaggle/reviewer_matrix/README.md` and the files `TRSO_Reviewer_Matrix_Session_01_OneCell.py` through `TRSO_Reviewer_Matrix_Session_18_OneCell.py`. The older files below remain only for submitted-manuscript/revision history.

---

> **Historical protocol notice (September 2026):** this file documents the frozen submitted-manuscript/46-run reproduction. It is preserved for reproducibility, but the active reviewer-ready baseline comparison uses `tools.run_fair_suite` with fresh heads and `tools.run_paper_fair_pairs` for paper-recipe paired checks.

# Kaggle six-session comparison runner

The corrected comparison contains **46 training runs** and is split into six independent Kaggle sessions. Run the files in numeric order; each session is self-contained and produces its own ZIP.

| Session | Scope | Reference controls | Literature baselines | TRSO | Ablations | Total |
|---:|---|---:|---:|---:|---:|---:|
| 1 | DTD / ResNet-50 | 2 | 3 | 1 | 0 | 6 |
| 2 | DTD / ViT-B/16 | 2 | 7 | 1 | 0 | 10 |
| 3 | DTD / ResNet-18 + Swin-T | 4 | 2 | 2 | 4 | 12 |
| 4 | Flowers-102 / ResNet-18 + Swin-T | 4 | 0 | 2 | 0 | 6 |
| 5 | Oxford-IIIT Pet classification / ResNet-18 + Swin-T | 4 | 0 | 2 | 0 | 6 |
| 6 | VOC2007 multilabel + Oxford-Pet segmentation | 4 | 0 | 2 | 0 | 6 |
| **Total** |  | **20** | **12** | **10** | **4** | **46** |

## GitHub source

Every training session clones the tested G-CREST implementation directly from:

```text
https://github.com/tydeptrai21042004/TRSO_GCREST.git
```

Kaggle requirements: enable **Internet** and a **GPU accelerator**. No repository ZIP or Kaggle Dataset upload is required. The default ref is `main`; optionally set `TRSO_GITHUB_REF` or `TRSO_GITHUB_COMMIT` in the notebook environment to pin a branch, tag, or exact commit. Each result archive records the resolved Git commit in `run_summary.json`.

## Training files

1. `TRSO_Paper_46_Session_01_OneCell.py`
2. `TRSO_Paper_46_Session_02_OneCell.py`
3. `TRSO_Paper_46_Session_03_OneCell.py`
4. `TRSO_Paper_46_Session_04_OneCell.py`
5. `TRSO_Paper_46_Session_05_OneCell.py`
6. `TRSO_Paper_46_Session_06_OneCell.py`

After downloading the six session ZIPs, add them to one Kaggle notebook and run `TRSO_Merge_6_Session_Results.py` to create `all_46_results.csv` and a merged archive. The merge utility performs no training and is not a seventh experiment session.

## Correct reference-control policy

- **Full Fine-Tuning (`full`)**: every backbone and task-head parameter is trainable; learning rate `1e-4`.
- **Linear Probing (`linear`)**: the pretrained backbone is frozen and only the task head is trainable; learning rate `1e-3`.
- In every session, reference controls execute first. The exact best linear-probe checkpoint initializes compatible literature baselines and TRSO runs.
- Session 3 initializes all four TRSO ablations from the same ResNet-18 linear-probe checkpoint. It never warm-starts an ablation from a trained full-TRSO checkpoint.

Fixed settings are seed `0`, split seed `0`, 30 epochs, 3 warm-up epochs, AdamW, cosine scheduling, shared strong augmentation, and input size 224. ViT-B/16 uses batch size 8 to keep full fine-tuning safe on common Kaggle GPUs.

The Kaggle protocol still excludes FacT-TT, FacT-TK, VQT, SPT-LoRA, SPT-Adapter, LoRA, BitFit, Side-Tuning, Norm-only, Bias-only, Last-block, and ConvPass-Attn.


## Revision Sessions 07-14

Sessions 01-06 above are the frozen submitted-manuscript reproduction. The following files are additional reviewer/revision evidence and do not alter the canonical 46-run protocol:

| Session | Purpose |
|---:|---|
| 07 | DTD / ResNet-50 / 3-seed Full + Linear + TRSO |
| 08 | Flowers-102 / ViT-B/16 / 3-seed Full + Linear + RepAdapter + TRSO |
| 09 | DTD / ResNet-18 reliability and full-core ablation |
| 10 | DTD / ResNet-18 automatic mode-count rule ablation |
| 11 | DTD / ResNet-18 calibration + loader-batch sensitivity |
| 12 | EuroSAT cross-domain generalization |
| 13 | PCAM medical-domain generalization |
| 14 | FGVC-Aircraft fine-grained generalization |

The shared definitions live in `tools/extended_paper_protocol.py`; each `TRSO_Revision_Session_XX_OneCell.py` produces its own result ZIP.
