# Reliability-Weighted Spectral Allocation with Full Spectral Cores for Parameter-Efficient Visual Fine-Tuning

Official research implementation accompanying the manuscript:

**Reliability-Weighted Spectral Allocation with Full Spectral Cores for Parameter-Efficient Visual Fine-Tuning**  
Ba Ty Dang, Kim Huong Tran, Thi Uyen Nguyen  
**Manuscript submitted to _Pattern Analysis and Applications_.**

The internal implementation/CLI name remains **G-CREST-TRSO / `trso`** for backward compatibility with existing checkpoints, scripts, and the historical submitted 46-run protocol. Public documentation follows the manuscript terminology: the method uses **two deterministic calibration partitions for partition-consistency weighting**; their agreement is not statistical cross-fitting or repeated-run reproducibility. New calibration reports use the public method identifier `reliability_weighted_spectral_allocation_full_core` and retain the historical identifier separately as `legacy_method_id` for traceability.

## Active proposal

For every eligible CNN or Transformer weight, the method forms pooled calibration-gradient spectral modes and weights them by consistency between two deterministic calibration partitions. A single model-wide evidence distribution determines

\[
R=\left\lceil\sqrt{D_0D_1}\right\rceil,
\]

then allocates the globally strongest modes to tensors. Each participating tensor learns a full spectral core in \(U_{\ell,S_\ell}K_\ell V_{\ell,S_\ell}^\top\). The default proposal is intentionally unchanged from the manuscript.

The full proposal has:

- no user-specified global rank budget, evidence threshold, or layer list;
- the same allocation rule for evaluated convolutional and Transformer tensors;
- full spectral cores rather than diagonal-only scaling;
- exact algebraic merging, leaving **no active method-specific inference branch** after merging;
- a complete merged backbone checkpoint per downstream task.

See `METHOD_GCREST_TRSO.md` and the manuscript for the formal definition.

### Optional tensor-level allocation stability diagnostics

The proposal itself is unchanged, but completed per-run calibration outputs can be
post-processed with:

```bash
python tools/aggregate_revision_results.py \
  --root <completed_output_root> \
  --out_csv revision_summary.csv
```

When `mdl_tangent_calibration.json` contains `layer_ranks` and participating
tensor names, the aggregator reports selected-tensor Jaccard overlap, tensor-rank
Spearman correlation, pairwise rank MAE, and aggregate allocation variation. For
`seeded_random` calibration-partition studies it also writes a separate
`*_partition_stability.csv` that compares partition seeds while holding the
optimization seed fixed. These are **analysis-only diagnostics**; they do not
change training or the proposed allocation rule. If the original per-run outputs
are unavailable, the relevant proposal runs must be rerun before these optional
diagnostics can be computed.

## Dataset and revision extension

The repository now exposes **54 canonical dataset routes** through a registry-backed interface. `--download auto` downloads only routes marked safe for unattended acquisition; `--download yes` explicitly permits large automatic routes, while manual/terms-sensitive datasets remain manual. Legacy `--download True/False` commands are still accepted.

Preprocessing is centralized and can resolve normalization/interpolation metadata from the pretrained checkpoint (`--preprocess auto`) while keeping all methods on the same backbone preprocessing contract. Optional MedMNIST routes include `pathmnist`, `dermamnist`, `bloodmnist`, `pneumoniamnist`, `organamnist`, and `tissuemnist`.

The active reviewer-ready Kaggle matrix is now in `kaggle/reviewer_matrix/`: **68 experiment groups × 3 seeds = 204 training runs**, split into **18 time-balanced sessions** with fastest-to-slowest ordering, per-run ETA/actual timing, resumable progress, and an 11h50m archive watchdog. Historical Sessions 01-06 and earlier revision cells are retained only for reproducibility.

## Controlled evidence

The included deterministic transfer controls are diagnostics, not real-dataset publication claims.

| Control | Seeds | Reference accuracy | G-CREST accuracy | Mean change |
|---|---:|---:|---:|---:|
| Standard TinyViT | 5 | 97.933% | **99.833%** | **+1.900 pt** |
| Standard TinyCNN | 5 | 100.000% | 100.000% | 0.000 pt |
| Hard TinyCNN | 6 | 92.708% | **92.854%** | **+0.146 pt** |
| Hard TinyCNN, 30 epochs | 1 | 98.125% | **99.000%** | **+0.875 pt** |

G-CREST won all five standard ViT seeds. The hard-CNN six-seed improvement is small and not statistically conclusive; it is reported rather than hidden.

## Proposal ablations

```bash
python -m tools.run_trso_ablation \
  --dataset dtd \
  --data_path /path/to/data \
  --backbone resnet50 \
  --model_source torchvision \
  --seeds 0,1,2 \
  --epochs 30 \
  --execute
```

The analysis-only variants are `diagonal_only`, `no_sampling_variance`, `no_crossfit`, and `head_only`. They are not capacity knobs of the full proposal. `no_crossfit` is a legacy machine/CLI token retained for compatibility; in manuscript terminology it means **without partition-consistency weighting**.

## Historical six Kaggle sessions

The frozen submitted-manuscript protocol contains 46 training runs split into sessions of 6, 10, 12, 6, 6 and 6 runs. It is retained for reproduction, but it is **not** the active reviewer-ready fairness protocol. Every historical session:

- clones `https://github.com/tydeptrai21042004/TRSO_GCREST.git` from `main`;
- verifies the G-CREST release contract;
- uses seed 0, 30 epochs and shared strong augmentation;
- separates strict baselines, reference controls, proposal rows and proposal ablations;
- records the resolved Git commit in `run_summary.json`.

See `kaggle/README.md` and `MINIMAL_PAPER_46_RUN_PROTOCOL.md`.

## Active Kaggle execution

For the complete requested reviewer matrix, use `kaggle/reviewer_matrix/README.md` and `TRSO_Reviewer_Matrix_Session_01_OneCell.py` through `TRSO_Reviewer_Matrix_Session_18_OneCell.py`. The planner covers exactly 204 seed-runs, uses seeds 0/1/2, and keeps the main-table fresh-head fairness policy. Each session is planned for roughly 7.45–7.95 T4 GPU-hours and will stop launching work early enough to create a ZIP before the 11h50m hard budget. Partial session ZIPs are resumable.

## Active reviewer-ready baseline comparison

The active code uses two complementary protocols:

1. **Controlled main table** — `tools.run_fair_suite` uses the same outer optimizer/data recipe across PEFT baselines and TRSO while retaining source-audited method structure from `baseline_recipes.py`. The default is now `--head_init_policy random --peft_head_lr_scale 1.0`; no hidden linear-probe checkpoint is loaded.
2. **Paper-recipe paired sensitivity** — `tools.run_paper_fair_pairs` gives a baseline its recoverable official paper/repository recipe (or source HPO grid) and reruns TRSO with the identical outer recipe and HPO budget. Missing source settings are labelled fallback rather than paper-exact.

Useful audit commands:

```bash
python -m tools.verify_fairness --manifest experiments/fair_manifest.json
python -m tools.verify_paper_pairs --manifest experiments/paper_fair_pairs.json
python -m tools.select_paper_pair_trials --manifest experiments/paper_fair_pairs.json
```

See `PAPER_BASELINE_REPRODUCTION.md`, `BASELINE_FIDELITY.md`, and `BASELINE_AND_ABLATION_SEPARATION.md`.

## Validation

```bash
pytest -q
python -m tools.validate_gcrest_controlled --suite standard --seeds 0,1,2,3,4
python -m tools.clean_release --check --manifest --zip ../TRSO_GCREST_release.zip
```

Repository tests and controlled diagnostics establish implementation correctness. A new real DTD ResNet-50 Session 1 run is required before claiming real-dataset superiority.

## Major-revision reviewer experiments

See [`REVIEWER_REVISION_GUIDE.md`](REVIEWER_REVISION_GUIDE.md) for multi-seed, allocation-stability, calibration-sensitivity, D0/D1-rule, R-scaling, and matched-budget LoRA experiments. The proposal defaults are unchanged; all new controls are ablation/evaluation switches.

## Reviewer revision Kaggle workflow

The final reviewer protocol is split into two runtime-safe workflows: `kaggle/reviewer_matrix` (204 main-table runs) and `kaggle/reviewer_sensitivity` (180 reviewer-requested sensitivity/ablation runs). Both clone the public `TRSO_GCREST` repository and can pin an immutable commit with `TRSO_GITHUB_COMMIT`. Use the generated plan/coverage files in each directory when preparing the point-by-point response.
