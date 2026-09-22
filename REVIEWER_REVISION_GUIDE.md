# Reviewer Revision Experiment Guide

This revision keeps the **default proposal unchanged**:

- mode-count rule: `geometric`, i.e. `R = ceil(sqrt(D0 * D1))`
- `r_scale = 1.0`
- no fixed `R`
- full calibration amount
- deterministic alternating partitions
- full reliability weighting
- full spectral core

All new switches below exist only to answer reviewer sensitivity, stability, and fairness questions.

## What changed in the code

### Reviewer 1 / Reviewer 2: multi-seed statistical reliability

Use `tools/run_reviewer_revision.py --study multiseed` with at least seeds `0,1,2`.
After runs finish:

```bash
python tools/aggregate_revision_results.py \
  --root outputs_reviewer_revision \
  --out_csv revision_summary.csv
```

The aggregator now emits:

- raw per-run CSV;
- mean/std CSV;
- 95% CI CSV;
- allocation-stability CSV;
- compact paper metrics CSV.

Allocation stability includes mean/std/CI for `D0`, `D1`, `R`, selected tensors,
core coordinates, and rank statistics. It also computes pairwise Jaccard overlap
of the participating tensor sets and layer-rank variability.

Example:

```bash
python tools/run_reviewer_revision.py \
  --study multiseed \
  --dataset flowers102 --backbone vit_b_16 \
  --seeds 0,1,2 --epochs 30 --batch_size 16 \
  --download True --execute
```

Repeat for the principal paper settings requested by the reviewers.

## Reviewer 1: calibration-size and partition sensitivity

```bash
python tools/run_reviewer_revision.py \
  --study calibration \
  --dataset dtd --backbone resnet18 \
  --calibration_fractions 0.25,0.5,1.0 \
  --partition_seeds 0,1,2 \
  --seeds 0,1,2 --download True --execute
```

The calibration JSON now reports:

- calibration batch count `B` and fold counts;
- calibration fraction and optional batch cap;
- `D0`, `D1`, and selected `R`;
- total candidate modes;
- candidate-mode count for every tensor;
- participating tensor names;
- every layer rank;
- minimum/median/maximum selected rank;
- trainable spectral-core coordinates;
- frozen basis values.

`--trso_partition_mode seeded_random` changes only the deterministic assignment
of calibration batches to the two consistency partitions. It does **not** use
validation or test data.

## Reviewer 2 / Reviewer 3: stress-test the D0/D1 rule

Alternative global rules are reviewer ablations only:

```bash
python tools/run_reviewer_revision.py \
  --study mode_rules \
  --dataset dtd --backbone resnet18 \
  --mode_rules shannon,harmonic,geometric,arithmetic \
  --seeds 0,1,2 --download True --execute
```

Definitions:

- `shannon`: `R = ceil(D1)`
- `harmonic`: `R = ceil(2 D0 D1 / (D0 + D1))`
- `geometric`: `R = ceil(sqrt(D0 D1))` (**proposal**)
- `arithmetic`: `R = ceil((D0 + D1)/2)`

The code also supports the requested R sensitivity:

```bash
python tools/run_reviewer_revision.py \
  --study r_scale \
  --dataset dtd --backbone resnet18 \
  --r_scales 0.5,1.0,2.0 \
  --seeds 0,1,2 --download True --execute
```

For an explicit fixed-R ablation, call `main.py` with `--trso_fixed_r <N>`.

## Reviewer 2: broader reliability ablations

The existing reliability variants can now be repeated under the exact same
multi-seed protocol:

```bash
python tools/run_reviewer_revision.py \
  --study reliability \
  --dataset flowers102 --backbone vit_b_16 \
  --seeds 0,1,2 --download True --execute
```

It plans:

- full proposal;
- no partition-consistency weighting (`no_crossfit`, retained as a legacy CLI name);
- no sampling-variance term;
- diagonal-only spectral core.

For the revised paper, describe `no_crossfit` as **without partition-consistency
weighting**. The source keeps the old token only for checkpoint/CLI compatibility.

### Optional tensor-level allocation stability

The core calibration report already exports `layer_ranks`,
`participating_tensor_names`, `partition_mode`, and `partition_seed`. Therefore no
method/training-code change is required to compute tensor-level diagnostics from
completed runs. Use:

```bash
python tools/aggregate_revision_results.py \
  --root <completed_output_root> \
  --out_csv revision_summary.csv
```

The aggregator writes the ordinary `*_allocation_stability.csv` across repeated
run seeds and, when multiple `seeded_random` partition seeds are present, a
separate `*_partition_stability.csv`. Tensor-level diagnostics include selected-
tensor Jaccard overlap, tensor-rank Spearman correlation, and pairwise mean
absolute rank difference. These diagnostics are optional and should only be
reported when the completed per-run JSON outputs are actually available. Their
presence in the code does not imply that the manuscript performed them.

## Reviewer 1 / 2 / 3: matched-budget low-rank comparison

A validation rank sweep is provided for LoRA. This is preferable to inventing an
unfaithful AdaLoRA/GoRA implementation. Run it on a backbone supported by the
repository's LoRA implementation (e.g. ViT-B/16):

```bash
python tools/run_reviewer_revision.py \
  --study lora_rank_sweep \
  --dataset dtd --backbone vit_b_16 \
  --lora_ranks 1,2,4,8,16,32,64 \
  --seeds 0,1,2 --download True --execute
```

Then select (a) the nearest parameter-budget run and (b) the best rank chosen
**on validation only**:

```bash
python tools/matched_budget_report.py \
  --proposal_dir <completed_proposal_run> \
  --baseline_root outputs_reviewer_revision \
  --out_csv matched_budget_lora.csv
```

Do not choose a rank by test accuracy. The helper intentionally selects the
oracle rank from validation metrics and reports test performance only after
selection.

If the revision requires an exact published AdaLoRA/GoRA/LoRA-GA baseline,
use the authors' official implementation or a separately fidelity-audited
implementation rather than relabeling the LoRA sweep as one of those methods.

## Run-environment reproducibility information now exported automatically

Each run writes `environment.json` containing:

- Python version;
- PyTorch version;
- torchvision/timm version when available;
- CUDA and cuDNN versions;
- GPU model and compute capability.

`mdl_tangent_calibration.json` now records randomized-SVD controls:

- oversampling (`0` means the documented deterministic automatic rule);
- power iterations;
- SVD seed.

Small matrices still use exact compact SVD; large matrices use the deterministic
randomized range finder.

## Important implementation correction for CNN calibration

BatchNorm is now kept in **evaluation mode during calibration**, so the spectral
basis is estimated under the same frozen-backbone BatchNorm behavior used for
adaptation/deployment. The previous code temporarily used calibration-batch
statistics and then restored the buffers; that could create a calibration/train
graph mismatch on CNN backbones.

## Minimum revision matrix recommended

1. Multi-seed proposal: Flowers-102/ViT-B16, DTD/ResNet-50, DTD/ResNet-18, and one weak setting.
2. Reliability ablation: DTD/ResNet-18 + Flowers-102/ViT-B16 + DTD/ViT-B16.
3. Mode-count rule: at least DTD/ResNet-18 and Flowers-102/ViT-B16.
4. R scaling: at least one CNN and one Transformer setting.
5. Calibration sensitivity: 25/50/100% on at least one CNN and one Transformer setting.
6. LoRA rank sweep / matched budget: at least DTD/ViT-B16 and Flowers-102/ViT-B16.

This set directly addresses the experimental requests without redefining the
paper's proposed default method.

---

## Canonical full main-table rerun (final revision)

The submitted 46-run reproduction is intentionally preserved as historical evidence. The revised
controlled main tables are generated from one source of truth and use **fresh task heads**
(`head_init_policy=random`, head LR scale `1.0`) with source-audited structural defaults:

```bash
python -m tools.revision_full_protocol
```

This prints and audits the **204 main training runs** across the 11 manuscript
settings.  Every method row is rerun with seeds `0,1,2`.  To execute one setting:

```bash
python -m tools.revision_full_protocol \
  --setting flowers_vit_b16 \
  --data_path ./data --download auto --execute
```

To execute all main settings:

```bash
python -m tools.revision_full_protocol \
  --data_path ./data --download auto --execute
```

The audit copies are written to:

- `experiments/revision_full_protocol.json`
- `experiments/revision_full_protocol.csv`

The revised ViT-B/16 tables include SSF, AdaptFormer, RepAdapter, ARC,
VPT-Shallow, VPT-Deep, ConvPass, FacT-TT, FacT-TK, VQT, SPT-LoRA and
SPT-Adapter.  VOC2007 uses ML-Decoder as its task-specific published baseline;
Pet trimap segmentation uses SegAdapter.

### Paper-recipe paired robustness check

The controlled table does **not** claim that one AdamW schedule reproduces every
baseline paper. To test optimizer/protocol sensitivity, use:

```bash
python -m tools.run_paper_fair_pairs \
  --dataset dtd --task single_label --data_path ./data --download auto \
  --backbone vit_b_16@torchvision \
  --baselines adaptformer,vpt_deep,vqt \
  --seeds 0,1,2 --search_mode full \
  --manifest experiments/dtd_vit_paper_pairs.json --execute

python -m tools.verify_paper_pairs \
  --manifest experiments/dtd_vit_paper_pairs.json

python -m tools.select_paper_pair_trials \
  --manifest experiments/dtd_vit_paper_pairs.json
```

For every source recipe/HPO trial, TRSO receives the same optimizer, LR, scheduler,
epoch/batch budget, augmentation, split, seed and fresh-head policy. Exact source
settings that are not encoded must be supplied through `--overrides_json` or the
method is transparently labelled a matched fallback. See `PAPER_BASELINE_REPRODUCTION.md`.

## Isolated calibration-batch-size study

`--study batch_sensitivity` now changes **only**
`--trso_calibration_batch_size`.  The optimization/training batch size remains
fixed to the manuscript value.  All requested seeds are used, so this study can
also be summarized as mean ± SD.

Example:

```bash
python tools/run_reviewer_revision.py \
  --study batch_sensitivity \
  --dataset flowers102 --backbone vit_b_16 \
  --batch_size 16 --batch_sizes 4,8,16,32 \
  --seeds 0,1,2 --download auto --execute
```

## Small-sample confidence intervals

Main tables should report **mean ± SD**.  When 95% confidence intervals are
exported, `tools/aggregate_revision_results.py` now uses a Student-t critical
value for the available number of seeds (for `n=3`, `t_{0.975,2}`), rather than
the large-sample 1.96 approximation.
