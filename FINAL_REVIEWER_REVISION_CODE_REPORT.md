# Final reviewer-revision code report

## Main experiment contract

The canonical revised main suite is `tools/revision_full_protocol.py`.
It contains 11 manuscript settings, seeds `0,1,2`, and 204 training runs.
The submitted 46-run protocol remains preserved as historical reproduction. The active controlled revision uses fresh task heads; paper-recipe sensitivity is handled by `tools/run_paper_fair_pairs.py`.

| Setting | Methods |
|---|---|
| DTD / ResNet-50 | Full, Linear, Visual Prompting, Conv-Adapter, Piggyback, Proposal |
| DTD / ViT-B/16 | Full, Linear, SSF, AdaptFormer, RepAdapter, ARC, VPT-Shallow, VPT-Deep, ConvPass, FacT-TT, FacT-TK, VQT, SPT-LoRA, SPT-Adapter, Proposal |
| DTD / ResNet-18 | Full, Linear, Visual Prompting, Proposal |
| DTD / Swin-T | Full, Linear, SSF, Proposal |
| Flowers-102 / ResNet-18 | Full, Linear, Visual Prompting, Proposal |
| Flowers-102 / ViT-B/16 | Full, Linear, SSF, AdaptFormer, RepAdapter, ARC, VPT-Shallow, VPT-Deep, ConvPass, FacT-TT, FacT-TK, VQT, SPT-LoRA, SPT-Adapter, Proposal |
| Flowers-102 / Swin-T | Full, Linear, SSF, Proposal |
| Oxford-IIIT Pet / ResNet-18 | Full, Linear, Visual Prompting, Proposal |
| Oxford-IIIT Pet / Swin-T | Full, Linear, SSF, Proposal |
| VOC2007 / MobileNetV3-S | Full, Linear, ML-Decoder, Proposal |
| Pet trimap / LR-ASPP MobileNetV3-L | Full, Linear, SegAdapter, Proposal |

## New / corrected baseline support

- ML-Decoder: task-specific published multilabel baseline; spatial MobileNet features are passed to the non-ZSL fixed-query/grouped-classifier decoder.
- SegAdapter: paper-equation reimplementation with SegAttention, HSA, FFN, learnable stage-wise scaled residual, four stage blocks, and auxiliary CE with lambda=0.4.
- FacT-TT / FacT-TK, VQT, SPT-LoRA / SPT-Adapter are in the paper-baseline registry for the plain-ViT routes used by the revision matrix.
- Local engineering/transferred controls are not paper-baseline rows.

## Baseline fairness correction

- `baseline_recipes.py` is the canonical source for baseline structural defaults and training-recipe provenance.
- AdaptFormer now uses the official image default bottleneck `64` in paper-structure comparisons.
- `tools.run_fair_suite` defaults to `head_init_policy=random` and `peft_head_lr_scale=1.0`; the former hidden LP warm start is an explicit ablation only.
- `tools.run_paper_fair_pairs` generates baseline/TRSO twins under each recoverable source recipe/HPO trial.
- `tools.verify_paper_pairs` verifies pair equality; `tools.select_paper_pair_trials` selects HPO by validation only.
- Partial/transferred recipes are labelled as such instead of being called exact paper reproductions.

## Reviewer fixes

- main results: seeds 0/1/2 for every method row;
- allocation diagnostics already export D0, D1, R, selected tensors, per-layer ranks and core coordinates;
- calibration amount/partition studies remain separate from the proposal default;
- calibration batch-size sensitivity now changes only the calibration loader, not the training batch size;
- reliability ablations cover CNN and Transformer settings;
- D0/D1 alternative mode-count rules and 0.5x/1x/2x R are reviewer-only ablations;
- LoRA rank sweep remains a reviewer-only matched-budget/oracle control;
- 95% CI export uses Student-t for small seed counts; main tables should report mean +/- SD.

## Validation

- full unit/integration test suite: **182 passed**;
- controlled-fair manifest audit: fresh-head baseline/TRSO manifest passes with zero errors;
- paper-recipe paired manifest audit: baseline/TRSO outer-recipe equality passes with zero errors;
- Adam + constant-scheduler one-epoch smoke: passed;
- `main.py` integration smoke: ML-Decoder produces `B x 20` VOC logits;
- `main.py` integration smoke: SegAdapter produces a 3-class main map plus the training-only auxiliary map;
- canonical revision protocol dry-run confirms **204** main training runs.

## Commands

Preview all main runs:

```bash
python -m tools.revision_full_protocol
```

Run one setting:

```bash
python -m tools.revision_full_protocol \
  --setting flowers_vit_b16 \
  --data_path ./data --download auto --execute
```

Run all main settings:

```bash
python -m tools.revision_full_protocol \
  --data_path ./data --download auto --execute
```

Reviewer studies are defined in `tools/run_reviewer_revision.py` and revision sessions 18-21 in `tools/extended_paper_protocol.py`.
