# Paper-method and reference-control implementation status

The strict main-table literature methods are Visual Prompting, Conv-Adapter, SSF, AdaptFormer, RepAdapter, ARC, Piggyback, VPT-Shallow, VPT-Deep, and ConvPass. `--methods auto` selects only these methods, subject to architecture compatibility.

## ARC

ARC follows the main NeurIPS 2023 Adapter Re-Composing configuration: separate cross-layer shared projection matrices for MHA and FFN, symmetric transpose up-projections, layer-specific rescaling and bias, sequential insertion before MHA/FFN, official initialization/default dropout, frozen backbone, and exact evaluation folding. See `ARC_BASELINE_IMPLEMENTATION.md`.

## Reference controls included in comparison

- `full`: corrected Full Fine-Tuning; every parameter is trainable.
- `linear`: corrected Linear Probing; the pretrained backbone is frozen and only the task head is trainable.

These are displayed under a separate Reference Controls heading and never counted as dedicated-paper literature baselines.

## Excluded mechanisms

`convpass_attn` is a paper-internal ablation. FacT, VQT, and SPT variants are development candidates rather than strict rows. LoRA, BitFit, Side-Tuning, Norm-only, Bias-only, and Last-block remain excluded from the six-session Kaggle comparison. TRSO structural variants are generated only by the separate proposal-ablation runner inside Session 3.
