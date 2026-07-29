# Recommended dense-prediction baseline: HST

## Recommendation

Add **Hierarchical Side-Tuning (HST)** as the next optional external paper
baseline for dense prediction.

HST is a better scientific match than adding another classification-only PEFT
method because its original paper evaluates a frozen plain ViT with a trainable
hierarchical side network on:

- COCO object detection and instance segmentation;
- ADE20K semantic segmentation;
- ViT-B and ViT-L backbones.

## Strict scope

The faithful project route should be:

| Task | Backbone | Framework/head | Dataset |
|---|---|---|---|
| Object detection / instance segmentation | ViT-B | Mask R-CNN or ATSS | COCO |
| Semantic segmentation | ViT-B | UPerNet or Semantic FPN | ADE20K |

Do not label an Oxford-Pet + lightweight-MobileNet rewrite as an original-paper
HST reproduction. Such a run may be useful as an engineering transfer, but it
must be reported separately.

## Why HST is outside the 46-run six-session protocol

The six-session protocol intentionally uses mostly small datasets and 30 epochs. HST's
paper-faithful dense routes rely on COCO/ADE20K and standard dense-prediction
schedules, so adding it would dominate the compute budget and destroy the
"minimal" character of the experiment set.

## Implementation policy

HST is currently a **recommended planned baseline**, not an executable method in
`--methods auto`. Integrate the official implementation with a pinned commit,
retain its ViT-B/HSN-L architecture and decoder configuration, and add a
separate capability contract before enabling it in reported comparisons.
