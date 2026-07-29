# Final Baseline and Reference-Control Recheck with ARC

## Publication rule

A main-table literature baseline must be a method or named primary variant from a dedicated vision paper, implemented on a paper-compatible architecture and task. Full Fine-Tuning and Linear Probing are corrected standard reference controls and are reported separately. TRSO and its ablations remain separate from both groups.

## Strict literature baselines

| ID | Report name | Active strict route |
|---|---|---|
| `prompt` | Visual Prompting | ResNet route with padding prompt and frequency output mapping |
| `conv` | Conv-Adapter | ResNet-50 Bottleneck route |
| `ssf` | SSF | published ViT/Swin/ConvNeXt placements |
| `adaptformer` | AdaptFormer | parallel FFN branch on plain ViT |
| `repadapter` | RepAdapter | ViT-B/16 route with exact folding |
| `arc` | Adapter Re-Composing (ARC) | plain ViT; shared attention/FFN projections and layer-specific re-composition |
| `piggyback` | Piggyback | corrected ResNet-50/VGG-16 masking route |
| `vpt_shallow` | VPT-Shallow | named shallow prompt variant on plain ViT |
| `vpt_deep` | VPT-Deep | named deep prompt variant on plain ViT |
| `convpass` | ConvPass | complete attention-and-MLP bypass route |

## Corrected reference controls

| ID | Trainability rule | Learning rate |
|---|---|---:|
| `full` | all backbone and task-head parameters train | 1e-4 |
| `linear` | frozen pretrained backbone; task head only | 1e-3 |

Both controls use the same dataset split, seed, epochs, augmentation, optimizer family, scheduler, input size, and evaluation pipeline as the corresponding TRSO comparison. Linear checkpoints are reused only to standardize task-head initialization; they remain separately reported runs.

## Experiment allocation

| Result group | Runs |
|---|---:|
| Strict literature baselines | 12 |
| Full + Linear reference controls | 20 |
| Full TRSO proposal | 10 |
| TRSO-only ablations | 4 |
| **Total** | **46** |

The workload is divided across six self-contained Kaggle sessions with totals 6, 10, 12, 6, 6, and 6 runs.
