# Baseline, Reference-Control, Proposal, and Ablation Separation

## Strict literature baseline table

| Method ID | Report name | Active strict route |
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

`--methods auto` resolves only to these ten method IDs. Architecture guards determine which methods are valid for each concrete backbone.

## Separate comparison groups

| Group | Methods | Reporting rule |
|---|---|---|
| Reference controls | `full`, `linear` | included in comparison tables under a separate Reference Controls heading |
| Proposal | `trso` | proposal result group only |
| Proposal ablations | `diagonal_only`, `no_sampling_variance`, `no_crossfit`, `head_only` | separate TRSO ablation table |
| Paper-internal ablation | `convpass_attn` | optional appendix only; excluded from the 46-run protocol |
| Qualified reproduction candidates | `fact_tt`, `fact_tk`, `vqt`, `spt_lora`, `spt_adapter` | excluded until exact certification/rewrite |
| Transferred controls | `lora`, `bitfit`, `sidetune` | excluded from the 46-run protocol |
| Engineering controls | `norm`, `bias`, `last_block` | excluded from the 46-run protocol |

## Corrected 46-run allocation

| Group | Runs |
|---|---:|
| Strict literature baselines | 12 |
| Reference controls: Full + Linear | 20 |
| Full TRSO proposal | 10 |
| TRSO-only ablations | 4 |
| **Total** | **46** |

## Head initialization and ablation fairness

Within each of the six Kaggle sessions, the matching Linear Probing run executes first. Its best checkpoint supplies the identical task-head initialization for compatible strict baselines and TRSO. Full Fine-Tuning is trained and reported independently. TRSO ablations use the same linear head but never reuse the trained full-TRSO checkpoint.
