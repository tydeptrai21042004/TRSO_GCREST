# Corrected original-paper baseline audit

## Final strict baseline set

| Method | Report name | Active paper-compatible route |
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

The strict set contains ten method IDs and twelve architecture-specific runs. Full Fine-Tuning and Linear Probing are restored as twenty corrected reference-control runs but remain outside this strict literature-baseline set.

## Final 46-run split

| Group | Runs |
|---|---:|
| Strict literature baselines | 12 |
| Full + Linear reference controls | 20 |
| Full TRSO proposal | 10 |
| TRSO-only ablations | 4 |
| **Total** | **46** |
