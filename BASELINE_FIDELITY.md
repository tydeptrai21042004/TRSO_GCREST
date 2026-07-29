# Baseline fidelity status

## Certified strict literature baselines

| Method | Fidelity status |
|---|---|
| Visual Prompting (`prompt`) | implemented paper route: padding prompt plus frequency mapping |
| Conv-Adapter (`conv`) | ResNet-50 route |
| SSF (`ssf`) | implemented ViT/Swin/ConvNeXt route |
| AdaptFormer (`adaptformer`) | plain-ViT route |
| RepAdapter (`repadapter`) | ViT-B/16 route and exact merge |
| ARC (`arc`) | main plain-ViT ARC route with shared attention/FFN banks, official initialization/defaults, and exact merge |
| Piggyback (`piggyback`) | corrected ResNet-50/VGG-16 route |
| VPT-Shallow (`vpt_shallow`) | named paper variant |
| VPT-Deep (`vpt_deep`) | named paper variant |
| ConvPass (`convpass`) | full paper method |

## Excluded from the strict table

- `convpass_attn`: baseline-paper internal ablation.
- `fact_tt`, `fact_tk`, `vqt`: qualified reproduction candidates.
- `spt_lora`, `spt_adapter`: official allocation rewrite required.
- `lora`, `bitfit`, `sidetune`: transferred/non-exact controls.
- `norm`, `bias`, `last_block`: engineering controls.
- `full`, `linear`: reference controls.
- `trso`: proposal.
