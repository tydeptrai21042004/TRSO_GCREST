# Publication-safe support matrix

| CLI method | Strict architecture | Strict task | Paper role |
|---|---|---|---|
| `prompt` | supported ResNet/ResNeXt with unchanged source classifier | single-label | Visual Prompting main method |
| `conv` | ResNet-50 Bottleneck | single-label | Conv-Adapter main method |
| `ssf` | ViT, Swin, or ConvNeXt | single-label | SSF main method |
| `adaptformer` | plain ViT | single-label | AdaptFormer main method |
| `repadapter` | ViT-B/16 | single-label | RepAdapter main method |
| `arc` | plain ViT | single-label | ARC main method |
| `piggyback` | ResNet-50 or VGG-16 | single-label | Piggyback main method |
| `vpt_shallow` | plain ViT | single-label | VPT-Shallow primary variant |
| `vpt_deep` | plain ViT | single-label | VPT-Deep primary variant |
| `convpass` | plain ViT | single-label | ConvPass main method |

ARC does not silently transfer to CNN or hierarchical Transformer routes. The current certified implementation rejects non-plain-ViT architectures. All controls, candidates, proposal runs, and proposal ablations remain in separate categories.
