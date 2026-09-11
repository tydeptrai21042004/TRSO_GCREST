# Publication-safe support matrix

Only the routes below are eligible for the revised paper's main comparison
matrix.  Unsupported combinations fail before training rather than silently
falling back to a different method.

| CLI method | Architecture | Task | Paper role |
|---|---|---|---|
| `prompt` | supported ResNet/ResNeXt source classifier | single-label | Visual Prompting |
| `conv` | ResNet-50 Bottleneck | single-label | Conv-Adapter |
| `piggyback` | supported CNN route (revision: ResNet-50) | single-label | Piggyback |
| `ssf` | ViT, Swin, ConvNeXt | single-label | SSF |
| `adaptformer` | plain ViT | single-label | AdaptFormer |
| `repadapter` | ViT-B/16 | single-label | RepAdapter |
| `arc` | plain ViT | single-label | ARC |
| `vpt_shallow` | plain ViT | single-label | VPT-Shallow |
| `vpt_deep` | plain ViT | single-label | VPT-Deep |
| `convpass` | plain ViT | single-label | ConvPass |
| `fact_tt` | plain ViT | single-label | FacT-TT |
| `fact_tk` | plain ViT | single-label | FacT-TK |
| `vqt` | plain ViT | single-label | VQT |
| `spt_lora` | plain ViT | single-label | SPT-LoRA |
| `spt_adapter` | plain ViT | single-label | SPT-Adapter |
| `ml_decoder` | MobileNetV3-Small | multilabel | ML-Decoder task-specific published baseline |
| `segadapter` | LR-ASPP/MobileNetV3-Large | semantic segmentation | SegAdapter task-specific published baseline |

Reference controls (`full`, `linear`), the proposal (`trso`), reviewer controls
(`lora`), engineering controls, and paper-internal ablations are intentionally
outside this main-baseline table.

## Canonical revised main matrix

Run `python -m tools.revision_full_protocol` to print the complete 11-setting
matrix.  The executable protocol is frozen to seeds `0,1,2` and 204 total main
training runs.  It uses the manuscript batch sizes and exact common optimizer
settings instead of generic runner defaults.
