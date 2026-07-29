# Baseline expansion guide

> **Status note:** This is an expansion/research guide, not the active strict baseline list. FacT, VQT, SPT, paper-internal ablations, transferred controls, and engineering controls are excluded from the main literature-baseline table. See `BASELINE_AND_ABLATION_SEPARATION.md`.


## Baselines implemented now

| CLI method | Purpose | Broad task support |
|---|---|---|
| `full` | Full fine-tuning upper reference | all tasks |
| `linear` | Head/decoder-only frozen-backbone baseline | all tasks |
| `norm` | Normalization affine tuning plus task head | all tasks |
| `bias` | Bias-only tuning plus task head | all tasks |
| `last_block` | Final stage/block plus task head | all tasks |
| `trso` | Active proposal | classification, multi-label, regression, segmentation, depth |
| `ssf`, `adaptformer`, `repadapter` | strict visual paper baselines | single-label recognition on their exact paper architectures |
| `lora`, `bitfit` | opt-in cross-domain controls | vision transfer only; never labeled original-paper reproduction |
| `sidetune` | opt-in paper-inspired transferred control | generic classifier wrapper; not an exact original-paper route |
| `vpt_shallow`, `vpt_deep` | paper VPT token prompts | plain ViT recognition |
| `convpass`, `convpass_attn` | paper convolutional ViT bypasses | plain ViT recognition |
| `fact_tt`, `fact_tk` | paper tensorized factor tuning | plain ViT recognition |
| `vqt` | paper query-only intermediate-feature aggregation | plain ViT recognition |
| `spt_lora`, `spt_adapter` | paper sensitivity-aware allocation | plain ViT recognition |

Do not count aliases as separate paper rows. The old image-border `prompt` baseline is not VPT.

See `PAPER_BASELINE_REPRODUCTION.md` for equations, settings, scope, and validation.

## Highest-priority additions remaining

1. **E²VPT** for key/value prompting and prompt pruning.
2. **DA-VPT** for a faithful recognition-and-segmentation prompt route.
3. **ViT-Adapter** with a genuine plain-ViT dense-prediction stack.
4. **Dense Conv-Adapter** for CNN/FPN segmentation and detection.
5. **DETR-family baselines** before adding LoRA or prompt methods to detection.

## Fidelity rule

Every literature baseline must have one method name, one cited paper, explicit supported architectures/tasks, paper-faithful operators and defaults, and tests that reject unsupported combinations. Approximate cross-architecture reuse must not be reported as the original paper method.
