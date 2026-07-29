# ARC baseline implementation report

## Source method

**Adapter Re-Composing (ARC)** follows Wei Dong, Dawei Yan, Zhijun Lin, and Peng Wang, *Efficient Adaptation of Large Vision Transformer via Adapter Re-Composing*, NeurIPS 2023.

## Implemented main configuration

- Two independent cross-layer shared down-projection matrices are registered: one for attention adapters and one for FFN adapters.
- The up-projection is the transpose of the corresponding shared down-projection.
- Every Transformer block has its own low-dimensional re-scaling vector and output bias for each branch.
- ARC is inserted sequentially before MHA and before FFN.
- Attention re-scaling starts at zero; FFN re-scaling uses Xavier-uniform initialization; shared projections use Xavier-uniform initialization; adapter biases start at zero.
- Dropout defaults to `0.1`, and the paper/release bottleneck dimension defaults to `50`.
- The pretrained backbone is frozen. Only ARC parameters and the downstream task head train.
- Evaluation-time folding is exact: ARC is folded into the packed QKV/input attention projections and the first FFN linear projection.

## Scope and guardrails

The certified route accepts only plain timm/torchvision Vision Transformers. CNNs, Swin, DeiT, BEiT, EVA, CaiT, MaxViT, and unrecognized hybrid architectures are rejected rather than silently approximated.

## CLI

```bash
--tuning_method arc \
--arc_dim 50 \
--arc_dropout 0.1 \
--arc_merge True
```

## Validation

The automated tests check registry classification, architecture compatibility, one-time shared-bank registration, layer-specific coefficients, official initialization behavior, trainable-parameter isolation, gradients, non-ViT rejection, fair-runner parameters, and exact/idempotent evaluation folding.
