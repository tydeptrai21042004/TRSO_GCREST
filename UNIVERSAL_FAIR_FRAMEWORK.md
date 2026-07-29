# Universal fair experiment framework

`tools.run_fair_suite` schedules all scientifically compatible method/backbone/task
pairs and records every rejected pair explicitly. For each backbone and seed, it
first trains one linear probe and reuses its best task head for compatible PEFT
methods, including G-CREST-TRSO.

## Controlled settings

All PEFT rows share optimizer, PEFT learning rate, cosine schedule, warm-up,
weight decay, epoch count, batch size, augmentation, data split, input size, and
validation checkpoint rule. Full fine-tuning and linear probing may use separate
learning rates.

## Example

```bash
python -m tools.run_fair_suite \
  --dataset dtd \
  --download True \
  --backbones resnet50@torchvision,vit_tiny_patch16_224@timm \
  --methods auto \
  --seeds 0,1,2 \
  --epochs 30 \
  --input_size 0 \
  --execute
```

Check the generated manifest:

```bash
python -m tools.verify_fairness \
  --manifest experiments/fair_manifest.json \
  --compatibility experiments/fair_manifest_compatibility.json
```

Aggregate completed results:

```bash
python -m tools.aggregate_revision_results \
  --root outputs_fair \
  --out_csv outputs_fair/all_results.csv
```

The framework does not force every baseline onto every architecture. Conv-Adapter
remains ResNet-50-specific, while AdaptFormer and RepAdapter retain their ViT-specific routes; invalid
pairs are skipped rather than silently approximated.
