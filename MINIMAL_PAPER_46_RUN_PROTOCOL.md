> **Historical protocol notice (September 2026):** this file documents the frozen submitted-manuscript/46-run reproduction. It is preserved for reproducibility, but the active reviewer-ready baseline comparison uses `tools.run_fair_suite` with fresh heads and `tools.run_paper_fair_pairs` for paper-recipe paired checks.

# Minimal comparison protocol: 46 training runs in six Kaggle sessions

| Result group | Runs |
|---|---:|
| Strict literature baselines | 12 |
| Full Fine-Tuning + Linear Probing reference controls | 20 |
| Full TRSO proposal | 10 |
| TRSO-only structural ablations | 4 |
| **Total** | **46** |

## Six-session split

| Session | Training content | Runs |
|---:|---|---:|
| 1 | DTD / ResNet-50: Full, Linear, Visual Prompting, Conv-Adapter, Piggyback, TRSO | 6 |
| 2 | DTD / ViT-B/16: Full, Linear, SSF, AdaptFormer, RepAdapter, ARC, VPT-Shallow, VPT-Deep, ConvPass, TRSO | 10 |
| 3 | DTD / ResNet-18 and Swin-T: Full, Linear, two literature baselines, two TRSO runs, four TRSO ablations | 12 |
| 4 | Flowers-102 / ResNet-18 and Swin-T: Full, Linear, TRSO | 6 |
| 5 | Oxford-IIIT Pet classification / ResNet-18 and Swin-T: Full, Linear, TRSO | 6 |
| 6 | VOC2007 multilabel and Oxford-Pet semantic segmentation: Full, Linear, TRSO | 6 |

## Fixed settings

- Seed: 0
- Split seed: 0
- Epochs: 30
- Warm-up: 3 epochs
- Input: 224 x 224
- Optimizer: AdamW
- Scheduler: cosine
- Augmentation: basic
- PEFT/TRSO learning rate: 1e-3
- Full Fine-Tuning learning rate: 1e-4
- Linear Probing learning rate: 1e-3

## Comparison policy

Full Fine-Tuning and Linear Probing are reference controls, not strict literature baselines. Every session trains its own matching linear probe first and reuses that exact best task-head checkpoint for compatible PEFT/TRSO comparisons. This keeps initialization fair while retaining separate result groups.

One seed provides fixed-seed evidence only. Do not claim statistical significance or report mean plus/minus standard deviation from this protocol.
