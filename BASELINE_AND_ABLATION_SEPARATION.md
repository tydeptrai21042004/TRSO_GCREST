# Baseline, reference-control, proposal, and ablation separation

This file is the **active revision taxonomy**.  The original 46-run Kaggle
protocol is preserved only as a historical submitted-manuscript reproduction;
it is not the recommended final reviewer comparison.

## Published-method baseline set

`--methods auto` in `tools.run_fair_suite` resolves to the published-method
implementations below, subject to explicit architecture/task guards.

| Method ID | Report name | Active route | Paper-training status |
|---|---|---|---|
| `prompt` | Visual Prompting | frozen source classifier + image prompt/mapping | source-audited partial |
| `conv` | Conv-Adapter | ResNet-50 Bottleneck | source-audited partial |
| `piggyback` | Piggyback | CNN binary masks + task head | source-audited partial |
| `ssf` | SSF | ViT/Swin/ConvNeXt placements | official scripts are dataset/backbone specific |
| `adaptformer` | AdaptFormer | parallel ViT FFN adapter, **d=64** | official image default encoded |
| `repadapter` | RepAdapter | ViT-B/16 + exact folding | paper search/config required |
| `arc` | ARC | ViT shared projection banks + re-composition | official config required |
| `vpt_shallow` | VPT-Shallow | shallow prompt tokens | official LR/WD search encoded; prompt length dataset specific |
| `vpt_deep` | VPT-Deep | deep prompt tokens | official LR/WD search encoded; prompt length dataset specific |
| `convpass` | ConvPass | attention + MLP convolutional bypass | official config required |
| `fact_tt` | FacT-TT | tensor factor tuning, paper rank | official config required |
| `fact_tk` | FacT-TK | tensor factor tuning, paper rank | official config required |
| `vqt` | VQT | query tokens + intermediate features | Adam/query-length source constraints verified; task config required |
| `spt_lora` | SPT-LoRA | sensitivity stage + LoRA | official config required |
| `spt_adapter` | SPT-Adapter | sensitivity stage + Adapter | official config required |
| `ml_decoder` | ML-Decoder | public decoder-head mechanism | revision route is a transferred backbone/task evaluation |
| `segadapter` | SegAdapter | paper-equation segmentation adapter | revision route is a transferred backbone evaluation |

The machine-readable source of truth is `baseline_recipes.py`.  A baseline is
never called *paper-exact* merely because its mechanism is implemented.
Mechanism fidelity, architecture fidelity, and training-recipe fidelity are
tracked separately.

## Separate comparison groups

| Group | Methods | Reporting rule |
|---|---|---|
| Reference controls | `full`, `linear` | separate Reference Controls rows |
| Proposal | `trso` | proposal result only |
| Proposal ablations | `diagonal_only`, `no_sampling_variance`, `no_crossfit`, `head_only` | separate ablation table |
| Paper-internal ablation | `convpass_attn` | appendix only |
| Transferred controls | `lora`, `bitfit`, `sidetune` | not literature-baseline rows |
| Engineering controls | `norm`, `bias`, `last_block` | diagnostics only |

## Two fair protocols

### 1. Controlled main-table protocol

`tools.run_fair_suite` keeps the same outer optimization/data protocol across
PEFT baselines and TRSO, while each baseline keeps the source-audited structural
defaults from `baseline_recipes.py`.

The default head policy is now **fresh/random**:

- no hidden linear-probe checkpoint is loaded;
- PEFT head LR scale is `1.0`;
- Full, baselines, and TRSO all start from their normal task-head initialization;
- Visual Prompting retains the source classifier because that is part of the
  published mechanism.

The former linear-probe warm start remains available only as the explicit
`--head_init_policy linear_probe` ablation and must not be mixed into the main
random-head table.

### 2. Paper-recipe paired protocol

`tools.run_paper_fair_pairs` tests robustness to each baseline author's
optimizer recipe.  For every baseline source recipe/HPO trial, it generates a
matched TRSO run with the **same**:

- data split and seed;
- backbone/weights/input resolution;
- batch and epoch budget;
- optimizer, LR, scheduler, warm-up and decay;
- augmentation;
- fresh-head policy.

Only the adaptation mechanism differs.  When an official method performs an
LR/WD search (e.g. VPT), the identical candidate grid is given to both the
baseline and TRSO.  `tools/select_paper_pair_trials` selects each method's
configuration by validation score only and reads test scores afterwards.

Use `tools.verify_paper_pairs` to mechanically verify pair equality.
