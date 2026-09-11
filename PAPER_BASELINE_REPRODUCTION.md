# Paper-faithful baseline and fair-comparison protocol

This is the active baseline protocol for the reviewer-ready revision.  The old
46-run Kaggle protocol is retained only for historical submitted-manuscript
reproduction.

## A. Controlled main comparison

Use one outer recipe for all methods while retaining each published method's
source-audited structural defaults:

```bash
python -m tools.run_fair_suite \
  --dataset dtd --task single_label --data_path ./data --download auto \
  --backbones vit_b_16@torchvision \
  --methods full,linear,ssf,adaptformer,repadapter,arc,vpt_shallow,vpt_deep,convpass,fact_tt,fact_tk,vqt,spt_lora,spt_adapter,trso \
  --seeds 0,1,2 \
  --epochs 30 --batch_size 16 \
  --optimizer adamw --peft_lr 1e-3 --full_lr 1e-4 --linear_lr 1e-3 \
  --weight_decay 1e-4 --warmup_epochs 3 \
  --head_init_policy random --peft_head_lr_scale 1.0 \
  --execute
```

This table is a **controlled reimplementation comparison**, not a claim that
one shared AdamW recipe reproduces every baseline paper's reported number.

### Linear-probe warm start is now an ablation

Only run the old warm-start design explicitly:

```bash
python -m tools.run_fair_suite ... \
  --head_init_policy linear_probe \
  --external_head_manifests experiments/reference_linear.json
```

Never merge those rows into the random-head main table.

## B. Paper-recipe paired comparison

For a baseline whose official recipe is encoded, run the baseline and TRSO under
that exact same outer recipe:

```bash
python -m tools.run_paper_fair_pairs \
  --dataset dtd --task single_label --data_path ./data --download auto \
  --backbone vit_b_16@torchvision \
  --baselines adaptformer,vpt_deep,vqt \
  --seeds 0,1,2 \
  --search_mode full \
  --manifest experiments/dtd_vit_paper_pairs.json \
  --execute
```

Then verify pair equality:

```bash
python -m tools.verify_paper_pairs \
  --manifest experiments/dtd_vit_paper_pairs.json
```

And select HPO trials by validation score only:

```bash
python -m tools.select_paper_pair_trials \
  --manifest experiments/dtd_vit_paper_pairs.json
```

### Dataset-specific paper configs

Some official releases use dataset-specific structural hyperparameters (for
example VPT prompt length) or external config files.  Supply those values
without changing the paired TRSO outer recipe:

```bash
python -m tools.run_paper_fair_pairs ... \
  --baselines vpt_deep \
  --method_overrides_json '{"vpt_deep":{"vpt_num_tokens":50}}'
```

For a method whose exact optimizer recipe is recovered from an official config,
provide it explicitly:

```bash
python -m tools.run_paper_fair_pairs ... \
  --baselines ssf \
  --overrides_json '{"ssf":{"optimizer":"adamw","epochs":100,"lr":0.001,"weight_decay":0.0,"warmup_epochs":5}}'
```

Such user-provided source settings are recorded as `user_source_override` in the
audit file.  If exact source settings are unavailable, use
`--require_verified_outer_recipe True` to skip the method rather than falling
back.

## C. Source-audited examples

### AdaptFormer

The official image-code defaults encoded in `baseline_recipes.py` are:

- ViT image route;
- bottleneck `64`;
- parallel FFN adapter, scalar `0.1`;
- SGD;
- 100 epochs;
- base LR `0.1 * effective_batch / 256`;
- weight decay `0`;
- 20 warm-up epochs.

### VPT

The official VTAB workflow searches LR/WD on validation data, scales LR by
batch size, and then performs final runs.  The paired runner reproduces the
encoded LR/WD search for **both VPT and TRSO** so neither side receives more HPO
budget.  Prompt length remains dataset/config specific and should be supplied
from the corresponding official paper config when available.

### VQT

The official source specifies query length `1` and Adam in its experiments.
Those constraints are preserved.  A row is not called paper-exact until the
corresponding dataset-specific LR/WD/epoch configuration is supplied.

## D. Provenance artifacts

Every paper-paired manifest has a sibling `*_audit.json` and `*_audit.csv`
containing:

- source repository/paper;
- mechanism/architecture/training fidelity;
- whether a fixed source recipe, source HPO grid, user override, or fallback was used;
- exact number of paired trials/runs;
- the equality rule applied to baseline and TRSO.
