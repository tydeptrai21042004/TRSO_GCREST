# Baseline Fairness and Paper-Fidelity Revision — 2026-09-11

This revision separates two scientifically different questions that should not be conflated:

1. **Controlled main comparison** — every method uses the same outer experimental conditions so that TRSO and the baselines are compared fairly.
2. **Paper-recipe paired comparison** — each baseline is run with the closest verified original-paper / official-repository recipe available, and TRSO is rerun under the same outer recipe for a paired head-to-head comparison.

## Main corrections

- Removed the implicit linear-probe classifier warm-start from the default PEFT comparison.
- Fresh/random task heads are now the default for the controlled main table, with `peft_head_lr_scale=1.0`.
- The old linear-probe warm-start remains available only as the explicit `--head_init_policy linear_probe` ablation.
- Added `baseline_recipes.py` as the single source of baseline method defaults and paper/official-repository provenance.
- Corrected the AdaptFormer image default bottleneck to 64 and encoded the official image-training defaults used by the paper-recipe runner.
- Added a source-grid VPT protocol rather than claiming one universal fixed VPT learning rate/weight decay.
- Added explicit fidelity labels for exact/default, source-audited partial, and transferred/controlled implementations.
- Added paired baseline-vs-TRSO generation, pair verification, and validation-only hyperparameter selection.
- Added optimizer support for Adam and a constant scheduler so verified source recipes can be represented without forcing AdamW/cosine.
- Updated reviewer documentation so historical protocols are no longer presented as the active certification state.

## New / major files

- `baseline_recipes.py`
- `tools/run_paper_fair_pairs.py`
- `tools/verify_paper_pairs.py`
- `tools/select_paper_pair_trials.py`
- `tests/test_paper_fair_pair_protocol.py`

Major revised files include:

- `main.py`
- `tools/run_fair_suite.py`
- `tools/verify_fairness.py`
- `tools/revision_full_protocol.py`
- `BASELINE_FIDELITY.md`
- `PAPER_BASELINE_REPRODUCTION.md`
- `BASELINE_AND_ABLATION_SEPARATION.md`
- `README.md`
- `REVIEWER_REVISION_GUIDE.md`
- `FINAL_REVIEWER_REVISION_CODE_REPORT.md`

## Recommended final experiments

### A. Controlled main table

Use the controlled runner with fresh heads. This is the primary method-comparison table.

```bash
python -m tools.revision_full_protocol --out_dir runs/reviewer_controlled
python -m tools.verify_fairness --manifest runs/reviewer_controlled/manifest.json
```

### B. Paper-recipe paired robustness table

Generate baseline/TRSO pairs. For methods whose official source exposes an exact/default recipe, the verified recipe is encoded. For methods with dataset-specific or unavailable source configurations, provide recovered values through the JSON override interface rather than inventing values.

```bash
python -m tools.run_paper_fair_pairs \
  --out_dir runs/paper_pairs \
  --baselines auto \
  --search_mode full

python -m tools.verify_paper_pairs \
  --manifest runs/paper_pairs/manifest.json
```

For a strict source-only sensitivity analysis:

```bash
python -m tools.run_paper_fair_pairs \
  --out_dir runs/paper_pairs_verified_only \
  --baselines auto \
  --require_verified_outer_recipe True
```

Dataset-specific source settings can be supplied without changing code:

```bash
python -m tools.run_paper_fair_pairs \
  --out_dir runs/paper_pairs_exact \
  --overrides_json official_outer_recipes.json \
  --method_overrides_json official_method_configs.json
```

After training all search trials, select hyperparameters **only from validation metrics** and then report test performance for the selected trial:

```bash
python -m tools.select_paper_pair_trials \
  --manifest runs/paper_pairs/manifest.json \
  --out_json runs/paper_pairs/selected_trials.json
```

## Validation status

- Full automated test suite: **182 passed**.
- Controlled fairness verifier: **fair=True, 0 errors, 0 warnings** in smoke validation.
- Paired-paper verifier: **fair=True, 0 errors** in smoke validation; partial-recipe fallbacks intentionally generate a warning rather than being mislabeled as exact paper reproduction.
- Training smoke test with Adam + constant scheduler completed successfully.

## Interpretation rule

Do **not** describe every method as an exact original-paper reproduction. The code now separates:

- mechanism fidelity,
- architecture fidelity,
- outer training-recipe fidelity,
- task/dataset transfer,
- and controlled-comparison status.

This makes the main TRSO comparison fair while allowing a second, independently auditable paper-recipe comparison without giving either TRSO or a baseline an initialization, training-budget, or validation-selection advantage.
