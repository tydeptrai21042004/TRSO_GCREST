# Baseline fidelity status

The revision distinguishes four questions that were previously conflated:

1. **Mechanism fidelity** — is the published adaptation operation implemented?
2. **Architecture fidelity** — is it evaluated on a backbone covered by the paper?
3. **Training fidelity** — are optimizer/schedule/HPO settings recoverable from the paper/official repository?
4. **Comparison fairness** — does TRSO receive the same outer experimental opportunity?

The canonical metadata lives in `baseline_recipes.py` and is copied into each
paper-paired audit manifest.

## Important corrections

- AdaptFormer's paper/official image default is now `adaptformer_dim=64` rather
  than the previous controlled `16`.
- The controlled main table no longer gives PEFT methods/TRSO an implicit
  linear-probe head warm start.  Default `head_init_policy=random` and
  `peft_head_lr_scale=1.0` are enforced by the runner/verifier.
- VPT is treated as a **search-based source recipe**, not as one universal LR.
- VQT preserves the official Adam/query-length constraints; missing task-specific
  LR/WD settings are never invented and labelled paper-exact.
- ML-Decoder and SegAdapter revision rows are explicitly described as published
  mechanisms transferred to the manuscript's chosen backbone/task settings,
  not exact reproductions of an original result table.

## Fidelity classes used by the code

| Label | Meaning |
|---|---|
| `exact_official_default` | a public official source exposes a single default recipe encoded here |
| `source_search` | official source specifies a search procedure/grid; use validation selection |
| `official_*_required` / `paper_search_required` | mechanism is implemented but exact dataset/task recipe must be supplied from the source config |
| `matched_source_constraints_fallback_not_paper_exact` | known source constraints are preserved, unknown outer fields use the transparent matched fallback |
| `matched_controlled_fallback_not_paper_exact` | no exact outer recipe is encoded; baseline and TRSO share a controlled fallback |
| `transferred_controlled` | published mechanism evaluated in a new backbone/task route |

## Reporting rule

The final paper should report two complementary tables/analyses:

- **Controlled comparison:** same outer training protocol, source-audited method
  structure, fresh task heads.  This is the cleanest causal comparison of the
  adaptation mechanisms.
- **Paper-recipe paired sensitivity:** each recoverable baseline paper recipe is
  paired with TRSO under the identical recipe/HPO budget.  This tests whether a
  conclusion depends on the optimizer chosen for the controlled table.

Do not label a transferred or fallback row as "exact reproduction".  The audit
JSON produced by the runners contains the exact fidelity label to use.
