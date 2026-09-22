# G-CREST-TRSO Ablation and Pipeline Audit

## Terminology

The public manuscript uses **partition-consistency weighting**. The historical
CLI token `no_crossfit` is retained only for compatibility and means **without
partition-consistency weighting**; the two deterministic calibration partitions
are not statistical cross-fitting and do not estimate repeated-run
reproducibility.

## Active full path

The active proposal uses one model-wide allocation path:

1. collect calibration-gradient statistics from two deterministic partitions;
2. compute partition-consistency-weighted evidence for every tensor-mode pair;
3. derive the default global retained-mode count from the geometric mean of positive support \(D_0\) and Shannon effective support \(D_1\);
4. allocate the globally strongest modes;
5. train a full spectral/tangent core in every participating tensor;
6. exactly merge the learned updates.

There is no loss gate, dense rescue, sparse rescue, architecture switch, manual
rank, manual global budget, or Linear fallback in the full proposal.

The geometric \(D_0\)--\(D_1\) rule is a parameter-free default intermediate
scale, not an accuracy-optimality claim. Realized capacity can still depend on
calibration construction through the candidate cap \(\rho_\ell\).

## Analysis-only removals

| Variant | Removed component |
|---|---|
| `diagonal_only` | Off-diagonal interaction coordinates inside each allocated core |
| `no_sampling_variance` | Tensor-level sampling-variance term in the partition-consistency weight |
| `no_crossfit` | **Partition-consistency weighting** (legacy token retained for compatibility) |
| `head_only` | All spectral-core backbone adaptation |

## Reporting requirements

The result table/metadata should expose:

- globally retained modes \(R\);
- global Shannon effective modes \(D_1\);
- global positive support \(D_0\);
- participating tensors and per-tensor ranks;
- spectral-core and head parameters separately;
- deployed extra parameters after merge;
- best trained epoch;
- calibration partition mode/seed and candidate-cap controls.

When completed per-run `mdl_tangent_calibration.json` files are available,
`tools/aggregate_revision_results.py` also reports optional tensor-level
stability diagnostics: selected-tensor Jaccard overlap, tensor-rank Spearman
correlation, pairwise rank MAE, and a separate partition-seed stability table.
These diagnostics are analysis outputs only and do not change training.
