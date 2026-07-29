# G-CREST-TRSO Ablation and Pipeline Audit

## Active full path

The active proposal uses one model-wide allocation path:

1. collect complete-loader odd/even gradient statistics;
2. compute reproducibility-weighted evidence for every layer-mode pair;
3. derive the global mode count from the geometric mean of numerical and Shannon support;
4. allocate the globally strongest modes;
5. train a full tangent core in every allocated tensor;
6. exactly merge the learned updates.

There is no loss gate, dense rescue, sparse rescue, architecture switch, manual rank, manual budget or Linear fallback.

## Analysis-only removals

| Variant | Removed component |
|---|---|
| `diagonal_only` | Off-diagonal interaction coordinates inside each allocated core |
| `no_sampling_variance` | Finite-sample variance term in mode reliability |
| `no_crossfit` | Independent odd/even reproducibility measurement |
| `head_only` | All tangent-core backbone adaptation |

## Reporting requirements

The result table must expose:

- globally selected modes;
- global Shannon effective modes;
- global numerical support;
- global geometric information dimension;
- allocated tensors and per-tensor ranks;
- tangent-core and head parameters separately;
- deployed extra parameters after merge;
- best trained epoch.

Only trained epochs are eligible for G-CREST model selection.
