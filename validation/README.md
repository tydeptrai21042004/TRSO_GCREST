# G-CREST controlled validation

These files record deterministic CPU transfer diagnostics. They are designed to expose implementation regressions and compare proposal geometries under matched conditions. They do **not** replace the real DTD/VTAB/structured-task benchmark.

- `GCREST_STANDARD_SEED*.json`: five-seed TinyCNN/TinyViT transfer controls.
- `GCREST_HARD_CNN_6SEED.json`: six-seed harder CNN transfer control.
- `GCREST_HARD_CNN_30EPOCH_SEED0.json`: long-horizon overfitting stress test.
- `GCREST_CONTROLLED_SUMMARY.json`: aggregate means, paired tests and bootstrap intervals.

The standard TinyViT control improves on all five seeds. The six-seed hard-CNN mean gain is small and its confidence interval includes zero; the limitation is intentionally preserved in the report.
