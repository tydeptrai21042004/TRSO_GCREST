# Extension validation report

## Proposal preservation

`models/tuning_modules/mdl_tangent_core.py` was kept byte-for-byte unchanged.

```text
SHA-256 before: d4d7af28492b40fd7b0b3687670ae34ab32d034f7a66f4f51a008a95c805a7e9
SHA-256 after:  d4d7af28492b40fd7b0b3687670ae34ab32d034f7a66f4f51a008a95c805a7e9
```

The extension is limited to task/data/model adapters, training and evaluation routing, metrics, runners, reporting, tests, baseline policies, documentation, and release cleanup.

## Executed tests

- Full repository regression suite: 97 tests after the final structured-runner test was added.
- Synthetic end-to-end classification route: completed.
- Synthetic TRSO semantic-segmentation route: calibration, one training epoch, validation safeguard, checkpoint reload, exact merge, final test completed.
- Synthetic depth route: one training epoch and all depth metrics completed.
- Synthetic object-detection route: detector loss training and box mAP/AP/AR evaluation completed.
- Result aggregation: all four task outputs and timing summaries were merged into one CSV.

Synthetic metrics validate software contracts only and are not research results.

## New task metrics

- Classification: Acc@1/5, macro/weighted F1, balanced accuracy, macro OVR AUROC, macro AP, MCC, Cohen kappa, ECE, Brier, NLL and per-class diagnostics.
- Multi-label: mAP, macro/micro AUROC, macro/micro precision/recall/F1, subset/Hamming accuracy, ECE and per-label diagnostics.
- Regression: MAE, median AE, RMSE, R², Pearson and Spearman.
- Segmentation: mIoU, Dice, pixel accuracy, class accuracy, frequency-weighted IoU and per-class diagnostics.
- Depth: AbsRel, SqRel, RMSE, RMSE-log, SILog, MAE, log10 and delta thresholds.
- Detection: mAP@[.50:.95], AP50, AP75 and AR1/10/100.

## Timing and system metrics

Every completed run writes proposal calibration, profiling, setup, total training, mean/median epoch, time-to-best, throughput, final evaluation, complete wall-clock, accelerator count, GPU-hours, peak training memory, inference latency/FPS/memory, FLOPs where supported, and parameter/storage metrics.

## Explicit limitation

TRSO is supported unchanged for classification, multi-label, regression, semantic segmentation and depth estimation. It is explicitly skipped for torchvision object detectors because their calibration loss requires `model(images, targets)`. Adding detector-specific target-aware calibration would alter the active proposal contract and was therefore not done.
