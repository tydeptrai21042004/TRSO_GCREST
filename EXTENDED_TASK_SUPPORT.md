# Extended open-data and multi-task support

The active proposal in `models/tuning_modules/mdl_tangent_core.py` is unchanged. This extension changes only data contracts, model construction, training/evaluation routing, reporting, baselines, tests, and release tooling.

## Task matrix

| Task | Open-data routes | Primary selection metric | Reported quality metrics | TRSO |
|---|---|---|---|---|
| Single-label classification | torchvision/timm datasets, ImageFolder, CSV, VTAB/few-shot adapters | Acc@1 ↑ | Acc@1/5, macro/weighted F1, balanced accuracy, macro OVR AUROC, macro AP, MCC, Cohen κ, ECE, Brier, NLL, entropy, per-class metrics | Yes |
| Multi-label classification | VOC2007, COCO, CelebA, CSV | mAP ↑ | mAP, macro/micro AUROC, macro/micro precision/recall/F1, subset and Hamming accuracy, ECE, Brier, per-label AP/AUROC | Yes |
| Regression | CelebA landmarks, CSV, synthetic | MAE ↓ | MAE, median AE, RMSE, R², Pearson, Spearman, per-output diagnostics | Yes |
| Semantic segmentation | VOC2012, Oxford Pet trimaps, SBD, paired folders | mIoU ↑ | mIoU, mean Dice, pixel accuracy, mean class accuracy, frequency-weighted IoU, per-class IoU/Dice/support | Yes |
| Monocular depth | paired open folder adapter | AbsRel ↓ | AbsRel, SqRel, RMSE, RMSE-log, SILog, MAE, log10, δ1/δ2/δ3, valid pixels | Yes |
| Object detection | Pascal VOC, COCO, generic synthetic CI | box mAP ↑ | mAP@[.50:.95], AP50, AP75, AR1/10/100, image/class counts | No: detector training needs target-aware forwards during calibration, which would change the proposal contract |

## Backbone coverage

### Classification, multi-label and regression

Any replaceable-head model exposed by torchvision or timm is eligible for the architecture-agnostic baselines. Examples include ResNet/ResNeXt/Wide-ResNet, ConvNeXt, EfficientNet, MobileNet, DenseNet, RegNet, ViT/DeiT/BEiT/EVA, Swin, MaxViT and MLP-Mixer families.

### Segmentation and depth

Built-in torchvision architectures:

- FCN ResNet-50/101
- DeepLabV3 ResNet-50/101
- DeepLabV3 MobileNetV3-Large
- LR-ASPP MobileNetV3-Large

Optional `segmentation_models_pytorch` syntax uses `architecture:encoder`, for example:

- `unet:resnet34`
- `unetplusplus:efficientnet-b0`
- `fpn:resnet50`
- `pspnet:resnet50`
- `deeplabv3plus:resnet50`

The same dense-output wrappers can produce semantic logits or positive depth maps.

### Object detection

- Faster R-CNN ResNet-50-FPN and MobileNetV3-FPN
- RetinaNet ResNet-50-FPN
- FCOS ResNet-50-FPN
- Mask R-CNN ResNet-50-FPN (box evaluation is enabled; mask evaluation is a future extension)

## Timing and efficiency report

Each run writes `timing_summary.json` and exports these fields through the aggregate CSV:

- proposal calibration time
- setup plus calibration time
- total training time
- mean and median epoch time
- time to best validation checkpoint
- training samples per second
- final evaluation time
- complete wall-clock time
- GPU-hours
- trainable and total parameters
- trainable ratio
- frozen proposal-basis storage
- FLOPs when the profiler supports the task
- inference latency, FPS and peak memory

## Example commands

Semantic segmentation:

```bash
python -m tools.run_fair_suite \
  --dataset voc2012_segmentation --download True \
  --task semantic_segmentation --backbones auto \
  --methods linear,norm,bias,last_block,trso,full \
  --seeds 0,1,2 --epochs 50 --execute
```

Depth folder:

```bash
python -m tools.run_fair_suite \
  --dataset depth_folder --task depth_estimation \
  --data_path /path/to/open_depth_dataset \
  --backbones auto --methods linear,norm,bias,last_block,trso,full \
  --seeds 0,1,2 --execute
```

Detection:

```bash
python -m tools.run_fair_suite \
  --dataset voc_detection --download True --task object_detection \
  --backbones auto --methods linear,norm,bias,last_block,full \
  --seeds 0,1,2 --execute
```
