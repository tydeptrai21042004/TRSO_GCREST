# Repository capability matrix

## Tasks and metrics

| Task | Primary metric | Direction | Required metrics |
|---|---|---|---|
| `single_label` | `acc1` | maximize | acc1, acc5, macro_f1, weighted_f1, balanced_accuracy, macro_auroc_ovr, macro_average_precision, mcc, cohen_kappa, ece, brier_score, nll |
| `multilabel` | `map` | maximize | map, macro_auroc, micro_auroc, micro_f1, macro_f1, weighted_f1, subset_accuracy, hamming_accuracy, ece, brier_score |
| `regression` | `mae` | minimize | mae, median_absolute_error, rmse, r2, pearson, spearman |
| `semantic_segmentation` | `miou` | maximize | miou, mean_dice, pixel_accuracy, mean_class_accuracy, frequency_weighted_iou, loss |
| `depth_estimation` | `abs_rel` | minimize | abs_rel, sq_rel, rmse, rmse_log, silog, delta1, delta2, delta3, mae, log10 |
| `object_detection` | `map` | maximize | map, map_50, map_75, mar_1, mar_10, mar_100, loss |

## Open datasets

| Dataset | Task | Access | Download | Notes |
|---|---|---|---|---|
| `cifar10` | `single_label` | open | automatic | 10-class natural-image classification |
| `cifar100` | `single_label` | open | automatic | 100-class natural-image classification |
| `dtd` | `single_label` | open | automatic | Describable Textures Dataset |
| `food101` | `single_label` | open | automatic | 101 food categories |
| `flowers102` | `single_label` | open | automatic | 102 flower categories |
| `oxfordiiitpet` | `single_label` | open | automatic | 37 pet breeds |
| `eurosat` | `single_label` | open | automatic | satellite land-use classification |
| `country211` | `single_label` | open | automatic | geolocation classification |
| `voc2007` | `multilabel` | open | automatic | 20-label image classification |
| `celeba` | `multilabel/regression` | open research | automatic | attributes or landmark regression |
| `voc2012_segmentation` | `semantic_segmentation` | open | automatic | 21 semantic classes |
| `oxford_pet_segmentation` | `semantic_segmentation` | open | automatic | three-class trimap segmentation |
| `sbd_segmentation` | `semantic_segmentation` | open | automatic | Semantic Boundaries Dataset |
| `segmentation_folder` | `semantic_segmentation` | open user data | local | generic paired images/masks |
| `depth_folder` | `depth_estimation` | open user data | local | generic paired images/depth arrays or images |
| `voc_detection` | `object_detection` | open | automatic | Pascal VOC bounding boxes |
| `coco_detection` | `object_detection` | open | manual | COCO images and instance annotations |
| `fake_segmentation` | `semantic_segmentation` | synthetic | none | CI and smoke tests |
| `fake_depth` | `depth_estimation` | synthetic | none | CI and smoke tests |
| `fake_detection` | `object_detection` | synthetic | none | CI and smoke tests |

## Backbone families

### Classification Multilabel Regression
- Sources: torchvision, timm, open_clip
- Examples: resnet18/50/101, resnext50_32x4d, wide_resnet50_2, convnext_tiny/base, efficientnet_v2_s, mobilenet_v3_large, vit_b_16, vit_tiny_patch16_224, deit_small_patch16_224, swin_t/swin_b, maxvit_t, mixer_b16_224

### Semantic Segmentation And Depth
- Sources: torchvision, segmentation_models_pytorch
- Examples: fcn_resnet50/101, deeplabv3_resnet50/101, deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large, unet:resnet34, unetplusplus:efficientnet-b0, fpn:resnet50, pspnet:resnet50, deeplabv3plus:resnet50

### Object Detection
- Sources: torchvision
- Examples: fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn, retinanet_resnet50_fpn_v2, fcos_resnet50_fpn, maskrcnn_resnet50_fpn_v2

## Strict benchmark methods

- Reference controls: `full`, `linear`.
- Proposal: `trso`.
- Opt-in engineering controls: `norm`, `bias`, `last_block`.
- Opt-in transferred controls: `lora`, `bitfit`, `sidetune`.

The last two groups are never scheduled by `--methods auto`; use `--allow_nonpaper_controls True` only when reporting them as separately labeled controls. Exact architecture/task boundaries are in `SUPPORT_MATRIX.md`.
