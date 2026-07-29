# Open dataset catalog

The automatic benchmark presets use openly available research datasets. Datasets that require a separate acceptance/download step are never silently fetched.

## Classification and related vector tasks

The existing registry includes CIFAR-10/100, MNIST-family datasets, SVHN, STL-10, Food-101, Oxford-IIIT Pet, Flowers-102, Stanford Cars, Caltech-101, DTD, EuroSAT, FGVC-Aircraft, SUN397, GTSRB, FER2013, PCam, Country211, Rendered-SST2, Places365, iNaturalist, CUB-200, NABirds, Stanford Dogs, VOC2007 multi-label, COCO multi-label, CelebA attributes/landmarks, VTAB adapters, few-shot metadata, ImageFolder and CSV.

## Semantic segmentation

| CLI name | Classes | Download behavior | Split policy |
|---|---:|---|---|
| `voc2012_segmentation` | 21 | torchvision automatic download | official train split partitioned into train/validation; official val reserved for final test |
| `oxford_pet_segmentation` | 3 | torchvision automatic download | trainval partitioned deterministically; official test retained |
| `sbd_segmentation` | 21 | torchvision automatic download | train partitioned deterministically; official val retained |
| `segmentation_folder` | user-defined | local open data | explicit train/val/test folders |
| `cityscapes_segmentation` | 19 | manual, registration/terms required | supported but excluded from open automatic presets |

## Depth estimation

`depth_folder` accepts any openly licensed paired depth dataset with one of these layouts:

```text
root/train/images + root/train/depth
root/val/images   + root/val/depth
root/test/images  + root/test/depth
```

or

```text
root/images/train + root/depth/train
root/images/val   + root/depth/val
root/images/test  + root/depth/test
```

Depth targets may be PNG/TIFF/JPEG, NumPy `.npy/.npz`, or PyTorch `.pt/.pth`. Use `--depth_scale` when stored units must be converted.

## Object detection

| CLI name | Classes | Download behavior | Metrics |
|---|---:|---|---|
| `voc_detection` | 21 including background | torchvision automatic download | COCO-style box mAP/AP50/AP75 and AR |
| `coco_detection` | 81 including background | local COCO 2017 files | COCO-style box metrics |

## Synthetic CI datasets

`fake_segmentation`, `fake_depth`, and `fake_detection` are deterministic contract tests. They must not be used as research evidence.
