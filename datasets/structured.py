"""Open structured-prediction datasets and paired transforms.

The module intentionally contains no proposal logic.  It only standardizes data
contracts for semantic segmentation, monocular depth, and object detection.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.datasets import CocoDetection
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from .preprocessing import resolve_normalization


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_DEPTH_EXTENSIONS = _IMAGE_EXTENSIONS | {".npy", ".npz", ".pt", ".pth"}



class PairedDenseTransform:
    """Apply identical geometry to image and dense target.

    Images use bilinear/bicubic interpolation.  Semantic masks always use
    nearest-neighbour interpolation; depth maps use bilinear interpolation.
    """

    def __init__(self, args, *, train: bool, target_kind: str):
        self.size = int(getattr(args, "input_size", 224))
        self.train = bool(train)
        self.target_kind = str(target_kind)
        self.augmentation = str(getattr(args, "dense_train_aug", "scale_crop_flip")).lower()
        self.hflip_probability = float(getattr(args, "dense_hflip_prob", 0.5))
        self.ignore_index = int(getattr(args, "segmentation_ignore_index", 255))
        self.normalization = resolve_normalization(args)

    def _to_target_tensor(self, target):
        if isinstance(target, torch.Tensor):
            tensor = target
        else:
            array = np.asarray(target)
            tensor = torch.from_numpy(array.copy())
        if self.target_kind == "segmentation":
            if tensor.ndim == 3:
                # Accept HWC masks and one-channel CHW masks.
                if tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                else:
                    tensor = tensor[..., 0]
            if tensor.ndim != 2:
                raise ValueError(f"Semantic mask must be HxW, got shape={tuple(tensor.shape)}")
            # torchvision functional resize/crop expects CHW for tensor inputs.
            return tensor.long().unsqueeze(0)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[-1] == 1:
            tensor = tensor.permute(2, 0, 1)
        return tensor.float()

    def __call__(self, image, target):
        if not isinstance(image, Image.Image):
            image = F.to_pil_image(image)
        target_tensor = self._to_target_tensor(target)

        if self.train and self.augmentation == "scale_crop_flip":
            scale = float(torch.empty(1).uniform_(0.75, 1.35).item())
            resized = max(self.size, int(round(self.size * scale)))
            image = F.resize(image, [resized, resized], interpolation=InterpolationMode.BICUBIC, antialias=True)
            interp = InterpolationMode.NEAREST if self.target_kind == "segmentation" else InterpolationMode.BILINEAR
            target_tensor = F.resize(target_tensor, [resized, resized], interpolation=interp, antialias=False)
            max_offset = max(0, resized - self.size)
            top = int(torch.randint(max_offset + 1, (1,)).item()) if max_offset else 0
            left = int(torch.randint(max_offset + 1, (1,)).item()) if max_offset else 0
            image = F.crop(image, top, left, self.size, self.size)
            target_tensor = F.crop(target_tensor, top, left, self.size, self.size)
            if bool(torch.rand(1).item() < self.hflip_probability):
                image = F.hflip(image)
                target_tensor = F.hflip(target_tensor)
        else:
            image = F.resize(image, [self.size, self.size], interpolation=InterpolationMode.BICUBIC, antialias=True)
            interp = InterpolationMode.NEAREST if self.target_kind == "segmentation" else InterpolationMode.BILINEAR
            target_tensor = F.resize(target_tensor, [self.size, self.size], interpolation=interp, antialias=False)

        image_tensor = F.to_tensor(image)
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.expand(3, -1, -1)
        elif image_tensor.shape[0] > 3:
            image_tensor = image_tensor[:3]
        if self.normalization is not None:
            image_tensor = F.normalize(image_tensor, *self.normalization)

        if self.target_kind == "segmentation":
            return image_tensor, target_tensor.squeeze(0).long()
        return image_tensor, target_tensor.float()


class TransformDataset(Dataset):
    def __init__(self, base: Dataset, transform, target_adapter=None):
        self.base = base
        self.transform = transform
        self.target_adapter = target_adapter

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, target = self.base[index]
        if self.target_adapter is not None:
            target = self.target_adapter(target)
        return self.transform(image, target)


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset, indices: Sequence[int]):
        self.base = base
        self.indices = list(map(int, indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.base[self.indices[index]]


def deterministic_split(length: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(int(length), generator=generator).tolist()
    val_count = min(max(1, int(round(length * float(val_ratio)))), max(1, length - 1))
    return order[val_count:], order[:val_count]


class OxfordPetSegmentation(Dataset):
    def __init__(self, root: str, split: str, download: bool, transform, seed: int, val_ratio: float):
        if split in {"train", "val"}:
            raw = datasets.OxfordIIITPet(root=root, split="trainval", target_types="segmentation", download=download)
            train_indices, val_indices = deterministic_split(len(raw), val_ratio, seed)
            raw = IndexedDataset(raw, train_indices if split == "train" else val_indices)
        else:
            raw = datasets.OxfordIIITPet(root=root, split="test", target_types="segmentation", download=download)
        self.base = TransformDataset(raw, transform, target_adapter=self._adapt)

    @staticmethod
    def _adapt(mask):
        # Official trimaps use 1=pet, 2=background, 3=border.  Preserve three
        # semantic classes while converting to zero-based labels.
        return torch.from_numpy(np.asarray(mask, dtype=np.int64).copy()) - 1

    def __len__(self): return len(self.base)
    def __getitem__(self, index): return self.base[index]


class VOCSegmentationThreeWay(Dataset):
    def __init__(self, root: str, split: str, download: bool, transform, seed: int, val_ratio: float, year: str = "2012"):
        if split in {"train", "val"}:
            raw = datasets.VOCSegmentation(root=root, year=year, image_set="train", download=download)
            train_indices, val_indices = deterministic_split(len(raw), val_ratio, seed)
            raw = IndexedDataset(raw, train_indices if split == "train" else val_indices)
        else:
            raw = datasets.VOCSegmentation(root=root, year=year, image_set="val", download=download)
        self.base = TransformDataset(raw, transform)

    def __len__(self): return len(self.base)
    def __getitem__(self, index): return self.base[index]


class SBDThreeWay(Dataset):
    def __init__(self, root: str, split: str, download: bool, transform, seed: int, val_ratio: float):
        if split in {"train", "val"}:
            raw = datasets.SBDataset(root=root, image_set="train", mode="segmentation", download=download)
            train_indices, val_indices = deterministic_split(len(raw), val_ratio, seed)
            raw = IndexedDataset(raw, train_indices if split == "train" else val_indices)
        else:
            raw = datasets.SBDataset(root=root, image_set="val", mode="segmentation", download=download)
        self.base = TransformDataset(raw, transform)

    def __len__(self): return len(self.base)
    def __getitem__(self, index): return self.base[index]


class CityscapesThreeWay(Dataset):
    def __init__(self, root: str, split: str, transform, seed: int, val_ratio: float):
        if split in {"train", "val"}:
            raw = datasets.Cityscapes(root=root, split="train", mode="fine", target_type="semantic")
            train_indices, val_indices = deterministic_split(len(raw), val_ratio, seed)
            raw = IndexedDataset(raw, train_indices if split == "train" else val_indices)
        else:
            raw = datasets.Cityscapes(root=root, split="val", mode="fine", target_type="semantic")
        self.base = TransformDataset(raw, transform, target_adapter=self._map_train_ids)

    @staticmethod
    def _map_train_ids(mask):
        raw = np.asarray(mask, dtype=np.int64)
        mapped = np.full(raw.shape, 255, dtype=np.int64)
        for category in datasets.Cityscapes.classes:
            train_id = int(category.train_id)
            if 0 <= train_id < 255:
                mapped[raw == int(category.id)] = train_id
        return torch.from_numpy(mapped)

    def __len__(self): return len(self.base)
    def __getitem__(self, index): return self.base[index]


class PairedFolderDataset(Dataset):
    """Generic open-data adapter for segmentation or depth folders.

    Supported layouts:
      root/{train,val,test}/images and root/{train,val,test}/{masks|depth}
      root/images/{train,val,test} and root/{masks|depth}/{train,val,test}
    Files are paired by relative stem.
    """

    def __init__(self, root: str, split: str, transform, target_dir: str, target_extensions: set[str]):
        root_path = Path(root)
        candidates = [
            (root_path / split / "images", root_path / split / target_dir),
            (root_path / "images" / split, root_path / target_dir / split),
        ]
        image_root = target_root = None
        for images, targets in candidates:
            if images.is_dir() and targets.is_dir():
                image_root, target_root = images, targets
                break
        if image_root is None:
            raise FileNotFoundError(
                f"Could not find paired folders for split={split!r}. Expected one of: "
                + ", ".join(f"{a} + {b}" for a, b in candidates)
            )
        target_by_stem = {
            path.relative_to(target_root).with_suffix("").as_posix(): path
            for path in target_root.rglob("*") if path.is_file() and path.suffix.lower() in target_extensions
        }
        samples = []
        for image_path in sorted(path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS):
            key = image_path.relative_to(image_root).with_suffix("").as_posix()
            target_path = target_by_stem.get(key)
            if target_path is not None:
                samples.append((image_path, target_path))
        if not samples:
            raise FileNotFoundError(f"No paired image/target files found under {image_root} and {target_root}")
        self.samples = samples
        self.transform = transform
        self.target_dir = target_dir

    def __len__(self): return len(self.samples)

    def _load_target(self, path: Path):
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return torch.from_numpy(np.load(path).astype(np.float32))
        if suffix == ".npz":
            archive = np.load(path)
            key = "depth" if "depth" in archive else archive.files[0]
            return torch.from_numpy(np.asarray(archive[key], dtype=np.float32))
        if suffix in {".pt", ".pth"}:
            value = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(value, dict):
                value = value.get("depth", value.get("target", next(iter(value.values()))))
            return torch.as_tensor(value)
        with Image.open(path) as image:
            array = np.asarray(image).copy()
        if self.target_dir == "depth":
            return torch.from_numpy(array.astype(np.float32))
        return torch.from_numpy(array.astype(np.int64))

    def __getitem__(self, index):
        image_path, target_path = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        target = self._load_target(target_path)
        return self.transform(image, target)


class FakeDenseDataset(Dataset):
    def __init__(self, size: int, image_size: int, task: str, classes: int, seed: int, transform):
        self.size = int(size)
        self.image_size = int(image_size)
        self.task = str(task)
        self.classes = int(classes)
        self.seed = int(seed)
        self.transform = transform

    def __len__(self): return self.size

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.seed + int(index))
        image = F.to_pil_image(torch.rand(3, self.image_size, self.image_size, generator=generator))
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, self.image_size), torch.linspace(0, 1, self.image_size), indexing="ij"
        )
        if self.task == "segmentation":
            target = ((xx * self.classes + yy * 2 + index) % self.classes).long()
        else:
            target = (0.5 + 4.5 * (0.7 * xx + 0.3 * yy)).unsqueeze(0).float()
        return self.transform(image, target)


class DetectionTransform:
    def __init__(self, args, train: bool):
        self.train = bool(train)
        self.normalization = None  # torchvision detection models normalize internally

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            image = F.to_pil_image(image)
        if self.train and bool(torch.rand(1).item() < 0.5):
            # Horizontal flipping boxes is handled by the dataset wrapper because
            # image width is needed there.
            pass
        return F.to_tensor(image)


VOC_CATEGORY_TO_INDEX = {
    name: index + 1 for index, name in enumerate((
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ))
}


class VOCObjectDetection(Dataset):
    def __init__(self, root: str, split: str, year: str, download: bool, train: bool):
        image_set = split
        self.base = datasets.VOCDetection(root=root, year=year, image_set=image_set, download=download)
        self.train = bool(train)

    def __len__(self): return len(self.base)

    def __getitem__(self, index):
        image, raw = self.base[index]
        width, _ = image.size
        objects = raw.get("annotation", {}).get("object", [])
        if isinstance(objects, dict):
            objects = [objects]
        boxes, labels, difficult = [], [], []
        for obj in objects:
            box = obj.get("bndbox", {})
            try:
                xmin, ymin = float(box["xmin"]) - 1.0, float(box["ymin"]) - 1.0
                xmax, ymax = float(box["xmax"]), float(box["ymax"])
                label = VOC_CATEGORY_TO_INDEX[obj["name"]]
            except Exception:
                continue
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
                difficult.append(int(obj.get("difficult", 0)))
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.train and bool(torch.rand(1).item() < 0.5):
            image = F.hflip(image)
            if boxes_tensor.numel():
                boxes_tensor[:, [0, 2]] = width - boxes_tensor[:, [2, 0]]
        image_tensor = F.to_tensor(image)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        area = ((boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])) if boxes_tensor.numel() else torch.zeros(0)
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area.float(),
            "iscrowd": torch.tensor(difficult, dtype=torch.int64),
        }
        return image_tensor, target


class CocoObjectDetection(Dataset):
    def __init__(self, image_root: str, annotation_file: str, train: bool):
        self.base = CocoDetection(image_root, annotation_file)
        category_ids = sorted(self.base.coco.getCatIds())
        self.category_to_label = {category_id: index + 1 for index, category_id in enumerate(category_ids)}
        self.train = bool(train)

    def __len__(self): return len(self.base)

    def __getitem__(self, index):
        image, annotations = self.base[index]
        width, _ = image.size
        boxes, labels, area, crowd = [], [], [], []
        for annotation in annotations:
            x, y, w, h = annotation.get("bbox", [0, 0, 0, 0])
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.category_to_label[int(annotation["category_id"])])
            area.append(float(annotation.get("area", w * h)))
            crowd.append(int(annotation.get("iscrowd", 0)))
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.train and bool(torch.rand(1).item() < 0.5):
            image = F.hflip(image)
            if boxes_tensor.numel():
                boxes_tensor[:, [0, 2]] = width - boxes_tensor[:, [2, 0]]
        target = {
            "boxes": boxes_tensor,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([int(self.base.ids[index])], dtype=torch.int64),
            "area": torch.tensor(area, dtype=torch.float32),
            "iscrowd": torch.tensor(crowd, dtype=torch.int64),
        }
        return F.to_tensor(image), target


class FakeDetectionDataset(Dataset):
    def __init__(self, size: int, image_size: int, classes: int, seed: int):
        self.size, self.image_size, self.classes, self.seed = int(size), int(image_size), int(classes), int(seed)

    def __len__(self): return self.size

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.rand(3, self.image_size, self.image_size, generator=generator)
        x1 = int(torch.randint(0, max(1, self.image_size // 3), (1,), generator=generator).item())
        y1 = int(torch.randint(0, max(1, self.image_size // 3), (1,), generator=generator).item())
        x2 = int(torch.randint(max(x1 + 2, self.image_size // 2), self.image_size + 1, (1,), generator=generator).item())
        y2 = int(torch.randint(max(y1 + 2, self.image_size // 2), self.image_size + 1, (1,), generator=generator).item())
        target = {
            "boxes": torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32),
            "labels": torch.tensor([1 + index % max(1, self.classes - 1)], dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.tensor([(x2 - x1) * (y2 - y1)], dtype=torch.float32),
            "iscrowd": torch.zeros(1, dtype=torch.int64),
        }
        return image, target


def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


__all__ = [
    "PairedDenseTransform", "OxfordPetSegmentation", "VOCSegmentationThreeWay",
    "SBDThreeWay", "CityscapesThreeWay", "PairedFolderDataset", "FakeDenseDataset",
    "VOCObjectDetection", "CocoObjectDetection", "FakeDetectionDataset", "detection_collate",
    "_DEPTH_EXTENSIONS", "_IMAGE_EXTENSIONS",
]
