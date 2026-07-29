"""Human-readable capability catalog for datasets, tasks, metrics, models, and baselines."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from task_registry import TASK_SPECS
from models.model_support import (
    ENGINEERING_CONTROL_METHODS, PAPER_BASELINE_METHODS,
    REFERENCE_CONTROL_METHODS, TRANSFERRED_CONTROL_METHODS,
)


@dataclass(frozen=True)
class DatasetCapability:
    name: str
    task: str
    access: str
    download: str
    notes: str


OPEN_DATASETS = (
    # Classification / multilabel / regression routes already supported by datasets.build.
    DatasetCapability("cifar10", "single_label", "open", "automatic", "10-class natural-image classification"),
    DatasetCapability("cifar100", "single_label", "open", "automatic", "100-class natural-image classification"),
    DatasetCapability("dtd", "single_label", "open", "automatic", "Describable Textures Dataset"),
    DatasetCapability("food101", "single_label", "open", "automatic", "101 food categories"),
    DatasetCapability("flowers102", "single_label", "open", "automatic", "102 flower categories"),
    DatasetCapability("oxfordiiitpet", "single_label", "open", "automatic", "37 pet breeds"),
    DatasetCapability("eurosat", "single_label", "open", "automatic", "satellite land-use classification"),
    DatasetCapability("country211", "single_label", "open", "automatic", "geolocation classification"),
    DatasetCapability("voc2007", "multilabel", "open", "automatic", "20-label image classification"),
    DatasetCapability("celeba", "multilabel/regression", "open research", "automatic", "attributes or landmark regression"),
    # Structured prediction.
    DatasetCapability("voc2012_segmentation", "semantic_segmentation", "open", "automatic", "21 semantic classes"),
    DatasetCapability("oxford_pet_segmentation", "semantic_segmentation", "open", "automatic", "three-class trimap segmentation"),
    DatasetCapability("sbd_segmentation", "semantic_segmentation", "open", "automatic", "Semantic Boundaries Dataset"),
    DatasetCapability("segmentation_folder", "semantic_segmentation", "open user data", "local", "generic paired images/masks"),
    DatasetCapability("depth_folder", "depth_estimation", "open user data", "local", "generic paired images/depth arrays or images"),
    DatasetCapability("voc_detection", "object_detection", "open", "automatic", "Pascal VOC bounding boxes"),
    DatasetCapability("coco_detection", "object_detection", "open", "manual", "COCO images and instance annotations"),
    DatasetCapability("fake_segmentation", "semantic_segmentation", "synthetic", "none", "CI and smoke tests"),
    DatasetCapability("fake_depth", "depth_estimation", "synthetic", "none", "CI and smoke tests"),
    DatasetCapability("fake_detection", "object_detection", "synthetic", "none", "CI and smoke tests"),
)


BACKBONE_CAPABILITIES = {
    "classification_multilabel_regression": {
        "sources": ("torchvision", "timm", "open_clip"),
        "examples": (
            "resnet18/50/101", "resnext50_32x4d", "wide_resnet50_2",
            "convnext_tiny/base", "efficientnet_v2_s", "mobilenet_v3_large",
            "vit_b_16", "vit_tiny_patch16_224", "deit_small_patch16_224",
            "swin_t/swin_b", "maxvit_t", "mixer_b16_224",
        ),
    },
    "semantic_segmentation_and_depth": {
        "sources": ("torchvision", "segmentation_models_pytorch"),
        "examples": (
            "fcn_resnet50/101", "deeplabv3_resnet50/101",
            "deeplabv3_mobilenet_v3_large", "lraspp_mobilenet_v3_large",
            "unet:resnet34", "unetplusplus:efficientnet-b0",
            "fpn:resnet50", "pspnet:resnet50", "deeplabv3plus:resnet50",
        ),
    },
    "object_detection": {
        "sources": ("torchvision",),
        "examples": (
            "fasterrcnn_resnet50_fpn_v2", "fasterrcnn_mobilenet_v3_large_fpn",
            "retinanet_resnet50_fpn_v2", "fcos_resnet50_fpn",
            "maskrcnn_resnet50_fpn_v2",
        ),
    },
}


IMPLEMENTED_PAPER_BASELINES = tuple(sorted(PAPER_BASELINE_METHODS))
REFERENCE_CONTROLS = tuple(sorted(REFERENCE_CONTROL_METHODS))
ENGINEERING_CONTROLS = tuple(sorted(ENGINEERING_CONTROL_METHODS))
TRANSFERRED_CONTROLS = tuple(sorted(TRANSFERRED_CONTROL_METHODS))
IMPLEMENTED_BASELINES = (
    *REFERENCE_CONTROLS,
    "trso",
    *IMPLEMENTED_PAPER_BASELINES,
    *ENGINEERING_CONTROLS,
    *TRANSFERRED_CONTROLS,
)



def as_dict() -> dict:
    return {
        "tasks": {name: asdict(spec) for name, spec in TASK_SPECS.items()},
        "open_datasets": [asdict(row) for row in OPEN_DATASETS],
        "backbones": BACKBONE_CAPABILITIES,
        "implemented_baselines": IMPLEMENTED_BASELINES,
        "strict_paper_baselines": IMPLEMENTED_PAPER_BASELINES,
        "reference_controls": REFERENCE_CONTROLS,
        "engineering_controls_opt_in": ENGINEERING_CONTROLS,
        "transferred_controls_opt_in": TRANSFERRED_CONTROLS,
    }
