"""Structured-prediction model builders.

These builders only adapt task heads and model I/O contracts.  They do not alter
G-CREST-TRSO or any proposal selection rule.
"""
from __future__ import annotations

import inspect
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F
import torchvision


TORCHVISION_SEGMENTATION_MODELS = {
    "fcn_resnet50", "fcn_resnet101", "deeplabv3_resnet50", "deeplabv3_resnet101",
    "deeplabv3_mobilenet_v3_large", "lraspp_mobilenet_v3_large",
}
TORCHVISION_DETECTION_MODELS = {
    "fasterrcnn_resnet50_fpn", "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_mobilenet_v3_large_fpn", "fasterrcnn_mobilenet_v3_large_320_fpn",
    "retinanet_resnet50_fpn", "retinanet_resnet50_fpn_v2", "fcos_resnet50_fpn",
    "maskrcnn_resnet50_fpn", "maskrcnn_resnet50_fpn_v2",
}


def _replace_last_conv(module: nn.Module, out_channels: int) -> int:
    direct = [(name, child) for name, child in module.named_children() if isinstance(child, nn.Conv2d)]
    count = 0
    if direct:
        chosen = direct if module.__class__.__name__.lower() == "lraspphead" else [direct[-1]]
        for name, conv in chosen:
            module._modules[name] = nn.Conv2d(
                conv.in_channels, int(out_channels), kernel_size=conv.kernel_size,
                stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
                groups=conv.groups, bias=conv.bias is not None, padding_mode=conv.padding_mode,
            )
            count += 1
        return count
    for _, child in module.named_children():
        count += _replace_last_conv(child, out_channels)
    return count


def replace_segmentation_head(model: nn.Module, out_channels: int) -> int:
    count = 0
    for attribute in ("classifier", "aux_classifier", "decode_head", "segmentation_head"):
        module = getattr(model, attribute, None)
        if isinstance(module, nn.Module):
            count += _replace_last_conv(module, out_channels)
    if count == 0:
        raise RuntimeError(f"No replaceable segmentation head found in {model.__class__.__name__}")
    return count


class DenseOutputModel(nn.Module):
    """Normalize dense model outputs to ``{'out': Tensor[B,C,H,W]}``."""

    def __init__(self, model: nn.Module, *, positive: bool = False, min_value: float = 1e-3):
        super().__init__()
        self.model = model
        self.positive = bool(positive)
        self.min_value = float(min_value)
        self.backbone_family = getattr(model, "backbone_family", "cnn")

    def forward(self, images: torch.Tensor):
        output = self.model(images)
        if isinstance(output, dict):
            tensor = output.get("out")
            if tensor is None:
                tensor = next(value for value in output.values() if isinstance(value, torch.Tensor))
        elif isinstance(output, (tuple, list)):
            tensor = output[0]
        else:
            tensor = output
        if self.positive:
            tensor = F.softplus(tensor) + self.min_value
        return {"out": tensor}


def _weights_enum_default(module, model_name: str):
    try:
        function = getattr(module, model_name)
        signature = inspect.signature(function)
        annotation = signature.parameters.get("weights").annotation
        # Torchvision annotations are often strings/unions, so use the public
        # get_model_weights helper when available.
        from torchvision.models import get_model_weights
        enum = get_model_weights(function)
        return enum.DEFAULT
    except Exception:
        return None


def build_torchvision_segmentation(model_name: str, num_outputs: int, weights: str = "DEFAULT", *, depth: bool = False) -> nn.Module:
    normalized = str(model_name).lower()
    if normalized not in TORCHVISION_SEGMENTATION_MODELS:
        raise ValueError(f"Unsupported torchvision segmentation model {model_name!r}. Available: {sorted(TORCHVISION_SEGMENTATION_MODELS)}")
    function = getattr(torchvision.models.segmentation, normalized)
    requested_pretrained = str(weights).lower() not in {"none", "scratch", "random", "false"}
    resolved_weights = _weights_enum_default(torchvision.models.segmentation, normalized) if requested_pretrained else None
    kwargs = {"weights": resolved_weights}
    if not requested_pretrained:
        kwargs["weights_backbone"] = None
    model = function(**kwargs)
    replace_segmentation_head(model, int(num_outputs))
    model.backbone_family = "resnet" if "resnet" in normalized else "cnn"
    return DenseOutputModel(model, positive=depth)


def build_smp_model(spec: str, num_outputs: int, *, depth: bool = False, encoder_weights: str | None = "imagenet") -> nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except Exception as exc:
        raise RuntimeError(
            "segmentation_models_pytorch is required for --model_source smp. "
            "Install requirements-structured.txt."
        ) from exc
    text = str(spec)
    if ":" in text:
        architecture, encoder = text.split(":", 1)
    else:
        architecture, encoder = "unet", text
    architecture_map = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "fpn": smp.FPN,
        "pspnet": smp.PSPNet,
        "deeplabv3": smp.DeepLabV3,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "pan": smp.PAN,
        "manet": smp.MAnet,
        "linknet": smp.Linknet,
    }
    key = architecture.lower().replace("_", "")
    if key not in architecture_map:
        raise ValueError(f"Unknown SMP architecture {architecture!r}. Available: {sorted(architecture_map)}")
    model = architecture_map[key](encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=int(num_outputs))
    model.backbone_family = "transformer" if any(token in encoder.lower() for token in ("mit_", "swin", "maxvit", "tu-vit", "mobilevit")) else "cnn"
    return DenseOutputModel(model, positive=depth)


def build_torchvision_detection(model_name: str, num_classes: int, weights: str = "DEFAULT", input_size: int = 800) -> nn.Module:
    normalized = str(model_name).lower()
    if normalized not in TORCHVISION_DETECTION_MODELS:
        raise ValueError(f"Unsupported detection model {model_name!r}. Available: {sorted(TORCHVISION_DETECTION_MODELS)}")
    module = torchvision.models.detection
    function = getattr(module, normalized)
    pretrained_backbone = str(weights).lower() not in {"none", "scratch", "random", "false"}

    kwargs = {
        "weights": None,
        "num_classes": int(num_classes),
        "min_size": int(input_size),
        "max_size": int(input_size),
    }
    if pretrained_backbone:
        # Detection heads must match the downstream category count.  Use the
        # official ImageNet-pretrained backbone and initialize task heads anew.
        if "mobilenet" in normalized:
            from torchvision.models import MobileNet_V3_Large_Weights
            kwargs["weights_backbone"] = MobileNet_V3_Large_Weights.DEFAULT
        else:
            from torchvision.models import ResNet50_Weights
            kwargs["weights_backbone"] = ResNet50_Weights.DEFAULT
    else:
        kwargs["weights_backbone"] = None
    model = function(**kwargs)
    model.backbone_family = "resnet" if "resnet" in normalized else "cnn"
    return model


def available_structured_backbones() -> dict[str, tuple[str, ...]]:
    return {
        "torchvision_segmentation": tuple(sorted(TORCHVISION_SEGMENTATION_MODELS)),
        "torchvision_detection": tuple(sorted(TORCHVISION_DETECTION_MODELS)),
        "smp_examples": (
            "unet:resnet50", "fpn:efficientnet-b0", "deeplabv3plus:resnet50",
            "unetplusplus:tu-convnext_tiny", "fpn:mit_b0",
        ),
    }


__all__ = [
    "TORCHVISION_SEGMENTATION_MODELS", "TORCHVISION_DETECTION_MODELS",
    "DenseOutputModel", "replace_segmentation_head", "build_torchvision_segmentation",
    "build_smp_model", "build_torchvision_detection", "available_structured_backbones",
]
