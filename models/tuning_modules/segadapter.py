"""SegAdapter baseline from Peng & Kameyama (ACML 2023 / PMLR 222).

This is a paper-faithful PyTorch reimplementation of the published SegAdapter
block for CNN feature maps.  It follows Eqs. (3)--(6) and (12):
  SegAttention(X) = Norm(DSConv(X)) * Linear(X)
  HSA(X)          = SegAttention(X) * SegAttention(Cls(X))
  Y               = X + mu * FFN(HSA(X))
and uses the paper defaults kernel=5, FFN ratio=3, stage-wise mu, and auxiliary
cross-entropy weight lambda=0.4.

The original paper demonstrates ConvNeXt/SegFormer; here the same plug-and-play
block is inserted into the four final MobileNetV3 stages used by LR-ASPP.  This
is a reimplementation of the published method, not a new baseline variant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm over channels independently at every spatial location."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(channels))

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SegAttention(nn.Module):
    """Equation (3): normalized depthwise-separable response times linear value."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        padding = int(kernel_size) // 2
        self.dsconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
        )
        self.norm = LayerNorm2d(out_channels)
        self.linear = nn.Conv2d(in_channels, out_channels, 1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.dsconv(x)) * self.linear(x)


class SegAdapterBlock(nn.Module):
    """Published HSA + FFN + learnable channel-wise scaled residual block."""
    def __init__(
        self,
        channels: int,
        num_classes: int,
        *,
        kernel_size: int = 5,
        ffn_ratio: float = 3.0,
        mu_init: float = 1e-5,
    ) -> None:
        super().__init__()
        channels = int(channels)
        hidden = max(1, int(round(channels * float(ffn_ratio))))
        self.side_classifier = nn.Conv2d(channels, int(num_classes), 1)
        self.feature_attention = SegAttention(channels, channels, kernel_size)
        self.semantic_attention = SegAttention(int(num_classes), channels, kernel_size)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.mu = nn.Parameter(torch.full((1, channels, 1, 1), float(mu_init)))
        self.last_side_logits: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        side = self.side_classifier(x)
        hsa = self.feature_attention(x) * self.semantic_attention(side)
        self.last_side_logits = side
        return x + self.mu * self.ffn(hsa)


class SegAdapterStage(nn.Module):
    """Wrap one existing backbone stage and apply one SegAdapter block after it."""
    def __init__(self, stage: nn.Module, adapter: SegAdapterBlock) -> None:
        super().__init__()
        self.stage = stage
        self.segadapter = adapter

    @property
    def last_side_logits(self):
        return self.segadapter.last_side_logits

    def forward(self, x: Tensor) -> Tensor:
        return self.segadapter(self.stage(x))


@dataclass(frozen=True)
class SegAdapterInstallReport:
    stage_names: tuple[str, ...]
    stage_channels: tuple[int, ...]
    mu_init: tuple[float, ...]
    kernel_size: int
    ffn_ratio: float
    aux_weight: float


def _infer_mobile_stage_channels(backbone: nn.Module, stage_names: list[str], input_size: int) -> dict[str, int]:
    modules = list(backbone._modules.items())
    was_training = backbone.training
    backbone.eval()
    device = next(backbone.parameters()).device
    x = torch.zeros(1, 3, int(input_size), int(input_size), device=device)
    channels: dict[str, int] = {}
    with torch.no_grad():
        for name, module in modules:
            x = module(x)
            if name in stage_names:
                channels[name] = int(x.shape[1])
    backbone.train(was_training)
    missing = [name for name in stage_names if name not in channels]
    if missing:
        raise RuntimeError(f"Could not infer SegAdapter channels for stages {missing}")
    return channels


def apply_segadapter_lraspp(
    dense_model: nn.Module,
    *,
    num_classes: int,
    input_size: int = 224,
    kernel_size: int = 5,
    ffn_ratio: float = 3.0,
    aux_weight: float = 0.4,
) -> SegAdapterInstallReport:
    """Insert SegAdapter at four MobileNetV3 stage ends inside torchvision LR-ASPP."""
    tv_model = getattr(dense_model, "model", None)
    backbone = getattr(tv_model, "backbone", None)
    if backbone is None or not hasattr(backbone, "_modules"):
        raise TypeError("SegAdapter LR-ASPP route requires DenseOutputModel(model.backbone=IntermediateLayerGetter).")

    names = list(backbone._modules.keys())
    # Torchvision MobileNetV3 blocks mark channel/downsample boundaries with
    # `_is_cn`; LR-ASPP is built from the same stage decomposition.  Use the
    # final four published-style stages, matching SegAdapter Nb=[1,1,1,1].
    boundaries = []
    for name, module in backbone._modules.items():
        if name == names[-1] or bool(getattr(module, "_is_cn", False)):
            boundaries.append(name)
    # Include the stem boundary if needed, then retain the final four semantic stages.
    if names and names[0] not in boundaries:
        boundaries.insert(0, names[0])
    stage_names = boundaries[-4:]
    if len(stage_names) < 2:
        raise RuntimeError("Could not identify enough MobileNetV3 stages for SegAdapter.")
    channels = _infer_mobile_stage_channels(backbone, stage_names, int(input_size))

    published_mu = (1e-5, 1e-4, 1e-3, 1e-2)
    if len(stage_names) < 4:
        mu_values = published_mu[-len(stage_names):]
    else:
        mu_values = published_mu
    # If a torchvision version exposes >4 selected boundaries, stage_names is already last four.
    for name, mu in zip(stage_names, mu_values):
        original = backbone._modules[name]
        backbone._modules[name] = SegAdapterStage(
            original,
            SegAdapterBlock(
                channels[name], int(num_classes), kernel_size=int(kernel_size),
                ffn_ratio=float(ffn_ratio), mu_init=float(mu),
            ),
        )

    setattr(tv_model, "_segadapter_aux_weight", float(aux_weight))
    setattr(dense_model, "_segadapter_aux_weight", float(aux_weight))
    report = SegAdapterInstallReport(
        stage_names=tuple(stage_names), stage_channels=tuple(channels[n] for n in stage_names),
        mu_init=tuple(float(v) for v in mu_values), kernel_size=int(kernel_size),
        ffn_ratio=float(ffn_ratio), aux_weight=float(aux_weight),
    )
    setattr(dense_model, "_segadapter_report", report.__dict__)
    return report


def collect_segadapter_aux(model: nn.Module, output_size: tuple[int, int]) -> Tensor | None:
    side_maps = []
    for module in model.modules():
        if isinstance(module, SegAdapterStage) and module.last_side_logits is not None:
            side_maps.append(
                F.interpolate(module.last_side_logits, size=output_size, mode="bilinear", align_corners=False)
            )
    if not side_maps:
        return None
    # Paper coarse output Sc is the sum of resized side predictions.
    return torch.stack(side_maps, dim=0).sum(dim=0)


def set_segadapter_trainability(model: nn.Module) -> None:
    """The SegAdapter paper fine-tunes the entire segmentation model."""
    for parameter in model.parameters():
        parameter.requires_grad_(True)


__all__ = [
    "LayerNorm2d", "SegAttention", "SegAdapterBlock", "SegAdapterStage",
    "SegAdapterInstallReport", "apply_segadapter_lraspp", "collect_segadapter_aux",
    "set_segadapter_trainability",
]
