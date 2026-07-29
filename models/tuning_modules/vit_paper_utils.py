"""Shared utilities for strict Vision-Transformer paper baselines.

The helpers intentionally recognize only plain ViT encoders from torchvision or
`timm`.  Hierarchical Transformers are rejected rather than silently mapped to
a different mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class ViTLayout:
    kind: str
    hidden_dim: int
    blocks: Sequence[nn.Module]
    num_prefix_tokens: int


def identify_plain_vit(model: nn.Module) -> ViTLayout:
    # timm VisionTransformer / DeiT-style models
    if all(hasattr(model, attr) for attr in ("patch_embed", "blocks", "pos_embed", "cls_token")):
        blocks = list(model.blocks)
        if not blocks:
            raise ValueError("The timm ViT has no Transformer blocks.")
        hidden = int(getattr(model, "embed_dim", model.pos_embed.shape[-1]))
        prefix = int(getattr(model, "num_prefix_tokens", 1))
        return ViTLayout("timm", hidden, blocks, prefix)

    # torchvision VisionTransformer
    if all(hasattr(model, attr) for attr in ("conv_proj", "encoder", "class_token", "heads")):
        layers = getattr(model.encoder, "layers", None)
        if layers is None:
            raise ValueError("The torchvision ViT encoder has no layers container.")
        blocks = list(layers.children())
        if not blocks:
            raise ValueError("The torchvision ViT has no Transformer blocks.")
        hidden = int(model.class_token.shape[-1])
        return ViTLayout("torchvision", hidden, blocks, 1)

    raise ValueError(
        "This paper baseline requires a plain Vision Transformer with explicit "
        "patch tokens and Transformer blocks (timm or torchvision ViT)."
    )


def classifier_module(model: nn.Module) -> nn.Module:
    for attr in ("head", "heads", "fc", "classifier"):
        module = getattr(model, attr, None)
        if isinstance(module, nn.Module):
            return module
    raise ValueError("No downstream classifier module was found.")


def set_only_named_components_trainable(
    model: nn.Module,
    *,
    component_prefixes: Iterable[str],
    train_head: bool = True,
) -> None:
    prefixes = tuple(str(value) for value in component_prefixes)
    for name, parameter in model.named_parameters():
        is_component = any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
        is_head = train_head and (
            name.startswith("head.") or name.startswith("heads.")
            or name.startswith("fc.") or name.startswith("classifier.")
            or ".head." in name or ".heads." in name
        )
        parameter.requires_grad_(is_component or is_head)


def square_patch_grid(token_count: int) -> int:
    side = int(round(float(token_count) ** 0.5))
    if side * side != int(token_count):
        raise ValueError(
            f"ConvPass requires a square patch-token grid; received {token_count} tokens."
        )
    return side


def timm_embed_tokens(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Mirror timm VisionTransformer token embedding while leaving prompts external."""
    x = model.patch_embed(images)
    if x.ndim == 4:
        x = x.flatten(2).transpose(1, 2)
    batch = x.shape[0]
    prefix_tokens = []
    if getattr(model, "cls_token", None) is not None:
        prefix_tokens.append(model.cls_token.expand(batch, -1, -1))
    if getattr(model, "reg_token", None) is not None:
        prefix_tokens.append(model.reg_token.expand(batch, -1, -1))
    elif getattr(model, "reg_tokens", None) is not None:
        prefix_tokens.append(model.reg_tokens.expand(batch, -1, -1))
    prefix = torch.cat(prefix_tokens, dim=1) if prefix_tokens else x[:, :0]

    if bool(getattr(model, "no_embed_class", False)):
        x = x + model.pos_embed
        x = torch.cat((prefix, x), dim=1)
    else:
        x = torch.cat((prefix, x), dim=1)
        x = x + model.pos_embed
    pos_drop = getattr(model, "pos_drop", nn.Identity())
    patch_drop = getattr(model, "patch_drop", nn.Identity())
    norm_pre = getattr(model, "norm_pre", nn.Identity())
    return norm_pre(patch_drop(pos_drop(x)))


def timm_forward_head(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_head"):
        return model.forward_head(tokens)
    pooled = tokens[:, 0]
    norm = getattr(model, "norm", nn.Identity())
    pooled = norm(pooled)
    return classifier_module(model)(pooled)


def torchvision_embed_tokens(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    batch = images.shape[0]
    x = model._process_input(images)
    cls = model.class_token.expand(batch, -1, -1)
    x = torch.cat([cls, x], dim=1)
    x = x + model.encoder.pos_embedding
    return model.encoder.dropout(x)


def torchvision_forward_head(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    x = model.encoder.ln(tokens)
    return model.heads(x[:, 0])
