"""ConvPass for plain Vision Transformers (clean-room reproduction).

The paper adds convolutional bypasses in parallel to the frozen attention and
MLP branches.  Patch tokens are reshaped to their 2-D grid, processed by a
3x3 convolution, and projected back; the class token is processed through the
same bottleneck convolution at spatial size 1x1.
"""
from __future__ import annotations

import copy
from typing import List

import torch
from torch import nn

from .vit_paper_utils import identify_plain_vit, square_patch_grid


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ConvPassAdapter(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_dim, int(bottleneck))
        self.act1 = QuickGELU()
        self.conv = nn.Conv2d(int(bottleneck), int(bottleneck), 3, padding=1)
        self.act2 = QuickGELU()
        self.dropout = nn.Dropout(float(dropout))
        self.up = nn.Linear(int(bottleneck), hidden_dim)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        with torch.no_grad():
            center = self.conv.kernel_size[0] // 2
            eye = torch.eye(int(bottleneck), dtype=self.conv.weight.dtype)
            self.conv.weight[:, :, center, center].copy_(eye)

    def forward(self, tokens: torch.Tensor, *, prefix_tokens: int = 1) -> torch.Tensor:
        projected = self.act1(self.down(tokens))
        prefix = projected[:, :prefix_tokens]
        patches = projected[:, prefix_tokens:]
        side = square_patch_grid(patches.shape[1])
        patches = patches.transpose(1, 2).reshape(tokens.shape[0], -1, side, side)
        patches = self.conv(patches).flatten(2).transpose(1, 2)
        if prefix_tokens:
            prefix = prefix.transpose(1, 2).unsqueeze(-1)
            prefix = self.conv(prefix).squeeze(-1).transpose(1, 2)
            projected = torch.cat((prefix, patches), dim=1)
        else:
            projected = patches
        return self.up(self.dropout(self.act2(projected)))


class TimmConvPassBlock(nn.Module):
    def __init__(self, block: nn.Module, hidden_dim: int, bottleneck: int, scale: float, dropout: float, attn_only: bool):
        super().__init__()
        self.block = block
        self.scale = float(scale)
        self.attn_adapter = ConvPassAdapter(hidden_dim, bottleneck, dropout)
        self.mlp_adapter = None if attn_only else ConvPassAdapter(hidden_dim, bottleneck, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.block
        norm1 = b.norm1(x)
        attn = b.attn(norm1)
        attn = getattr(b, "ls1", nn.Identity())(attn)
        attn = getattr(b, "drop_path1", getattr(b, "drop_path", nn.Identity()))(attn)
        bypass = self.attn_adapter(norm1, prefix_tokens=1)
        bypass = getattr(b, "drop_path1", getattr(b, "drop_path", nn.Identity()))(bypass)
        x = x + attn + self.scale * bypass
        norm2 = b.norm2(x)
        mlp = b.mlp(norm2)
        mlp = getattr(b, "ls2", nn.Identity())(mlp)
        mlp = getattr(b, "drop_path2", getattr(b, "drop_path", nn.Identity()))(mlp)
        if self.mlp_adapter is not None:
            bypass = self.mlp_adapter(norm2, prefix_tokens=1)
            bypass = getattr(b, "drop_path2", getattr(b, "drop_path", nn.Identity()))(bypass)
            mlp = mlp + self.scale * bypass
        return x + mlp


class TorchvisionConvPassBlock(nn.Module):
    def __init__(self, block: nn.Module, hidden_dim: int, bottleneck: int, scale: float, dropout: float, attn_only: bool):
        super().__init__()
        self.block = block
        self.scale = float(scale)
        self.attn_adapter = ConvPassAdapter(hidden_dim, bottleneck, dropout)
        self.mlp_adapter = None if attn_only else ConvPassAdapter(hidden_dim, bottleneck, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.block
        norm1 = b.ln_1(x)
        attn, _ = b.self_attention(norm1, norm1, norm1, need_weights=False)
        x = x + b.dropout(attn) + self.scale * self.attn_adapter(norm1, prefix_tokens=1)
        norm2 = b.ln_2(x)
        mlp = b.mlp(norm2)
        if self.mlp_adapter is not None:
            mlp = mlp + self.scale * self.mlp_adapter(norm2, prefix_tokens=1)
        return x + mlp


def apply_convpass(
    model: nn.Module,
    *,
    bottleneck: int = 8,
    scale: float = 1.0,
    dropout: float = 0.1,
    attn_only: bool = False,
) -> List[str]:
    layout = identify_plain_vit(model)
    if layout.num_prefix_tokens != 1:
        raise ValueError("Strict ConvPass reproduction currently requires one class prefix token.")
    records: List[str] = []
    if layout.kind == "timm":
        for index, block in enumerate(list(model.blocks)):
            model.blocks[index] = TimmConvPassBlock(block, layout.hidden_dim, bottleneck, scale, dropout, attn_only)
            records.append(f"blocks.{index}")
    else:
        names = list(model.encoder.layers._modules.keys())
        for index, name in enumerate(names):
            block = model.encoder.layers._modules[name]
            model.encoder.layers._modules[name] = TorchvisionConvPassBlock(
                block, layout.hidden_dim, bottleneck, scale, dropout, attn_only
            )
            records.append(f"encoder.layers.{name}")
    model._convpass_records = records
    return records


def set_convpass_trainability(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        lower = name.lower()
        trainable = (
            "attn_adapter" in lower or "mlp_adapter" in lower
            or name.startswith("head.") or name.startswith("heads.")
            or name.startswith("fc.") or name.startswith("classifier.")
        )
        parameter.requires_grad_(trainable)
