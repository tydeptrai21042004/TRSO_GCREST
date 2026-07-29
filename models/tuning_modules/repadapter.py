"""RepAdapter (RepBlock) for plain Vision Transformers.

Clean-room integration of Luo et al., *Towards Efficient Visual Adaption via
Structural Re-parameterization* (2023), following the released RepBlock route:

    x <- x + B(dropout(A(x))) * scale

The residual affine adapter is placed before both the attention input projection
and the first MLP projection of every ViT block.  The second projection is
zero-initialized, so insertion is exactly identity-safe.  At deployment the two
adapter branches are algebraically folded into the frozen projections, leaving
no runtime adapter modules.

Strict benchmark registration is intentionally narrower than the mechanism:
only ViT-B/16 single-label image classification is exposed as an original-paper
comparison.  The generic plain-ViT support below exists for deterministic unit
verification and checkpoint portability, not to broaden the reported paper
scope silently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn

from .vit_paper_utils import identify_plain_vit


class RepAdapter(nn.Module):
    """Official-style 1-D RepAdapter residual affine branch."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 8,
        groups: int = 2,
        scale: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_features = int(in_features)
        hidden_dim = int(hidden_dim)
        groups = int(groups)
        if in_features <= 0 or hidden_dim <= 0:
            raise ValueError("RepAdapter dimensions must be positive.")
        if groups <= 0 or hidden_dim % groups != 0 or in_features % groups != 0:
            raise ValueError(
                "RepAdapter groups must divide both hidden_dim and in_features."
            )
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("RepAdapter dropout must be in [0, 1).")

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.groups = groups
        self.scale = float(scale)
        self.conv_A = nn.Conv1d(in_features, hidden_dim, 1, bias=True)
        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)
        self.dropout = nn.Dropout(float(dropout))
        self.is_repadapter = True

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"RepAdapter expects [batch, tokens, {self.in_features}], got {tuple(x.shape)}"
            )
        residual = x
        x = x.transpose(1, 2)
        x = self.conv_B(self.dropout(self.conv_A(x))) * self.scale + x
        return x.transpose(1, 2).contiguous()

    def equivalent_affine(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``M, c`` such that eval-time output is ``x @ M.T + c``."""
        a = self.conv_A.weight.squeeze(-1)  # [hidden, in]
        b_grouped = self.conv_B.weight.squeeze(-1)  # [out, hidden/groups]
        b = _dense_grouped_weight(b_grouped, self.groups)  # [out, hidden]
        identity = torch.eye(
            self.in_features, device=a.device, dtype=a.dtype
        )
        matrix = identity + self.scale * (b @ a)
        bias = self.scale * (b @ self.conv_A.bias + self.conv_B.bias)
        return matrix, bias


@dataclass(frozen=True)
class RepAdapterRecord:
    name: str
    kind: str


class TimmRepAdapterBlock(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        hidden_dim: int,
        adapter_dim: int,
        groups: int,
        scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block = block
        self.repadapter_attn = RepAdapter(hidden_dim, adapter_dim, groups, scale, dropout)
        self.repadapter_mlp = RepAdapter(hidden_dim, adapter_dim, groups, scale, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.block
        attn_input = self.repadapter_attn(b.norm1(x))
        attn = b.attn(attn_input)
        attn = getattr(b, "ls1", nn.Identity())(attn)
        attn = getattr(b, "drop_path1", getattr(b, "drop_path", nn.Identity()))(attn)
        x = x + attn

        mlp_input = self.repadapter_mlp(b.norm2(x))
        mlp = b.mlp(mlp_input)
        mlp = getattr(b, "ls2", nn.Identity())(mlp)
        mlp = getattr(b, "drop_path2", getattr(b, "drop_path", nn.Identity()))(mlp)
        return x + mlp


class TorchvisionRepAdapterBlock(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        hidden_dim: int,
        adapter_dim: int,
        groups: int,
        scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block = block
        self.repadapter_attn = RepAdapter(hidden_dim, adapter_dim, groups, scale, dropout)
        self.repadapter_mlp = RepAdapter(hidden_dim, adapter_dim, groups, scale, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.block
        torch._assert(x.dim() == 3, f"Expected (batch, seq, hidden), got {x.shape}")
        attn_input = self.repadapter_attn(b.ln_1(x))
        attn, _ = b.self_attention(
            attn_input, attn_input, attn_input, need_weights=False
        )
        x = x + b.dropout(attn)
        mlp_input = self.repadapter_mlp(b.ln_2(x))
        return x + b.mlp(mlp_input)


def _dense_grouped_weight(weight: torch.Tensor, groups: int) -> torch.Tensor:
    """Expand grouped 1x1-convolution weights into an ordinary dense matrix."""
    out_features, in_per_group = weight.shape
    groups = int(groups)
    if out_features % groups != 0:
        raise ValueError("Grouped RepAdapter output width is not divisible by groups.")
    out_per_group = out_features // groups
    dense = weight.new_zeros(out_features, in_per_group * groups)
    for group in range(groups):
        out_slice = slice(group * out_per_group, (group + 1) * out_per_group)
        in_slice = slice(group * in_per_group, (group + 1) * in_per_group)
        dense[out_slice, in_slice] = weight[out_slice]
    return dense


def _compose_into_linear_(linear: nn.Linear, adapter: RepAdapter) -> None:
    matrix, bias = adapter.equivalent_affine()
    weight = linear.weight.detach()
    old_bias = linear.bias.detach() if linear.bias is not None else None
    merged_weight = weight @ matrix
    merged_bias = weight @ bias
    if old_bias is not None:
        merged_bias = merged_bias + old_bias
    with torch.no_grad():
        linear.weight.copy_(merged_weight)
        if linear.bias is None:
            linear.bias = nn.Parameter(merged_bias.clone())
        else:
            linear.bias.copy_(merged_bias)


def _compose_into_multihead_attention_(attention: nn.MultiheadAttention, adapter: RepAdapter) -> None:
    if attention.in_proj_weight is None:
        raise TypeError("RepAdapter merge requires packed MultiheadAttention in_proj_weight.")
    matrix, bias = adapter.equivalent_affine()
    weight = attention.in_proj_weight.detach()
    old_bias = attention.in_proj_bias.detach() if attention.in_proj_bias is not None else None
    merged_weight = weight @ matrix
    merged_bias = weight @ bias
    if old_bias is not None:
        merged_bias = merged_bias + old_bias
    with torch.no_grad():
        attention.in_proj_weight.copy_(merged_weight)
        if attention.in_proj_bias is None:
            attention.in_proj_bias = nn.Parameter(merged_bias.clone())
        else:
            attention.in_proj_bias.copy_(merged_bias)


def _first_linear(module: nn.Module) -> nn.Linear:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return child
    raise TypeError("RepAdapter could not find the first MLP Linear projection.")


def apply_repadapter(
    model: nn.Module,
    *,
    hidden_dim: int = 8,
    groups: int = 2,
    scale: float = 1.0,
    dropout: float = 0.1,
) -> List[RepAdapterRecord]:
    """Insert the paper's RepBlock route in every plain-ViT block."""
    layout = identify_plain_vit(model)
    records: List[RepAdapterRecord] = []
    if layout.kind == "timm":
        for index, block in enumerate(list(model.blocks)):
            model.blocks[index] = TimmRepAdapterBlock(
                block, layout.hidden_dim, hidden_dim, groups, scale, dropout
            )
            records.append(RepAdapterRecord(f"blocks.{index}", "timm"))
    else:
        names = list(model.encoder.layers._modules.keys())
        for name in names:
            block = model.encoder.layers._modules[name]
            model.encoder.layers._modules[name] = TorchvisionRepAdapterBlock(
                block, layout.hidden_dim, hidden_dim, groups, scale, dropout
            )
            records.append(RepAdapterRecord(f"encoder.layers.{name}", "torchvision"))
    model._repadapter_records = [record.__dict__ for record in records]
    return records


def set_repadapter_trainability(model: nn.Module) -> None:
    """Freeze the pretrained ViT; train RepAdapter branches and downstream head."""
    for name, parameter in model.named_parameters():
        lower = name.lower()
        trainable = (
            "repadapter_" in lower
            or name.startswith("head.")
            or name.startswith("heads.")
            or name.startswith("classifier.")
            or name.startswith("fc.")
            or ".head." in name
            or ".heads." in name
        )
        parameter.requires_grad_(trainable)


def merge_repadapter_(model: nn.Module) -> int:
    """Fold every RepAdapter into ViT projections and remove runtime wrappers."""
    layout = identify_plain_vit(model)
    merged = 0
    if layout.kind == "timm":
        for index, wrapper in enumerate(list(model.blocks)):
            if not isinstance(wrapper, TimmRepAdapterBlock):
                continue
            block = wrapper.block
            qkv = getattr(getattr(block, "attn", None), "qkv", None)
            fc1 = getattr(getattr(block, "mlp", None), "fc1", None)
            if not isinstance(qkv, nn.Linear) or not isinstance(fc1, nn.Linear):
                raise TypeError("RepAdapter merge requires timm ViT qkv and mlp.fc1 Linear layers.")
            _compose_into_linear_(qkv, wrapper.repadapter_attn)
            _compose_into_linear_(fc1, wrapper.repadapter_mlp)
            model.blocks[index] = block
            merged += 2
    else:
        for name in list(model.encoder.layers._modules.keys()):
            wrapper = model.encoder.layers._modules[name]
            if not isinstance(wrapper, TorchvisionRepAdapterBlock):
                continue
            block = wrapper.block
            if not isinstance(block.self_attention, nn.MultiheadAttention):
                raise TypeError("RepAdapter merge requires torchvision MultiheadAttention.")
            _compose_into_multihead_attention_(block.self_attention, wrapper.repadapter_attn)
            _compose_into_linear_(_first_linear(block.mlp), wrapper.repadapter_mlp)
            model.encoder.layers._modules[name] = block
            merged += 2
    if hasattr(model, "_repadapter_records"):
        delattr(model, "_repadapter_records")
    return merged


__all__ = [
    "RepAdapter",
    "RepAdapterRecord",
    "TimmRepAdapterBlock",
    "TorchvisionRepAdapterBlock",
    "apply_repadapter",
    "set_repadapter_trainability",
    "merge_repadapter_",
]
