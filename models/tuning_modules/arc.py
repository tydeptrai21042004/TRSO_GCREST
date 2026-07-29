"""Adapter Re-Composing (ARC) for plain Vision Transformers.

Clean-room reproduction of the main ARC configuration from:
    Dong et al., "Efficient Adaptation of Large Vision Transformer via
    Adapter Re-Composing", NeurIPS 2023.

The strict route implemented here follows the paper and official ViT release:
- one shared down-projection for all attention adapters;
- one independent shared down-projection for all FFN adapters;
- symmetric up-projections given by the transposes of the shared projections;
- one layer-specific re-scaling vector and output bias per adapter;
- sequential insertion before both MHA and FFN;
- frozen pretrained weights, with only ARC parameters and the task head trained;
- exact deployment folding into attention input projections and the first FFN
  projection when ``merge_arc_`` is requested.

Only plain timm/torchvision ViTs are accepted. Hierarchical Transformers are
rejected rather than silently approximated.
"""
from __future__ import annotations

from typing import List

import torch
from torch import nn

from .vit_paper_utils import identify_plain_vit


class ARCProjectionBank(nn.Module):
    """Cross-layer shared symmetric bottleneck projections."""

    def __init__(self, hidden_dim: int, adapter_dim: int) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        adapter_dim = int(adapter_dim)
        if hidden_dim <= 0 or adapter_dim <= 0:
            raise ValueError("ARC hidden_dim and adapter_dim must be positive.")
        if adapter_dim >= hidden_dim:
            raise ValueError(
                f"ARC requires a bottleneck adapter_dim < hidden_dim; got {adapter_dim} >= {hidden_dim}."
            )
        self.hidden_dim = hidden_dim
        self.adapter_dim = adapter_dim
        self.att_down_projection = nn.Parameter(torch.empty(hidden_dim, adapter_dim))
        self.mlp_down_projection = nn.Parameter(torch.empty(hidden_dim, adapter_dim))
        nn.init.xavier_uniform_(self.att_down_projection)
        nn.init.xavier_uniform_(self.mlp_down_projection)


class ARCAdapter(nn.Module):
    """Layer-specific re-composition coefficients over a shared projection."""

    def __init__(
        self,
        hidden_dim: int,
        adapter_dim: int,
        *,
        dropout: float = 0.1,
        position: str,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        adapter_dim = int(adapter_dim)
        dropout = float(dropout)
        if position not in {"att", "mlp"}:
            raise ValueError("ARC position must be 'att' or 'mlp'.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("ARC dropout must be in [0, 1).")
        self.hidden_dim = hidden_dim
        self.adapter_dim = adapter_dim
        self.position = position
        self.adapter_rescale = nn.Parameter(torch.empty(1, adapter_dim))
        self.adapter_bias = nn.Parameter(torch.empty(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.is_arc_adapter = True

        # Matches the released ARC ViT implementation exactly:
        # attention re-scaling starts at zero; FFN re-scaling uses Xavier.
        if position == "att":
            nn.init.zeros_(self.adapter_rescale)
        else:
            nn.init.xavier_uniform_(self.adapter_rescale)
        nn.init.zeros_(self.adapter_bias)

    def adaptation_matrix(self, down_projection: torch.Tensor) -> torch.Tensor:
        """Return W_down diag(c_l) W_down^T in row-vector convention."""
        scaled_down = down_projection * self.adapter_rescale
        return scaled_down @ down_projection.transpose(0, 1)

    def forward(self, x: torch.Tensor, down_projection: torch.Tensor) -> torch.Tensor:
        projected = torch.matmul(x, down_projection * self.adapter_rescale)
        projected = self.dropout(projected)
        adapter_output = torch.matmul(projected, down_projection.transpose(0, 1))
        adapter_output = adapter_output + self.adapter_bias
        return x + adapter_output


class TimmARCBlock(nn.Module):
    """ARC wrapper for a timm-style plain ViT block."""

    def __init__(self, block: nn.Module, bank: ARCProjectionBank, hidden_dim: int, adapter_dim: int, dropout: float) -> None:
        super().__init__()
        self.block = block
        # Avoid registering the same shared bank once per block. The bank is
        # registered exactly once at model.arc_projection_bank.
        object.__setattr__(self, "_arc_bank", bank)
        self.att_adapter = ARCAdapter(hidden_dim, adapter_dim, dropout=dropout, position="att")
        self.mlp_adapter = ARCAdapter(hidden_dim, adapter_dim, dropout=dropout, position="mlp")
        self.arc_merged = False

    @property
    def bank(self) -> ARCProjectionBank:
        return object.__getattribute__(self, "_arc_bank")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.block
        norm1 = b.norm1(x)
        if not self.arc_merged:
            norm1 = self.att_adapter(norm1, self.bank.att_down_projection)
        attn = b.attn(norm1)
        attn = getattr(b, "ls1", nn.Identity())(attn)
        attn = getattr(b, "drop_path1", getattr(b, "drop_path", nn.Identity()))(attn)
        x = x + attn

        norm2 = b.norm2(x)
        if not self.arc_merged:
            norm2 = self.mlp_adapter(norm2, self.bank.mlp_down_projection)
        mlp = b.mlp(norm2)
        mlp = getattr(b, "ls2", nn.Identity())(mlp)
        mlp = getattr(b, "drop_path2", getattr(b, "drop_path", nn.Identity()))(mlp)
        return x + mlp


class TorchvisionARCBlock(nn.Module):
    """ARC wrapper for torchvision VisionTransformer EncoderBlock."""

    def __init__(self, block: nn.Module, bank: ARCProjectionBank, hidden_dim: int, adapter_dim: int, dropout: float) -> None:
        super().__init__()
        self.block = block
        object.__setattr__(self, "_arc_bank", bank)
        self.att_adapter = ARCAdapter(hidden_dim, adapter_dim, dropout=dropout, position="att")
        self.mlp_adapter = ARCAdapter(hidden_dim, adapter_dim, dropout=dropout, position="mlp")
        self.arc_merged = False

    @property
    def bank(self) -> ARCProjectionBank:
        return object.__getattribute__(self, "_arc_bank")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        torch._assert(input.dim() == 3, f"Expected (batch, seq, hidden), got {input.shape}")
        b = self.block
        x = b.ln_1(input)
        if not self.arc_merged:
            x = self.att_adapter(x, self.bank.att_down_projection)
        x, _ = b.self_attention(x, x, x, need_weights=False)
        x = b.dropout(x)
        x = x + input

        y = b.ln_2(x)
        if not self.arc_merged:
            y = self.mlp_adapter(y, self.bank.mlp_down_projection)
        y = b.mlp(y)
        return x + y



def _first_linear(module: nn.Module) -> nn.Linear:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return child
    raise TypeError(f"ARC could not find the first FFN linear projection in {type(module).__name__}.")


def _merge_linear_input_(linear: nn.Linear, adapter: ARCAdapter, down_projection: torch.Tensor) -> None:
    """Fold x -> ARC(x) -> Linear into a single Linear exactly in eval mode."""
    with torch.no_grad():
        delta = adapter.adaptation_matrix(down_projection)
        identity = torch.eye(delta.shape[0], device=delta.device, dtype=delta.dtype)
        transform = identity + delta
        old_weight = linear.weight.detach().clone()
        linear.weight.copy_(old_weight @ transform.transpose(0, 1))
        bias_shift = old_weight @ adapter.adapter_bias
        if linear.bias is None:
            linear.bias = nn.Parameter(bias_shift.clone(), requires_grad=False)
        else:
            linear.bias.add_(bias_shift)


def _merge_torchvision_attention_(attention: nn.MultiheadAttention, adapter: ARCAdapter, down_projection: torch.Tensor) -> None:
    if attention.in_proj_weight is None:
        raise TypeError("ARC torchvision merge requires packed MultiheadAttention in_proj_weight.")
    with torch.no_grad():
        delta = adapter.adaptation_matrix(down_projection)
        identity = torch.eye(delta.shape[0], device=delta.device, dtype=delta.dtype)
        transform = identity + delta
        old_weight = attention.in_proj_weight.detach().clone()
        attention.in_proj_weight.copy_(old_weight @ transform.transpose(0, 1))
        bias_shift = old_weight @ adapter.adapter_bias
        if attention.in_proj_bias is None:
            attention.in_proj_bias = nn.Parameter(bias_shift.clone(), requires_grad=False)
        else:
            attention.in_proj_bias.add_(bias_shift)


def _merge_timm_attention_(attention: nn.Module, adapter: ARCAdapter, down_projection: torch.Tensor) -> str:
    qkv = getattr(attention, "qkv", None)
    if isinstance(qkv, nn.Linear):
        _merge_linear_input_(qkv, adapter, down_projection)
        return "attn.qkv"
    # Conservative support for implementations with explicit q/k/v linears.
    projections = [getattr(attention, name, None) for name in ("q", "k", "v")]
    if all(isinstance(module, nn.Linear) for module in projections):
        for module in projections:
            _merge_linear_input_(module, adapter, down_projection)
        return "attn.{q,k,v}"
    raise TypeError(
        "ARC timm merge requires attn.qkv or explicit attn.q/attn.k/attn.v Linear projections."
    )


def apply_arc(
    model: nn.Module,
    *,
    adapter_dim: int = 50,
    dropout: float = 0.1,
) -> List[str]:
    """Insert the main ARC configuration before MHA and FFN in every ViT block."""
    if hasattr(model, "arc_projection_bank"):
        raise ValueError("ARC is already applied to this model.")
    layout = identify_plain_vit(model)
    adapter_dim = int(adapter_dim)
    dropout = float(dropout)
    bank = ARCProjectionBank(layout.hidden_dim, adapter_dim)
    model.add_module("arc_projection_bank", bank)

    records: List[str] = []
    if layout.kind == "timm":
        for index, block in enumerate(list(model.blocks)):
            model.blocks[index] = TimmARCBlock(block, bank, layout.hidden_dim, adapter_dim, dropout)
            records.append(f"blocks.{index}")
    else:
        names = list(model.encoder.layers._modules.keys())
        for name in names:
            block = model.encoder.layers._modules[name]
            model.encoder.layers._modules[name] = TorchvisionARCBlock(
                block, bank, layout.hidden_dim, adapter_dim, dropout
            )
            records.append(f"encoder.layers.{name}")
    model._arc_records = tuple(records)
    model._arc_adapter_dim = adapter_dim
    model._arc_dropout = dropout
    return records


def set_arc_trainability(model: nn.Module) -> None:
    """Freeze the pretrained ViT and train only ARC parameters plus task head."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        lower = name.lower()
        trainable = (
            name.startswith("arc_projection_bank.")
            or ".att_adapter." in lower
            or ".mlp_adapter." in lower
            or name.startswith("head.")
            or name.startswith("heads.")
            or name.startswith("classifier.")
            or name.startswith("fc.")
            or ".head." in name
            or ".heads." in name
        )
        parameter.requires_grad_(trainable)


def merge_arc_(model: nn.Module) -> int:
    """Exactly fold all ARC adapters into frozen ViT projections for inference.

    Returns the number of folded ARC branches (two per Transformer block).
    The model must be in evaluation mode because ARC dropout is inactive only in
    eval mode; the algebraic folding itself is exact.
    """
    if model.training:
        raise RuntimeError("Call model.eval() before merge_arc_ so ARC dropout is disabled.")
    bank = getattr(model, "arc_projection_bank", None)
    if not isinstance(bank, ARCProjectionBank):
        return 0

    merged = 0
    for module in model.modules():
        if isinstance(module, TimmARCBlock):
            if module.arc_merged:
                continue
            _merge_timm_attention_(module.block.attn, module.att_adapter, bank.att_down_projection)
            _merge_linear_input_(_first_linear(module.block.mlp), module.mlp_adapter, bank.mlp_down_projection)
            module.arc_merged = True
            merged += 2
        elif isinstance(module, TorchvisionARCBlock):
            if module.arc_merged:
                continue
            _merge_torchvision_attention_(
                module.block.self_attention, module.att_adapter, bank.att_down_projection
            )
            _merge_linear_input_(_first_linear(module.block.mlp), module.mlp_adapter, bank.mlp_down_projection)
            module.arc_merged = True
            merged += 2
    return merged


__all__ = [
    "ARCProjectionBank",
    "ARCAdapter",
    "TimmARCBlock",
    "TorchvisionARCBlock",
    "apply_arc",
    "set_arc_trainability",
    "merge_arc_",
]
