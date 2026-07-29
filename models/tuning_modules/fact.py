"""FacT-TT and FacT-TK for plain Vision Transformers (AAAI 2023).

The implementation follows the paper's shared two-sided factors.  Every ViT
block contributes 12 operation slices: Q, K, V, attention projection, four MLP
expansion slices, and four MLP contraction slices.  TT learns an operation core
for each slice; TK generates those cores from a shared third-order tensor and
operation factors.  The resulting updates are registered as exact weight
parametrizations and can be merged for deployment.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import nn
from torch.nn.utils import parametrize

from .vit_paper_utils import identify_plain_vit


class FacTBank(nn.Module):
    def __init__(self, hidden_dim: int, rank: int, operations: int, variant: str) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.operations = int(operations)
        self.variant = str(variant).lower()
        if self.rank <= 0:
            raise ValueError("FacT rank must be positive.")
        if self.variant not in {"tt", "tk"}:
            raise ValueError("FacT variant must be 'tt' or 'tk'.")
        self.u = nn.Parameter(torch.empty(self.rank, self.hidden_dim))
        self.v = nn.Parameter(torch.empty(self.hidden_dim, self.rank))
        nn.init.xavier_uniform_(self.u)
        nn.init.zeros_(self.v)
        if self.variant == "tt":
            self.operation_cores = nn.Parameter(torch.empty(self.operations, self.rank, self.rank))
            nn.init.xavier_uniform_(self.operation_cores)
            self.register_parameter("core_tensor", None)
            self.register_parameter("operation_factors", None)
        else:
            self.core_tensor = nn.Parameter(torch.empty(self.rank, self.rank, self.rank))
            self.operation_factors = nn.Parameter(torch.empty(self.rank, self.operations))
            nn.init.xavier_uniform_(self.core_tensor)
            nn.init.xavier_uniform_(self.operation_factors)
            self.register_parameter("operation_cores", None)

    def cores(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.to(device=self.u.device, dtype=torch.long)
        if self.variant == "tt":
            return self.operation_cores.index_select(0, indices)
        factors = self.operation_factors.index_select(1, indices)  # r x operations
        return torch.einsum("ijk,ko->oij", self.core_tensor, factors)


class FacTWeightParametrization(nn.Module):
    def __init__(
        self,
        bank: FacTBank,
        operation_indices: Sequence[int],
        *,
        layout: str,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self._bank_ref = weakref.ref(bank)
        self.register_buffer("operation_indices", torch.as_tensor(operation_indices, dtype=torch.long))
        if layout not in {"square", "output_stack", "input_stack"}:
            raise ValueError(f"Unknown FacT layout: {layout}")
        self.layout = layout
        self.scale = float(scale)

    def _bank(self) -> FacTBank:
        bank = self._bank_ref()
        if bank is None:
            raise RuntimeError("The shared FacT bank was released.")
        return bank

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        bank = self._bank()
        cores = bank.cores(self.operation_indices)
        deltas = torch.einsum("ir,ors,sj->oij", bank.v, cores, bank.u)
        if self.layout == "square":
            delta = deltas[0]
        elif self.layout == "output_stack":
            delta = torch.cat(tuple(deltas.unbind(0)), dim=0)
        else:
            delta = torch.cat(tuple(deltas.unbind(0)), dim=1)
        if delta.shape != weight.shape:
            raise RuntimeError(
                f"FacT update shape {tuple(delta.shape)} does not match weight {tuple(weight.shape)}."
            )
        return weight + self.scale * delta.to(dtype=weight.dtype)


@dataclass(frozen=True)
class FacTRecord:
    module_name: str
    parameter_name: str
    operations: tuple[int, ...]
    layout: str


def _register(module: nn.Module, parameter_name: str, bank: FacTBank, indices, layout: str, scale: float) -> None:
    parametrize.register_parametrization(
        module,
        parameter_name,
        FacTWeightParametrization(bank, indices, layout=layout, scale=scale),
        unsafe=False,
    )


def apply_fact(model: nn.Module, *, variant: str = "tk", rank: int = 8, scale: float = 1.0) -> List[FacTRecord]:
    layout = identify_plain_vit(model)
    block_specs = []
    total_operations = 0
    for block in layout.blocks:
        if layout.kind == "timm":
            fc1, fc2 = block.mlp.fc1, block.mlp.fc2
        else:
            fc1, fc2 = block.mlp[0], block.mlp[3]
        if fc1.in_features != layout.hidden_dim or fc2.out_features != layout.hidden_dim:
            raise ValueError("FacT requires the standard ViT MLP input/output width.")
        expansion = int(fc1.out_features // layout.hidden_dim)
        if expansion <= 0 or fc1.out_features != expansion * layout.hidden_dim or fc2.in_features != expansion * layout.hidden_dim:
            raise ValueError("FacT requires an integer ViT MLP expansion ratio.")
        operation_count = 4 + 2 * expansion
        block_specs.append((expansion, total_operations))
        total_operations += operation_count

    bank = FacTBank(layout.hidden_dim, int(rank), total_operations, variant)
    model.add_module("fact_bank", bank)
    records: List[FacTRecord] = []

    for block_index, (block, spec) in enumerate(zip(layout.blocks, block_specs)):
        expansion, base = spec
        qkv_ops = (base, base + 1, base + 2)
        proj_ops = (base + 3,)
        fc1_ops = tuple(range(base + 4, base + 4 + expansion))
        fc2_ops = tuple(range(base + 4 + expansion, base + 4 + 2 * expansion))
        if layout.kind == "timm":
            if not all(hasattr(block, attr) for attr in ("attn", "mlp")):
                raise ValueError("FacT requires standard timm ViT attention and MLP blocks.")
            _register(block.attn.qkv, "weight", bank, qkv_ops, "output_stack", scale)
            _register(block.attn.proj, "weight", bank, proj_ops, "square", scale)
            _register(block.mlp.fc1, "weight", bank, fc1_ops, "output_stack", scale)
            _register(block.mlp.fc2, "weight", bank, fc2_ops, "input_stack", scale)
            prefix = f"blocks.{block_index}"
            records.extend([
                FacTRecord(prefix + ".attn.qkv", "weight", qkv_ops, "output_stack"),
                FacTRecord(prefix + ".attn.proj", "weight", proj_ops, "square"),
                FacTRecord(prefix + ".mlp.fc1", "weight", fc1_ops, "output_stack"),
                FacTRecord(prefix + ".mlp.fc2", "weight", fc2_ops, "input_stack"),
            ])
        else:
            attn = block.self_attention
            if getattr(attn, "in_proj_weight", None) is None:
                raise ValueError("FacT requires a packed torchvision QKV projection.")
            _register(attn, "in_proj_weight", bank, qkv_ops, "output_stack", scale)
            _register(attn.out_proj, "weight", bank, proj_ops, "square", scale)
            _register(block.mlp[0], "weight", bank, fc1_ops, "output_stack", scale)
            _register(block.mlp[3], "weight", bank, fc2_ops, "input_stack", scale)
            prefix = f"encoder.layers.{block_index}"
            records.extend([
                FacTRecord(prefix + ".self_attention", "in_proj_weight", qkv_ops, "output_stack"),
                FacTRecord(prefix + ".self_attention.out_proj", "weight", proj_ops, "square"),
                FacTRecord(prefix + ".mlp.0", "weight", fc1_ops, "output_stack"),
                FacTRecord(prefix + ".mlp.3", "weight", fc2_ops, "input_stack"),
            ])
    model._fact_records = [record.__dict__ for record in records]
    model._fact_variant = str(variant).lower()
    return records


def set_fact_trainability(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        trainable = (
            name.startswith("fact_bank.")
            or name.startswith("head.") or name.startswith("heads.")
            or name.startswith("fc.") or name.startswith("classifier.")
        )
        parameter.requires_grad_(trainable)


def merge_fact_(model: nn.Module) -> int:
    merged = 0
    for module in model.modules():
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is None:
            continue
        for parameter_name in list(parametrizations.keys()):
            stack = parametrizations[parameter_name]
            if any(isinstance(item, FacTWeightParametrization) for item in stack):
                parametrize.remove_parametrizations(module, parameter_name, leave_parametrized=True)
                merged += 1
    if hasattr(model, "fact_bank"):
        delattr(model, "fact_bank")
    return merged
