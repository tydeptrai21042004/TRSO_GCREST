"""Sensitivity-aware Parameter-efficient Tuning (ICCV 2023).

Faithful two-stage implementation for plain ViTs:
1. accumulate per-parameter sensitivity as the sum of squared gradients;
2. select the globally most sensitive weight connections under a requested
   backbone budget, then replace sufficiently dense selected matrices by LoRA
   or sequential Adapter tuning when that structured module costs no more than
   the selected unstructured connections.

The desired budget, sensitivity sample count, and structured rank/dimension are
paper hyperparameters and are intentionally explicit rather than hidden.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from .vit_paper_utils import identify_plain_vit


class SPTSparseParametrization(nn.Module):
    def __init__(self, flat_indices: torch.Tensor, shape: torch.Size) -> None:
        super().__init__()
        self.register_buffer("flat_indices", flat_indices.detach().long().flatten())
        self.shape = tuple(int(v) for v in shape)
        self.values = nn.Parameter(torch.zeros(self.flat_indices.numel()))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        delta = weight.new_zeros(weight.numel())
        delta = delta.scatter(0, self.flat_indices.to(weight.device), self.values.to(weight.dtype))
        return weight + delta.reshape(self.shape)


class SPTLoRAParametrization(nn.Module):
    def __init__(self, out_features: int, in_features: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.rank = int(rank)
        self.scale = float(alpha) / float(max(1, rank))
        self.down = nn.Parameter(torch.empty(self.rank, int(in_features)))
        self.up = nn.Parameter(torch.zeros(int(out_features), self.rank))
        nn.init.kaiming_uniform_(self.down, a=5 ** 0.5)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight + (self.up @ self.down).to(weight.dtype) * self.scale


class SPTAdapterLinear(nn.Module):
    """Sequential Adapter replacement used by SPT-Adapter."""
    def __init__(self, base: nn.Linear, bottleneck: int) -> None:
        super().__init__()
        self.base = base
        dim = int(base.out_features)
        self.adapter_down = nn.Linear(dim, int(bottleneck))
        self.adapter_up = nn.Linear(int(bottleneck), dim)
        nn.init.kaiming_uniform_(self.adapter_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y + self.adapter_up(F.gelu(self.adapter_down(y)))


@dataclass(frozen=True)
class SPTCandidate:
    name: str
    module_name: str
    parameter_name: str
    shape: tuple[int, int]
    selected: int
    allocation: str
    trainable_parameters: int


@dataclass(frozen=True)
class SPTReport:
    variant: str
    requested_budget: int
    selected_connections: int
    actual_backbone_trainable_parameters: int
    sensitivity_samples: int
    sensitivity_time_sec: float
    structured_matrices: int
    sparse_matrices: int
    skipped_matrices: int
    records: tuple[SPTCandidate, ...]


def _is_head_name(name: str) -> bool:
    lower = name.lower()
    return (
        lower.startswith("head.") or lower.startswith("heads.")
        or lower.startswith("fc.") or lower.startswith("classifier.")
        or ".head." in lower or ".heads." in lower
    )


def _candidate_parameters(model: nn.Module):
    identify_plain_vit(model)
    rows = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not _is_head_name(module_name + ".weight"):
            rows.append((module_name + ".weight", module_name, module, "weight", module.weight))
        elif isinstance(module, nn.MultiheadAttention) and getattr(module, "in_proj_weight", None) is not None:
            name = module_name + ".in_proj_weight"
            if not _is_head_name(name):
                rows.append((name, module_name, module, "in_proj_weight", module.in_proj_weight))
    # Remove duplicate Linear out_proj entries nested under MultiheadAttention only once.
    unique = []
    seen = set()
    for row in rows:
        key = id(row[-1])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _get_parent(model: nn.Module, module_name: str):
    if not module_name:
        return None, ""
    pieces = module_name.split(".")
    parent = model
    for piece in pieces[:-1]:
        parent = parent._modules[piece]
    return parent, pieces[-1]


def _snapshot_bn(model: nn.Module):
    return {
        id(module): (
            module.running_mean.detach().clone() if getattr(module, "running_mean", None) is not None else None,
            module.running_var.detach().clone() if getattr(module, "running_var", None) is not None else None,
            module.num_batches_tracked.detach().clone() if getattr(module, "num_batches_tracked", None) is not None else None,
        )
        for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)
    }


def _restore_bn(model: nn.Module, state) -> None:
    for module in model.modules():
        values = state.get(id(module))
        if values is None:
            continue
        mean, var, count = values
        if mean is not None:
            module.running_mean.copy_(mean)
        if var is not None:
            module.running_var.copy_(var)
        if count is not None:
            module.num_batches_tracked.copy_(count)


def calibrate_spt(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    variant: str = "lora",
    budget: int = 400_000,
    sensitivity_samples: int = 400,
    rank: int = 8,
    adapter_dim: int = 8,
    alpha: float = 8.0,
    output_path: Optional[str | Path] = None,
) -> dict:
    variant = str(variant).lower()
    if variant not in {"lora", "adapter"}:
        raise ValueError("SPT variant must be 'lora' or 'adapter'.")
    candidates = _candidate_parameters(model)
    if not candidates:
        raise RuntimeError("SPT found no eligible ViT weight matrices.")
    requested_budget = int(budget)
    if requested_budget <= 0:
        raise ValueError("SPT requires a positive desired parameter budget.")
    target_samples = int(sensitivity_samples)
    if target_samples <= 0:
        raise ValueError("SPT sensitivity sample count must be positive.")

    original_requires_grad = {id(parameter): parameter.requires_grad for parameter in model.parameters()}
    was_training = model.training
    bn_state = _snapshot_bn(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for _, _, _, _, parameter in candidates:
        parameter.requires_grad_(True)
    # Algorithm 1 in SPT defines connection sensitivity as a sum of squared
    # *per-sample* gradients. Squaring a minibatch-averaged gradient introduces
    # cross-sample cancellation and changes the ranking with batch size, so each
    # target sample is differentiated independently here. Evaluation mode removes
    # stochastic dropout while the loader still supplies the paper's data
    # augmentations; BatchNorm buffers are restored defensively afterward.
    model.eval()
    sensitivity = {name: torch.zeros_like(parameter, device="cpu", dtype=torch.float32) for name, *_, parameter in candidates}
    criterion = nn.CrossEntropyLoss()
    seen = 0
    started = time.perf_counter()
    for batch in loader:
        if seen >= target_samples:
            break
        images, targets = batch[0].to(device), batch[1].to(device)
        remaining = target_samples - seen
        images = images[:remaining]
        targets = targets[:remaining]
        for sample_index in range(int(images.shape[0])):
            model.zero_grad(set_to_none=True)
            logits = model(images[sample_index:sample_index + 1])
            loss = criterion(logits, targets[sample_index:sample_index + 1].long())
            loss.backward()
            for name, _, _, _, parameter in candidates:
                if parameter.grad is not None:
                    sensitivity[name].add_(parameter.grad.detach().float().cpu().square())
            seen += 1
            if seen >= target_samples:
                break
    elapsed = time.perf_counter() - started
    _restore_bn(model, bn_state)
    model.train(was_training)
    for parameter in model.parameters():
        parameter.requires_grad_(original_requires_grad.get(id(parameter), False))
    model.zero_grad(set_to_none=True)
    if seen == 0:
        raise RuntimeError("SPT calibration loader produced no samples.")

    all_scores = torch.cat([sensitivity[name].flatten() for name, *_ in candidates])
    selected_total = min(requested_budget, int(all_scores.numel()))
    top = torch.topk(all_scores, k=selected_total, largest=True, sorted=False).indices
    offsets = {}
    cursor = 0
    for name, *_, parameter in candidates:
        offsets[name] = (cursor, cursor + parameter.numel())
        cursor += parameter.numel()
    selected_by_name: Dict[str, torch.Tensor] = {}
    for name, *_, parameter in candidates:
        start, end = offsets[name]
        local = top[(top >= start) & (top < end)] - start
        selected_by_name[name] = local.long()

    records: List[SPTCandidate] = []
    actual = 0
    structured = sparse = skipped = 0
    for name, module_name, module, parameter_name, parameter in candidates:
        indices = selected_by_name[name]
        count = int(indices.numel())
        if count == 0:
            skipped += 1
            records.append(SPTCandidate(name, module_name, parameter_name, tuple(parameter.shape), 0, "frozen", 0))
            continue
        out_features, in_features = map(int, parameter.shape)
        allocation = "sparse"
        trainable = count
        if variant == "lora":
            structured_cost = int(rank) * (in_features + out_features)
            if count >= structured_cost:
                parametrize.register_parametrization(
                    module, parameter_name,
                    SPTLoRAParametrization(out_features, in_features, int(rank), float(alpha)),
                    unsafe=False,
                )
                allocation = "lora"
                trainable = structured_cost
                structured += 1
            else:
                parametrize.register_parametrization(
                    module, parameter_name, SPTSparseParametrization(indices, parameter.shape), unsafe=False
                )
                sparse += 1
        else:
            adapter_cost = out_features * int(adapter_dim) + int(adapter_dim) + int(adapter_dim) * out_features + out_features
            parent, child = _get_parent(model, module_name)
            # torchvision MultiheadAttention consumes out_proj.weight directly in
            # its functional kernel instead of calling out_proj.forward(). Wrapping
            # that nested Linear would therefore be incorrect; retain faithful
            # unstructured tuning there. Directly called Linear projections can
            # receive the paper's sequential Adapter after their hidden output.
            directly_called_linear = (
                isinstance(module, nn.Linear)
                and parameter_name == "weight"
                and not isinstance(parent, nn.MultiheadAttention)
            )
            if count >= adapter_cost and directly_called_linear:
                if parent is None:
                    raise RuntimeError("Cannot replace the root module with an SPT Adapter.")
                parent._modules[child] = SPTAdapterLinear(module, int(adapter_dim))
                allocation = "adapter"
                trainable = adapter_cost
                structured += 1
            else:
                parametrize.register_parametrization(
                    module, parameter_name, SPTSparseParametrization(indices, parameter.shape), unsafe=False
                )
                sparse += 1
        actual += trainable
        records.append(SPTCandidate(name, module_name, parameter_name, tuple(parameter.shape), count, allocation, trainable))

    set_spt_trainability(model)
    report = SPTReport(
        variant=variant,
        requested_budget=requested_budget,
        selected_connections=selected_total,
        actual_backbone_trainable_parameters=actual,
        sensitivity_samples=seen,
        sensitivity_time_sec=elapsed,
        structured_matrices=structured,
        sparse_matrices=sparse,
        skipped_matrices=skipped,
        records=tuple(records),
    )
    report_dict = asdict(report)
    model._spt_report = report_dict
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    return report_dict


def set_spt_trainability(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        lower = name.lower()
        trainable = (
            "parametrizations" in lower and (
                lower.endswith(".values") or lower.endswith(".down") or lower.endswith(".up")
            )
            or "adapter_down" in lower or "adapter_up" in lower
            or _is_head_name(name)
        )
        parameter.requires_grad_(trainable)


def merge_spt_(model: nn.Module) -> int:
    """Merge sparse/LoRA SPT parametrizations; sequential adapters remain explicit."""
    merged = 0
    for module in list(model.modules()):
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is None:
            continue
        for parameter_name in list(parametrizations.keys()):
            stack = parametrizations[parameter_name]
            if any(isinstance(item, (SPTSparseParametrization, SPTLoRAParametrization)) for item in stack):
                parametrize.remove_parametrizations(module, parameter_name, leave_parametrized=True)
                merged += 1
    return merged
