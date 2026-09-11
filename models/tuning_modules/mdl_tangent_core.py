"""Global Cross-Fitted Reproducibility-Entropy Spectral Tangent Core (G-CREST-TRSO).

The full proposal uses one rule for every eligible pretrained matrix tensor.
The complete calibration loader is split into deterministic odd/even folds.
The pooled gradient supplies singular directions; odd/even agreement and the
finite-sample variance of the mean continuously discount each singular mode.
All layer-mode evidence values form one global distribution. Its parameter-free geometric information dimension determines one model-wide mode budget, which is allocated to the globally strongest modes. Each allocated layer receives one full trainable tangent core, exactly merged for deployment.

There is no task-loss gate, rescue path, positive-gain threshold, manual rank, manual budget, layer list, core switch, or zero-update validation fallback in the full method.  The task head follows one fixed policy: it is fully trained.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Iterable, Iterator, Optional

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize


@dataclass(frozen=True)
class MDLTangentRecord:
    name: str
    shape: tuple[int, ...]
    role: str
    rank: int
    core_mode: str
    core_parameters: int
    basis_values: int
    total_gradient_energy: float
    noise_energy: float
    captured_mean_energy: float
    mdl_zero: float
    mdl_selected: float
    diagonal_bic: float | None
    dense_bic: float | None
    stable_diagonal_coordinates: int = 0
    stable_cross_coordinates: int = 0
    heldout_predictive_gain: float = 0.0
    candidate_modes: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MDLTangentReport:
    method: str
    ablation: str
    calibration_batches: int
    calibration_examples: int
    candidate_tensors: int
    selected_tensors: int
    skipped_tensors: int
    adapter_parameters: int
    proposal_added_parameters: int
    effective_update_coordinates: int
    head_policy: str
    head_trainable_parameters: int
    frozen_basis_values: int
    fallback_reason: str | None
    records: tuple[MDLTangentRecord, ...]
    calibration_mean_loss: float | None = None
    chance_reference_loss: float | None = None
    dense_rescue_activated: bool = False
    zero_added_parameter: bool = False
    # Reviewer-facing allocation diagnostics. These are experiment metadata,
    # not additional trainable hyperparameters of the default proposal.
    global_selected_modes: int = 0
    global_shannon_effective_modes: float = 0.0
    global_numerical_support: int = 0
    global_rule_value: float = 0.0
    mode_count_rule: str = "geometric"
    r_scale: float = 1.0
    fixed_r: int = 0
    candidate_modes: int = 0
    rank_min: int = 0
    rank_median: float = 0.0
    rank_max: int = 0
    calibration_fraction: float = 1.0
    calibration_max_batches: int = 0
    partition_mode: str = "alternating"
    partition_seed: int = 0
    svd_oversampling: int = 0
    svd_power_iterations: int = 2

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["records"] = [record.to_dict() for record in self.records]
        return payload


TRSO_ABLATIONS = (
    "full",
    "diagonal_only",
    "no_sampling_variance",
    "no_crossfit",
    "head_only",
)


@dataclass(frozen=True)
class _Candidate:
    module_name: str
    parameter_name: str
    module: nn.Module
    parameter: Tensor
    is_head: bool

    @property
    def name(self) -> str:
        return f"{self.module_name}.{self.parameter_name}" if self.module_name else self.parameter_name

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.parameter.shape)


@dataclass
class _Statistic:
    count: int
    fold_counts: list[int]
    fold_sums: list[Tensor]
    squared_norm_sum: float

    @classmethod
    def empty(cls, parameter: Tensor) -> "_Statistic":
        shape = tuple(parameter.shape)
        return cls(
            count=0,
            fold_counts=[0, 0],
            fold_sums=[
                torch.zeros(shape, dtype=torch.float32, device="cpu"),
                torch.zeros(shape, dtype=torch.float32, device="cpu"),
            ],
            squared_norm_sum=0.0,
        )

    def update(self, gradient: Tensor, fold: int) -> None:
        value = gradient.detach().to(device="cpu", dtype=torch.float32)
        self.fold_sums[fold].add_(value)
        self.squared_norm_sum += float(value.square().sum().item())
        self.count += 1
        self.fold_counts[fold] += 1

    def mean(self) -> Tensor:
        if self.count <= 0:
            raise RuntimeError("gradient statistic has no observations")
        return (self.fold_sums[0] + self.fold_sums[1]) / float(self.count)

    def fold_mean(self, fold: int) -> Tensor | None:
        count = self.fold_counts[fold]
        if count <= 0:
            return None
        return self.fold_sums[fold] / float(count)

    def noise_sse(self) -> float:
        mean_energy = float(self.mean().double().square().sum().item())
        return max(0.0, self.squared_norm_sum - self.count * mean_energy)


class MDLTangentCoreParametrization(nn.Module):
    """Two-sided tangent update allocated by the global evidence rule."""

    def __init__(
        self,
        left_basis: Tensor,
        right_basis: Tensor,
        original_shape: Iterable[int],
        *,
        coordinate_rows: Tensor | None = None,
        coordinate_columns: Tensor | None = None,
        dense_core: bool | None = None,
    ) -> None:
        super().__init__()
        if left_basis.ndim != 2 or right_basis.ndim != 2:
            raise ValueError("left_basis and right_basis must be matrices")
        if left_basis.shape[1] != right_basis.shape[1]:
            raise ValueError("left/right basis ranks must match")
        rank = int(left_basis.shape[1])
        if rank <= 0:
            raise ValueError("rank must be positive")
        shape = tuple(int(value) for value in original_shape)
        if len(shape) < 2:
            raise ValueError("tangent-core parameters require ndim >= 2")
        flat_input = math.prod(shape[1:])
        if left_basis.shape[0] != shape[0] or right_basis.shape[0] != flat_input:
            raise ValueError("basis dimensions do not match the original tensor")

        # Backward-compatible construction for older checkpoints/tests.
        if coordinate_rows is None or coordinate_columns is None:
            if dense_core:
                rows = torch.arange(rank, device=left_basis.device).repeat_interleave(rank)
                columns = torch.arange(rank, device=left_basis.device).repeat(rank)
            else:
                rows = torch.arange(rank, device=left_basis.device)
                columns = torch.arange(rank, device=left_basis.device)
        else:
            rows = coordinate_rows.detach().to(device=left_basis.device, dtype=torch.long).flatten()
            columns = coordinate_columns.detach().to(device=left_basis.device, dtype=torch.long).flatten()
        if rows.numel() == 0 or rows.numel() != columns.numel():
            raise ValueError("at least one matched tangent coordinate is required")
        if int(rows.min()) < 0 or int(columns.min()) < 0 or int(rows.max()) >= rank or int(columns.max()) >= rank:
            raise ValueError("coordinate index exceeds tangent rank")

        self.original_shape = shape
        self.rank = rank
        self.dense_core = bool(rows.numel() == rank * rank)
        self.register_buffer("left_basis", left_basis.detach().contiguous())
        self.register_buffer("right_basis", right_basis.detach().contiguous())
        self.register_buffer("coordinate_rows", rows.contiguous())
        self.register_buffer("coordinate_columns", columns.contiguous())
        self.core = nn.Parameter(
            torch.zeros(rows.numel(), dtype=left_basis.dtype, device=left_basis.device)
        )

    def delta_matrix(self) -> Tensor:
        left = self.left_basis.index_select(1, self.coordinate_rows)
        right = self.right_basis.index_select(1, self.coordinate_columns)
        return (left * self.core.unsqueeze(0)) @ right.transpose(0, 1)

    def forward(self, parameter: Tensor) -> Tensor:
        return parameter + self.delta_matrix().reshape(self.original_shape).to(dtype=parameter.dtype)

    @property
    def trainable_parameter_count(self) -> int:
        return int(self.core.numel())

    @property
    def basis_value_count(self) -> int:
        return int(self.left_basis.numel() + self.right_basis.numel())


def _default_is_head(name: str) -> bool:
    parts = [part for part in name.lower().split(".") if part]
    modules = parts[:-1]
    if any(part in {"head", "heads", "classifier", "classifiers", "logits", "output"} for part in modules):
        return True
    return len(modules) == 1 and modules[0] in {"fc", "linear"}


def _is_normalization_or_embedding(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            nn.Embedding,
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        ),
    )


def _iter_calibration_candidates(model: nn.Module, is_head: Callable[[str], bool]) -> Iterator[_Candidate]:
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if id(parameter) in seen or not parameter.is_floating_point():
                continue
            seen.add(id(parameter))
            full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            head = bool(is_head(full_name))
            if head:
                # The full task head is trained directly and does not need a
                # gradient-history basis.
                continue
            if parameter.ndim < 2 or _is_normalization_or_embedding(module):
                continue
            yield _Candidate(module_name, parameter_name, module, parameter, False)


def _iter_head_parameters(model: nn.Module, is_head: Callable[[str], bool]) -> Iterator[tuple[str, Tensor]]:
    for name, parameter in model.named_parameters():
        if ".parametrizations." not in name and is_head(name):
            yield name, parameter


def _unpack_batch(batch, device: torch.device | str):
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("calibration batches must provide at least (inputs, targets)")
    inputs, targets = batch[0], batch[1]
    if hasattr(inputs, "to"):
        inputs = inputs.to(device, non_blocking=True)
    if hasattr(targets, "to"):
        targets = targets.to(device, non_blocking=True)
    return inputs, targets


def _truncated_svd(
    matrix: Tensor,
    rank: int,
    *,
    oversampling: int = 0,
    power_iterations: int = 2,
    seed: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Deterministic compact/truncated SVD with reviewer-visible controls.

    ``oversampling=0`` preserves the original automatic oversampling rule.  The
    exact-SVD path is retained for small matrices.  These controls exist so the
    randomized-SVD approximation can be reported and sensitivity-tested without
    changing the default proposal.
    """
    rows, columns = matrix.shape
    resolved = max(1, min(int(rank), rows, columns))
    matrix = matrix.float()
    if min(rows, columns) <= 128 or matrix.numel() <= 262_144 or resolved == min(rows, columns):
        left, singular, right_h = torch.linalg.svd(matrix, full_matrices=False)
        return left[:, :resolved], singular[:resolved], right_h[:resolved]
    automatic = max(4, int(math.ceil(math.log2(max(2, min(rows, columns))))))
    resolved_oversampling = automatic if int(oversampling) <= 0 else int(oversampling)
    sketch = min(min(rows, columns), resolved + max(0, resolved_oversampling))
    generator = torch.Generator(device=matrix.device)
    resolved_seed = rows * 1009 + columns * 9176 + resolved * 53 if seed is None else int(seed)
    generator.manual_seed(resolved_seed)
    omega = torch.randn(columns, sketch, generator=generator, device=matrix.device, dtype=matrix.dtype)
    projected = matrix @ omega
    for _ in range(max(0, int(power_iterations))):
        projected = matrix @ (matrix.transpose(0, 1) @ projected)
    q, _ = torch.linalg.qr(projected, mode="reduced")
    compressed = q.transpose(0, 1) @ matrix
    small_left, singular, right_h = torch.linalg.svd(compressed, full_matrices=False)
    left = q @ small_left
    return left[:, :resolved], singular[:resolved], right_h[:resolved]


def _distributed_reduce_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return torch.device("cpu")
    backend = str(torch.distributed.get_backend()).lower()
    return resolved if "nccl" in backend else torch.device("cpu")


def _synchronize_statistics(
    statistics: dict[str, _Statistic], *, device, batches: int, examples: int,
    loss_sum: float, class_count: int | None,
) -> tuple[int, int, float, int | None]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return batches, examples, loss_sum, class_count
    reduce_device = _distributed_reduce_device(device)
    for statistic in statistics.values():
        for index in range(2):
            value = statistic.fold_sums[index].to(reduce_device)
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            statistic.fold_sums[index] = value.to(device="cpu", dtype=torch.float32)
        metadata = torch.tensor(
            [float(statistic.count), float(statistic.fold_counts[0]), float(statistic.fold_counts[1]), float(statistic.squared_norm_sum)],
            dtype=torch.float64,
            device=reduce_device,
        )
        torch.distributed.all_reduce(metadata, op=torch.distributed.ReduceOp.SUM)
        statistic.count = int(round(float(metadata[0].item())))
        statistic.fold_counts = [int(round(float(metadata[1].item()))), int(round(float(metadata[2].item())))]
        statistic.squared_norm_sum = float(metadata[3].item())
    totals = torch.tensor(
        [float(batches), float(examples), float(loss_sum)],
        dtype=torch.float64, device=reduce_device,
    )
    torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    classes = torch.tensor(float(class_count or 0), dtype=torch.float64, device=reduce_device)
    torch.distributed.all_reduce(classes, op=torch.distributed.ReduceOp.MAX)
    resolved_classes = int(round(float(classes.item()))) or None
    return (
        int(round(float(totals[0].item()))),
        int(round(float(totals[1].item()))),
        float(totals[2].item()),
        resolved_classes,
    )


def _collect_statistics(
    model, candidates, data_loader, loss_fn, *, device, logits_fn, batch_to_device,
    max_batches: int = 0, partition_mode: str = "alternating", partition_seed: int = 0,
):
    original_requires_grad = {id(parameter): parameter.requires_grad for parameter in model.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for candidate in candidates:
        candidate.parameter.requires_grad_(True)
    statistics = {candidate.name: _Statistic.empty(candidate.parameter) for candidate in candidates}
    was_training = model.training
    module_training = {id(module): module.training for module in model.modules()}
    batchnorm_state = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            batchnorm_state.append((
                module,
                module.running_mean.detach().clone() if module.running_mean is not None else None,
                module.running_var.detach().clone() if module.running_var is not None else None,
                module.num_batches_tracked.detach().clone() if module.num_batches_tracked is not None else None,
            ))
    # Calibration must use the same frozen-backbone inference graph used by
    # adaptation/deployment.  Keep BatchNorm in eval mode (running statistics)
    # rather than silently using calibration-batch statistics.  This avoids a
    # reviewer-visible train/eval graph mismatch for CNN backbones.
    model.eval()
    batches = 0
    examples = 0
    loss_sum = 0.0
    class_count: int | None = None
    classification_compatible = True
    mode = str(partition_mode).strip().lower()
    if mode not in {"alternating", "seeded_random"}:
        raise ValueError("partition_mode must be 'alternating' or 'seeded_random'")
    total_batches = int(max_batches) if int(max_batches) > 0 else (len(data_loader) if hasattr(data_loader, "__len__") else 0)
    fold_by_batch: list[int] | None = None
    if mode == "seeded_random" and total_batches > 0:
        generator = torch.Generator(device="cpu").manual_seed(int(partition_seed))
        permutation = torch.randperm(total_batches, generator=generator).tolist()
        fold_by_batch = [0] * total_batches
        midpoint = (total_batches + 1) // 2
        for position, batch_index in enumerate(permutation):
            fold_by_batch[int(batch_index)] = 0 if position < midpoint else 1
    try:
        for batch_index, batch in enumerate(data_loader):
            if int(max_batches) > 0 and batch_index >= int(max_batches):
                break
            inputs, targets = batch_to_device(batch, device)
            model.zero_grad(set_to_none=True)
            logits = logits_fn(model(inputs))
            loss = loss_fn(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite tangent calibration loss")
            batch_examples = int(targets.shape[0]) if getattr(targets, "ndim", 0) else 1
            loss_sum += float(loss.detach().item()) * batch_examples
            if (
                isinstance(logits, Tensor) and logits.ndim == 2
                and getattr(targets, "ndim", 0) == 1
                and targets.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
            ):
                observed_classes = int(logits.shape[-1])
                class_count = observed_classes if class_count in (None, observed_classes) else None
            else:
                classification_compatible = False
            loss.backward()
            if mode == "alternating":
                fold = batches & 1
            elif fold_by_batch is not None and batches < len(fold_by_batch):
                fold = int(fold_by_batch[batches])
            else:
                # Deterministic fallback for loaders without a known length.
                fold = int(((batches * 1103515245 + int(partition_seed) + 12345) >> 16) & 1)
            for candidate in candidates:
                gradient = candidate.parameter.grad
                if gradient is not None:
                    statistics[candidate.name].update(gradient, fold)
            examples += batch_examples
            batches += 1
    finally:
        model.zero_grad(set_to_none=True)
        for module, running_mean, running_var, num_batches_tracked in batchnorm_state:
            if running_mean is not None:
                module.running_mean.copy_(running_mean)
            if running_var is not None:
                module.running_var.copy_(running_var)
            if num_batches_tracked is not None:
                module.num_batches_tracked.copy_(num_batches_tracked)
        for module in model.modules():
            module.train(module_training[id(module)])
        model.train(was_training)
        for parameter in model.parameters():
            parameter.requires_grad_(original_requires_grad[id(parameter)])
    if batches == 0:
        raise RuntimeError("calibration loader produced no batches")
    if not classification_compatible:
        class_count = None
    batches, examples, loss_sum, class_count = _synchronize_statistics(
        statistics, device=device, batches=batches, examples=examples,
        loss_sum=loss_sum, class_count=class_count,
    )
    mean_loss = float(loss_sum / max(1, examples))
    return statistics, batches, examples, mean_loss, class_count


def _cross_fold_gain(first: Tensor, second: Tensor) -> Tensor:
    """Exact symmetric held-out improvement over the zero-coordinate predictor."""
    return 4.0 * first * second - first.square() - second.square()



@dataclass
class _PreparedWeight:
    candidate: _Candidate
    statistic: _Statistic
    left: Tensor | None
    right: Tensor | None
    energy: Tensor
    evidence: Tensor
    total: float
    noise: float
    zero_risk: float
    first_matrix: Tensor | None
    second_matrix: Tensor | None


def _prepare_weight(
    candidate: _Candidate, statistic: _Statistic, *, ablation: str = "full",
    svd_oversampling: int = 0, svd_power_iterations: int = 2, svd_seed: int = 0,
) -> _PreparedWeight:
    """Compute fold-reproducible mode evidence without selecting a local rank."""
    pooled = statistic.mean().float()
    matrix = pooled.reshape(pooled.shape[0], -1)
    first = statistic.fold_mean(0)
    second = statistic.fold_mean(1)
    if ablation == "no_crossfit":
        first = pooled
        second = pooled
    total = float(pooled.double().square().sum().item())
    noise = statistic.noise_sse()
    if first is None or second is None or matrix.numel() == 0 or total == 0.0:
        return _PreparedWeight(
            candidate=candidate, statistic=statistic, left=None, right=None,
            energy=torch.empty(0, dtype=torch.float64),
            evidence=torch.empty(0, dtype=torch.float64),
            total=total, noise=noise, zero_risk=0.0,
            first_matrix=None, second_matrix=None,
        )

    identifiable_rank = min(
        matrix.shape[0], matrix.shape[1], max(1, min(statistic.fold_counts))
    )
    left, singular, right_h = _truncated_svd(
        matrix, identifiable_rank, oversampling=svd_oversampling,
        power_iterations=svd_power_iterations,
        seed=(int(svd_seed) + sum(ord(ch) for ch in candidate.name)) & 0x7FFFFFFF,
    )
    right = right_h.transpose(0, 1)
    first_matrix = first.float().reshape(matrix.shape)
    second_matrix = second.float().reshape(matrix.shape)
    first_modes = torch.diagonal(left.transpose(0, 1) @ first_matrix @ right)
    second_modes = torch.diagonal(left.transpose(0, 1) @ second_matrix @ right)
    mean_coordinates = 0.5 * (first_modes + second_modes)
    fold_disagreement = 0.5 * (first_modes - second_modes)
    if ablation == "no_sampling_variance":
        coordinate_variance = 0.0
    else:
        coordinate_variance = noise / max(
            1, statistic.count * max(1, statistic.count - 1) * matrix.numel()
        )
    eps = torch.finfo(mean_coordinates.dtype).eps
    reliability = mean_coordinates.square() / (
        mean_coordinates.square() + fold_disagreement.square()
        + float(coordinate_variance) + eps
    )
    energy = singular.double().square()
    evidence = energy * reliability.double()
    zero_risk = float(
        first_matrix.double().square().sum().item()
        + second_matrix.double().square().sum().item()
    )
    return _PreparedWeight(
        candidate=candidate, statistic=statistic, left=left, right=right,
        energy=energy, evidence=evidence, total=total, noise=noise,
        zero_risk=zero_risk, first_matrix=first_matrix, second_matrix=second_matrix,
    )


def _mode_count_value(d0: int, d1: float, rule: str) -> float:
    """Reviewer-requested alternatives for the automatic global mode count."""
    rule = str(rule).strip().lower()
    if d0 <= 0:
        return 0.0
    d1 = max(1.0, min(float(d1), float(d0)))
    if rule in {"geometric", "geometric_mean", "default"}:
        return math.sqrt(float(d0) * d1)
    if rule in {"shannon", "d1", "entropy"}:
        return d1
    if rule in {"arithmetic", "arithmetic_mean"}:
        return 0.5 * (float(d0) + d1)
    if rule in {"harmonic", "harmonic_mean"}:
        return (2.0 * float(d0) * d1) / max(float(d0) + d1, 1e-30)
    raise ValueError(f"Unknown mode_count_rule={rule!r}")


def _global_evidence_allocation(
    prepared: list[_PreparedWeight], *, mode_count_rule: str = "geometric",
    r_scale: float = 1.0, fixed_r: int = 0,
) -> tuple[list[Tensor], int, float, float, int, int]:
    """Allocate globally ranked evidence modes with reviewer-facing R controls.

    The paper proposal remains ``mode_count_rule='geometric', r_scale=1,
    fixed_r=0``.  Alternative rules/scales are ablations only.
    """
    locations: list[tuple[int, int]] = []
    values: list[Tensor] = []
    for layer_index, item in enumerate(prepared):
        for mode_index in range(int(item.evidence.numel())):
            locations.append((layer_index, mode_index))
            values.append(item.evidence[mode_index].double())
    selected = [torch.empty(0, dtype=torch.long) for _ in prepared]
    if not values:
        return selected, 0, 0.0, 0.0, 0, 0
    vector = torch.stack(values)
    tiny = torch.finfo(torch.float64).tiny
    total = vector.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0.0:
        return selected, 0, 0.0, 0.0, 0, int(vector.numel())
    probabilities = vector / total.clamp_min(tiny)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(tiny))).sum()
    shannon_modes = float(torch.exp(entropy).item())
    numerical_support = int(torch.count_nonzero(vector > 0).item())
    rule_value = _mode_count_value(numerical_support, shannon_modes, mode_count_rule)
    if int(fixed_r) > 0:
        requested_modes = int(fixed_r)
    else:
        if float(r_scale) <= 0:
            raise ValueError("r_scale must be positive")
        requested_modes = int(math.ceil(rule_value * float(r_scale)))
    global_modes = max(1, min(numerical_support, len(locations), requested_modes))
    top = torch.topk(vector, k=global_modes, largest=True, sorted=False).indices.tolist()
    per_layer: list[list[int]] = [[] for _ in prepared]
    for flat_index in top:
        layer_index, mode_index = locations[int(flat_index)]
        per_layer[layer_index].append(mode_index)
    selected = [
        torch.tensor(sorted(indices), dtype=torch.long)
        if indices else torch.empty(0, dtype=torch.long)
        for indices in per_layer
    ]
    return selected, global_modes, rule_value, shannon_modes, numerical_support, int(vector.numel())


def _finalize_weight(item: _PreparedWeight, selected_modes: Tensor, *, ablation: str):
    candidate = item.candidate
    if item.left is None or item.right is None or selected_modes.numel() == 0:
        record = MDLTangentRecord(
            name=candidate.name, shape=candidate.shape, role="backbone", rank=0,
            core_mode="global_geometric_unallocated", core_parameters=0, basis_values=0,
            total_gradient_energy=item.total, noise_energy=item.noise,
            captured_mean_energy=0.0, mdl_zero=item.zero_risk, mdl_selected=item.zero_risk,
            diagonal_bic=None, dense_bic=None, candidate_modes=int(item.evidence.numel()),
        )
        return record, None, None, None, None
    selected_modes = selected_modes.to(device=item.left.device)
    left_selected = item.left.index_select(1, selected_modes).contiguous()
    right_selected = item.right.index_select(1, selected_modes).contiguous()
    rank = int(selected_modes.numel())
    if ablation == "diagonal_only":
        rows = torch.arange(rank, device=item.left.device, dtype=torch.long)
        columns = rows.clone()
        core_mode = "global_geometric_diagonal_ablation"
    else:
        rows = torch.arange(rank, device=item.left.device).repeat_interleave(rank)
        columns = torch.arange(rank, device=item.left.device).repeat(rank)
        core_mode = (
            "global_geometric_no_crossfit_ablation" if ablation == "no_crossfit"
            else "global_geometric_no_sampling_variance_ablation" if ablation == "no_sampling_variance"
            else "global_geometric_evidence_full_core"
        )
    captured = float(item.energy.index_select(0, selected_modes.cpu()).sum().item())
    weighted = float(item.evidence.index_select(0, selected_modes.cpu()).sum().item())
    selected_risk = max(0.0, item.zero_risk - weighted)
    return MDLTangentRecord(
        name=candidate.name, shape=candidate.shape, role="backbone", rank=rank,
        core_mode=core_mode, core_parameters=int(rows.numel()),
        basis_values=int(left_selected.numel() + right_selected.numel()),
        total_gradient_energy=item.total, noise_energy=item.noise,
        captured_mean_energy=captured, mdl_zero=item.zero_risk,
        mdl_selected=selected_risk, diagonal_bic=None, dense_bic=None,
        stable_diagonal_coordinates=rank,
        stable_cross_coordinates=0 if ablation == "diagonal_only" else rank * rank - rank,
        heldout_predictive_gain=weighted, candidate_modes=int(item.evidence.numel()),
    ), left_selected, right_selected, rows, columns

def _attach(candidate, record, left, right, rows, columns):
    transformation = MDLTangentCoreParametrization(
        left.to(device=candidate.parameter.device, dtype=candidate.parameter.dtype),
        right.to(device=candidate.parameter.device, dtype=candidate.parameter.dtype),
        candidate.shape,
        coordinate_rows=rows,
        coordinate_columns=columns,
    )
    parametrize.register_parametrization(candidate.module, candidate.parameter_name, transformation)


def calibrate_mdl_tangent_core(
    model: nn.Module,
    data_loader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    *,
    device: torch.device | str,
    logits_fn: Optional[Callable[[object], Tensor]] = None,
    is_head: Optional[Callable[[str], bool]] = None,
    batch_to_device: Optional[Callable[[object, torch.device | str], tuple[Tensor, Tensor]]] = None,
    ablation: str = "full",
    mode_count_rule: str = "geometric",
    r_scale: float = 1.0,
    fixed_r: int = 0,
    calibration_fraction: float = 1.0,
    calibration_max_batches: int = 0,
    partition_mode: str = "alternating",
    partition_seed: int = 0,
    svd_oversampling: int = 0,
    svd_power_iterations: int = 2,
    svd_seed: int = 0,
) -> MDLTangentReport:
    """Calibrate and attach the budget-free global tangent-core proposal."""
    if ablation not in TRSO_ABLATIONS:
        raise ValueError(f"Unknown G-CREST-TRSO ablation {ablation!r}; choose from {TRSO_ABLATIONS}")
    logits_fn = logits_fn or (lambda output: output)
    is_head = is_head or _default_is_head
    batch_to_device = batch_to_device or _unpack_batch
    candidates = list(_iter_calibration_candidates(model, is_head))
    head_parameters = list(_iter_head_parameters(model, is_head))
    if not candidates and not head_parameters:
        raise RuntimeError("no eligible backbone or task-head parameters were found")

    fraction = float(calibration_fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError("calibration_fraction must be in (0, 1]")
    effective_max_batches = int(calibration_max_batches)
    if hasattr(data_loader, "__len__"):
        loader_batches = int(len(data_loader))
        fraction_batches = max(1, int(math.ceil(loader_batches * fraction)))
        if loader_batches >= 2:
            fraction_batches = max(2, fraction_batches)
        effective_max_batches = fraction_batches if effective_max_batches <= 0 else min(effective_max_batches, fraction_batches)

    if candidates and ablation != "head_only":
        statistics, batches, examples, calibration_mean_loss, class_count = _collect_statistics(
            model, candidates, data_loader, loss_fn, device=device,
            logits_fn=logits_fn, batch_to_device=batch_to_device,
            max_batches=effective_max_batches, partition_mode=partition_mode,
            partition_seed=partition_seed,
        )
    else:
        batches = len(data_loader) if hasattr(data_loader, "__len__") else 0
        examples = 0
        calibration_mean_loss = None
        class_count = None
        statistics = {}

    chance_reference_loss = math.log(class_count) if class_count and class_count > 1 else None
    dense_rescue = False

    records: list[MDLTangentRecord] = []
    global_modes = 0
    global_effective_modes = 0.0
    global_shannon_modes = 0.0
    global_numerical_support = 0
    candidate_modes = 0
    if ablation == "head_only":
        prepared = []
        selected_by_layer = []
    else:
        prepared = [
            _prepare_weight(
                candidate, statistics[candidate.name], ablation=ablation,
                svd_oversampling=svd_oversampling, svd_power_iterations=svd_power_iterations,
                svd_seed=svd_seed,
            )
            for candidate in candidates
        ]
        selected_by_layer, global_modes, global_effective_modes, global_shannon_modes, global_numerical_support, candidate_modes = _global_evidence_allocation(
            prepared, mode_count_rule=mode_count_rule, r_scale=r_scale, fixed_r=fixed_r,
        )
    for index, candidate in enumerate(candidates):
        if ablation == "head_only":
            record = MDLTangentRecord(
                name=candidate.name, shape=candidate.shape, role="backbone", rank=0,
                core_mode="head_only_ablation", core_parameters=0, basis_values=0,
                total_gradient_energy=0.0, noise_energy=0.0, captured_mean_energy=0.0,
                mdl_zero=0.0, mdl_selected=0.0, diagonal_bic=None, dense_bic=None,
            )
            left = right = rows = columns = None
        else:
            record, left, right, rows, columns = _finalize_weight(
                prepared[index], selected_by_layer[index], ablation=ablation,
            )
        records.append(record)
        if record.rank > 0 and left is not None and right is not None and rows is not None and columns is not None:
            _attach(candidate, record, left, right, rows, columns)

    # The task head is always fully trainable. This is part of the method, not
    # a dataset/backbone-specific option.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, MDLTangentCoreParametrization):
            module.core.requires_grad_(True)
    head_trainable = 0
    for _, parameter in _iter_head_parameters(model, is_head):
        parameter.requires_grad_(True)
        head_trainable += int(parameter.numel())

    adapter_parameters = sum(
        module.trainable_parameter_count for module in model.modules()
        if isinstance(module, MDLTangentCoreParametrization)
    )
    basis_values = sum(
        module.basis_value_count for module in model.modules()
        if isinstance(module, MDLTangentCoreParametrization)
    )
    report = MDLTangentReport(
        method="global_cross_fitted_reproducibility_entropy_spectral_tangent_core",
        ablation=ablation,
        calibration_batches=batches,
        calibration_examples=examples,
        candidate_tensors=len(records),
        selected_tensors=sum(record.rank > 0 for record in records),
        skipped_tensors=sum(record.rank == 0 for record in records),
        adapter_parameters=adapter_parameters,
        proposal_added_parameters=adapter_parameters,
        effective_update_coordinates=adapter_parameters,
        head_policy="full",
        head_trainable_parameters=head_trainable,
        frozen_basis_values=basis_values,
        fallback_reason=None,
        records=tuple(records),
        calibration_mean_loss=calibration_mean_loss,
        chance_reference_loss=chance_reference_loss,
        dense_rescue_activated=dense_rescue,
        zero_added_parameter=False,
        global_selected_modes=int(global_modes),
        global_shannon_effective_modes=float(global_shannon_modes),
        global_numerical_support=int(global_numerical_support),
        global_rule_value=float(global_effective_modes),
        mode_count_rule=str(mode_count_rule),
        r_scale=float(r_scale),
        fixed_r=int(fixed_r),
        candidate_modes=int(candidate_modes),
        rank_min=min((record.rank for record in records if record.rank > 0), default=0),
        rank_median=float(torch.tensor([record.rank for record in records if record.rank > 0], dtype=torch.float32).median().item()) if any(record.rank > 0 for record in records) else 0.0,
        rank_max=max((record.rank for record in records if record.rank > 0), default=0),
        calibration_fraction=fraction,
        calibration_max_batches=int(effective_max_batches),
        partition_mode=str(partition_mode),
        partition_seed=int(partition_seed),
        svd_oversampling=int(svd_oversampling),
        svd_power_iterations=int(svd_power_iterations),
    )
    payload = report.to_dict()
    payload["selection_rule"] = (
        "head_only_ablation" if ablation == "head_only"
        else "pooled_evidence_entropy_ablation" if ablation == "no_crossfit"
        else "diagonal_variance_calibrated_ablation" if ablation == "diagonal_only"
        else "crossfit_without_sampling_variance_ablation" if ablation == "no_sampling_variance"
        else "global_geometric_information_dimension_allocation_full_core"
    )
    payload["head_policy_scores"] = {}
    payload["global_selected_modes"] = int(global_modes)
    payload["global_geometric_information_dimension"] = float(global_effective_modes) if str(mode_count_rule).lower() == "geometric" else None
    payload["global_mode_count_rule_value"] = float(global_effective_modes)
    payload["global_shannon_effective_modes"] = float(global_shannon_modes)
    payload["global_numerical_support"] = int(global_numerical_support)
    payload["candidate_modes"] = int(candidate_modes)
    payload["mode_count_rule"] = str(mode_count_rule)
    payload["r_scale"] = float(r_scale)
    payload["fixed_r"] = int(fixed_r)
    payload["calibration_fraction"] = float(fraction)
    payload["calibration_max_batches"] = int(effective_max_batches)
    payload["partition_mode"] = str(partition_mode)
    payload["partition_seed"] = int(partition_seed)
    payload["svd_oversampling"] = int(svd_oversampling)
    payload["svd_power_iterations"] = int(svd_power_iterations)
    payload["svd_seed"] = int(svd_seed)
    selected_ranks = [int(record.rank) for record in records if record.rank > 0]
    payload["rank_min"] = min(selected_ranks, default=0)
    payload["rank_median"] = float(torch.tensor(selected_ranks, dtype=torch.float32).median().item()) if selected_ranks else 0.0
    payload["rank_max"] = max(selected_ranks, default=0)
    payload["layer_ranks"] = {record.name: int(record.rank) for record in records if record.rank > 0}
    payload["participating_tensor_names"] = [record.name for record in records if record.rank > 0]
    payload["candidate_modes_by_tensor"] = {record.name: int(record.candidate_modes) for record in records}
    if statistics:
        first_statistic = next(iter(statistics.values()))
        payload["calibration_fold_counts"] = [int(value) for value in first_statistic.fold_counts]
    else:
        payload["calibration_fold_counts"] = [0, 0]
    model._mdl_tangent_report = payload  # type: ignore[attr-defined]
    return report


def set_mdl_tangent_trainability(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, MDLTangentCoreParametrization):
            module.core.requires_grad_(True)
    for name, parameter in model.named_parameters():
        if ".parametrizations." not in name and _default_is_head(name):
            parameter.requires_grad_(True)


def iter_mdl_tangent_parametrizations(model: nn.Module):
    for module_name, module in model.named_modules():
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is None:
            continue
        for parameter_name, sequence in list(parametrizations.items()):
            for transformation in sequence:
                if isinstance(transformation, MDLTangentCoreParametrization):
                    yield module_name, module, parameter_name, transformation


def mdl_tangent_parameter_count(model: nn.Module) -> int:
    return sum(transformation.trainable_parameter_count for _, _, _, transformation in iter_mdl_tangent_parametrizations(model))


def mdl_tangent_basis_value_count(model: nn.Module) -> int:
    return sum(transformation.basis_value_count for _, _, _, transformation in iter_mdl_tangent_parametrizations(model))


def mdl_tangent_effective_coordinate_count(model: nn.Module) -> int:
    return mdl_tangent_parameter_count(model)


@torch.no_grad()
def merge_mdl_tangent_cores_(model: nn.Module) -> int:
    targets: list[tuple[nn.Module, str]] = []
    seen: set[tuple[int, str]] = set()
    for _, module, parameter_name, _ in iter_mdl_tangent_parametrizations(model):
        key = (id(module), parameter_name)
        if key not in seen:
            seen.add(key)
            targets.append((module, parameter_name))
    for module, parameter_name in targets:
        parametrize.remove_parametrizations(module, parameter_name, leave_parametrized=True)
    return len(targets)


__all__ = [
    "MDLTangentCoreParametrization",
    "MDLTangentRecord",
    "MDLTangentReport",
    "TRSO_ABLATIONS",
    "calibrate_mdl_tangent_core",
    "iter_mdl_tangent_parametrizations",
    "mdl_tangent_basis_value_count",
    "mdl_tangent_effective_coordinate_count",
    "mdl_tangent_parameter_count",
    "merge_mdl_tangent_cores_",
    "set_mdl_tangent_trainability",
]
