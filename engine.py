"""Shared training/evaluation loops for classification, multi-label, regression."""
from __future__ import annotations

import math
import time
from contextlib import nullcontext
from typing import Iterable, Optional

import torch
try:
    from timm.data import Mixup
    from timm.utils import ModelEma, accuracy
except Exception:
    from compat.timm_compat import Mixup, ModelEma, accuracy

import utils
from task_registry import (
    TASK_DEPTH_ESTIMATION, TASK_MULTILABEL, TASK_OBJECT_DETECTION,
    TASK_REGRESSION, TASK_SEMANTIC_SEGMENTATION, TASK_SINGLE_LABEL,
)


def _amp_context(device: torch.device, enabled: bool):
    enabled = bool(enabled and device.type == "cuda" and torch.cuda.is_available())
    return torch.amp.autocast(device_type="cuda") if enabled else nullcontext()


def _set_frozen_batchnorm_eval(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            affine_trainable = any(
                parameter is not None and parameter.requires_grad
                for parameter in (module.weight, module.bias)
            )
            if not affine_trainable:
                module.eval()


def _extract_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if isinstance(output, dict):
        for key in ("logits", "out", "pred"):
            if isinstance(output.get(key), torch.Tensor):
                return output[key]
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def _prepare_target(target: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type in {TASK_SINGLE_LABEL, TASK_SEMANTIC_SEGMENTATION}:
        return target.long()
    return target.float()


def _move_detection_targets(targets, device):
    return [
        {key: (value.to(device, non_blocking=True) if hasattr(value, "to") else value) for key, value in target.items()}
        for target in targets
    ]



def _apply_post_optimizer_constraints(model: torch.nn.Module) -> None:
    # G-CREST-TRSO optimizes explicit tangent-core parameters, so the backbone
    # displacement is constrained by construction and needs no post-step projection.
    return None


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    wandb_logger=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch=None,
    update_freq=None,
    use_amp: bool = False,
    task_type: str = TASK_SINGLE_LABEL,
):
    model.train(True)
    _set_frozen_batchnorm_eval(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header, print_freq = f"Epoch: [{epoch}]", 10

    update_freq = int(update_freq or 1)
    start_steps = int(start_steps or 0)
    total_microbatches = len(data_loader)
    num_training_steps_per_epoch = int(
        num_training_steps_per_epoch
        or math.ceil(total_microbatches / update_freq)
    )

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    epoch_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, targets = batch[0], batch[1]
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            break
        iteration = start_steps + step
        window_start = step * update_freq
        window_size = min(update_freq, total_microbatches - window_start)
        is_update_step = (data_iter_step + 1 == total_microbatches) or (
            (data_iter_step + 1) % update_freq == 0
        )

        if data_iter_step % update_freq == 0:
            if lr_schedule_values is not None:
                for group in optimizer.param_groups:
                    group["lr"] = lr_schedule_values[iteration] * float(group.get("lr_scale", 1.0))
            if wd_schedule_values is not None:
                for group in optimizer.param_groups:
                    if group.get("weight_decay", 0) > 0:
                        group["weight_decay"] = wd_schedule_values[iteration]

        if task_type == TASK_OBJECT_DETECTION:
            samples = [image.to(device, non_blocking=True) for image in samples]
            targets = _move_detection_targets(targets, device)
            if mixup_fn is not None:
                raise ValueError("Mixup/CutMix is not defined for object detection.")
            with _amp_context(device, use_amp):
                loss_output = model(samples, targets)
                if not isinstance(loss_output, dict):
                    raise TypeError("Detection models must return a loss dictionary during training.")
                loss = sum(value for value in loss_output.values())
            output = None
            batch_size = len(samples)
        else:
            samples = samples.to(device, non_blocking=True)
            targets = _prepare_target(targets.to(device, non_blocking=True), task_type)
            if mixup_fn is not None:
                if task_type != TASK_SINGLE_LABEL:
                    raise ValueError("Mixup/CutMix is supported only for single-label classification.")
                samples, targets = mixup_fn(samples, targets)

            with _amp_context(device, use_amp):
                raw_output = model(samples)
                output = _extract_output(raw_output)
                loss = criterion(output, targets)
                # SegAdapter Eq. (12): main CE + lambda * coarse auxiliary CE.
                if (
                    task_type == TASK_SEMANTIC_SEGMENTATION
                    and isinstance(raw_output, dict)
                    and isinstance(raw_output.get("aux"), torch.Tensor)
                ):
                    aux_output = raw_output["aux"]
                    if tuple(aux_output.shape[-2:]) != tuple(targets.shape[-2:]):
                        aux_output = torch.nn.functional.interpolate(
                            aux_output, size=targets.shape[-2:], mode="bilinear", align_corners=False
                        )
                    aux_weight = float(raw_output.get("aux_weight", 0.4))
                    loss = loss + aux_weight * criterion(aux_output, targets)
            batch_size = int(samples.shape[0])

        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Loss is not finite: {loss_value}")

        # Normalize by the actual window size so a final incomplete
        # accumulation window has the same mean-gradient semantics.
        scaled_loss = loss / max(1, window_size)
        if use_amp and device.type == "cuda":
            is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            grad_norm = loss_scaler(
                scaled_loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=is_update_step,
            )
            if is_update_step:
                _apply_post_optimizer_constraints(model)
                optimizer.zero_grad(set_to_none=True)
                if model_ema is not None:
                    model_ema.update(model)
        else:
            scaled_loss.backward()
            grad_norm = None
            if is_update_step:
                if max_norm is not None and max_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                _apply_post_optimizer_constraints(model)
                optimizer.zero_grad(set_to_none=True)
                if model_ema is not None:
                    model_ema.update(model)

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

        metric_logger.meters["loss"].update(loss_value, n=batch_size)
        if task_type == TASK_SINGLE_LABEL and mixup_fn is None:
            batch_acc = (output.argmax(dim=-1) == targets).float().mean()
            metric_logger.update(class_acc=float(batch_acc.item()))
        elif task_type == TASK_REGRESSION:
            metric_logger.update(batch_mae=float((output - targets).abs().mean().item()))
        elif task_type == TASK_SEMANTIC_SEGMENTATION:
            valid = targets >= 0
            if valid.any():
                batch_pixel_acc = (output.argmax(dim=1)[valid] == targets[valid]).float().mean()
                metric_logger.update(pixel_acc=float(batch_pixel_acc.item()))
        elif task_type == TASK_DEPTH_ESTIMATION:
            valid = torch.isfinite(targets) & (targets > 0)
            if valid.any():
                metric_logger.update(batch_abs_rel=float(((output[valid] - targets[valid]).abs() / targets[valid].clamp_min(1e-6)).mean().item()))

        min_lr = min(group["lr"] for group in optimizer.param_groups)
        max_lr = max(group["lr"] for group in optimizer.param_groups)
        metric_logger.update(lr=max_lr, min_lr=min_lr)
        positive_wd = [group["weight_decay"] for group in optimizer.param_groups if group.get("weight_decay", 0) > 0]
        if positive_wd:
            metric_logger.update(weight_decay=positive_wd[-1])
        if grad_norm is not None:
            metric_logger.update(grad_norm=float(grad_norm))

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(lr=max_lr, min_lr=min_lr, head="opt")
            log_writer.set_step()

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elapsed_epoch = time.time() - epoch_start
    metric_logger.update(epoch_time=elapsed_epoch)
    dataset_size = len(getattr(data_loader, "dataset", []))
    if dataset_size:
        metric_logger.update(train_samples_per_second=float(dataset_size / max(elapsed_epoch, 1e-12)))
    if device.type == "cuda" and torch.cuda.is_available():
        metric_logger.update(peak_train_memory_mb=torch.cuda.max_memory_allocated(device) / 1024**2)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


def _distributed_concat(tensor: torch.Tensor) -> torch.Tensor:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return tensor
    gathered = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(gathered, tensor.cpu())
    return torch.cat(gathered, dim=0)


def _binary_average_precision(scores: torch.Tensor, targets: torch.Tensor) -> float:
    positives = float(targets.sum().item())
    if positives <= 0:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    sorted_targets = targets[order].float()
    precision = sorted_targets.cumsum(0) / torch.arange(1, len(sorted_targets) + 1, dtype=torch.float32)
    return float((precision * sorted_targets).sum().item() / positives)




def _binary_auroc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Exact rank-based AUROC with average ranks for ties."""
    scores = scores.flatten().float()
    truth = targets.flatten().bool()
    positives = int(truth.sum().item())
    negatives = int((~truth).sum().item())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _rankdata(scores)
    positive_rank_sum = ranks[truth].sum()
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc.item())

def _multilabel_metrics(logits: torch.Tensor, targets: torch.Tensor, ece_bins: int = 15):
    """Comprehensive multi-label metrics from the complete prediction set.

    Thresholded metrics use 0.5. Percentage-valued metrics follow the 0--100
    convention used by classification accuracy. Per-class diagnostics are kept
    in JSON rather than collapsed into a single opaque score.
    """
    logits = logits.float()
    targets = targets.float()
    probabilities = logits.sigmoid()
    predictions = probabilities >= 0.5
    truth = targets >= 0.5

    tp_c = (predictions & truth).sum(dim=0).float()
    fp_c = (predictions & ~truth).sum(dim=0).float()
    fn_c = (~predictions & truth).sum(dim=0).float()
    support = truth.sum(dim=0).float()
    precision_c = tp_c / (tp_c + fp_c).clamp_min(1.0)
    recall_c = tp_c / (tp_c + fn_c).clamp_min(1.0)
    f1_c = 2.0 * precision_c * recall_c / (precision_c + recall_c).clamp_min(1e-12)
    valid = support > 0

    tp = tp_c.sum()
    fp = fp_c.sum()
    fn = fn_c.sum()
    micro_precision = tp / (tp + fp).clamp_min(1.0)
    micro_recall = tp / (tp + fn).clamp_min(1.0)
    micro_f1 = 2.0 * micro_precision * micro_recall / (micro_precision + micro_recall).clamp_min(1e-12)
    macro_precision = precision_c[valid].mean() if valid.any() else torch.tensor(0.0)
    macro_recall = recall_c[valid].mean() if valid.any() else torch.tensor(0.0)
    macro_f1 = f1_c[valid].mean() if valid.any() else torch.tensor(0.0)
    weighted_f1 = (f1_c * support).sum() / support.sum().clamp_min(1.0)

    aps = [_binary_average_precision(probabilities[:, c], truth[:, c]) for c in range(probabilities.shape[1])]
    aucs = [_binary_auroc(probabilities[:, c], truth[:, c]) for c in range(probabilities.shape[1])]
    valid_aps = [value for value in aps if not math.isnan(value)]
    valid_aucs = [value for value in aucs if not math.isnan(value)]
    micro_auc = _binary_auroc(probabilities.flatten(), truth.flatten())
    subset_accuracy = predictions.eq(truth).all(dim=1).float().mean()
    hamming_accuracy = predictions.eq(truth).float().mean()
    label_cardinality_error = (predictions.sum(dim=1).float() - truth.sum(dim=1).float()).abs().mean()
    brier = (probabilities - targets).square().mean()

    flat_confidence = torch.maximum(probabilities, 1.0 - probabilities).flatten()
    flat_correct = predictions.eq(truth).float().flatten()
    ece = torch.tensor(0.0)
    boundaries = torch.linspace(0.0, 1.0, int(ece_bins) + 1)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (flat_confidence > lower) & (flat_confidence <= upper)
        if in_bin.any():
            ece += in_bin.float().mean() * (flat_correct[in_bin].mean() - flat_confidence[in_bin].mean()).abs()

    scalar = {
        "map": 100.0 * sum(valid_aps) / max(1, len(valid_aps)),
        "macro_auroc": 100.0 * sum(valid_aucs) / max(1, len(valid_aucs)),
        "micro_auroc": 0.0 if math.isnan(micro_auc) else 100.0 * micro_auc,
        "micro_precision": float(micro_precision.item() * 100.0),
        "micro_recall": float(micro_recall.item() * 100.0),
        "micro_f1": float(micro_f1.item() * 100.0),
        "macro_precision": float(macro_precision.item() * 100.0),
        "macro_recall": float(macro_recall.item() * 100.0),
        "macro_f1": float(macro_f1.item() * 100.0),
        "weighted_f1": float(weighted_f1.item() * 100.0),
        "subset_accuracy": float(subset_accuracy.item() * 100.0),
        "hamming_accuracy": float(hamming_accuracy.item() * 100.0),
        "label_cardinality_error": float(label_cardinality_error.item()),
        "ece": float(ece.item() * 100.0),
        "brier_score": float(brier.item()),
        "mean_confidence": float(flat_confidence.mean().item() * 100.0),
        "num_samples": int(targets.shape[0]),
        "num_labels": int(targets.shape[1]),
    }
    diagnostic = {
        "per_class_average_precision": [None if math.isnan(value) else float(value * 100.0) for value in aps],
        "per_class_auroc": [None if math.isnan(value) else float(value * 100.0) for value in aucs],
        "per_class_precision": [float(value * 100.0) for value in precision_c.tolist()],
        "per_class_recall": [float(value * 100.0) for value in recall_c.tolist()],
        "per_class_f1": [float(value * 100.0) for value in f1_c.tolist()],
        "per_class_support": [int(value) for value in support.tolist()],
    }
    return scalar, diagnostic


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    """Average ranks for a one-dimensional tensor, including ties."""
    values = values.flatten().float()
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    i = 0
    while i < sorted_values.numel():
        j = i + 1
        while j < sorted_values.numel() and bool(sorted_values[j] == sorted_values[i]):
            j += 1
        average_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = average_rank
        i = j
    return ranks


def _correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.flatten().float()
    y = y.flatten().float()
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    if denominator <= 1e-12:
        return float("nan")
    return float((x * y).sum().item() / denominator.item())


def _regression_metrics(predictions: torch.Tensor, targets: torch.Tensor):
    predictions = predictions.float()
    targets = targets.float()
    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(1)
        targets = targets.unsqueeze(1)
    error = predictions - targets
    absolute = error.abs()
    squared = error.square()
    mae_per_output = absolute.mean(dim=0)
    rmse_per_output = squared.mean(dim=0).sqrt()
    target_mean = targets.mean(dim=0, keepdim=True)
    ss_res = squared.sum(dim=0)
    ss_tot = (targets - target_mean).square().sum(dim=0)
    r2_per_output = 1.0 - ss_res / ss_tot.clamp_min(1e-12)
    pearson_per_output = [_correlation(predictions[:, i], targets[:, i]) for i in range(predictions.shape[1])]
    spearman_per_output = [
        _correlation(_rankdata(predictions[:, i]), _rankdata(targets[:, i]))
        for i in range(predictions.shape[1])
    ]
    finite_pearson = [value for value in pearson_per_output if not math.isnan(value)]
    finite_spearman = [value for value in spearman_per_output if not math.isnan(value)]
    scalar = {
        "mae": float(absolute.mean().item()),
        "median_absolute_error": float(absolute.median().item()),
        "rmse": float(squared.mean().sqrt().item()),
        "r2": float(r2_per_output.mean().item()),
        "pearson": float(sum(finite_pearson) / max(1, len(finite_pearson))),
        "spearman": float(sum(finite_spearman) / max(1, len(finite_spearman))),
        "num_samples": int(targets.shape[0]),
        "output_dim": int(targets.shape[1]),
    }
    diagnostic = {
        "per_output_mae": [float(value) for value in mae_per_output.tolist()],
        "per_output_rmse": [float(value) for value in rmse_per_output.tolist()],
        "per_output_r2": [float(value) for value in r2_per_output.tolist()],
        "per_output_pearson": [None if math.isnan(value) else float(value) for value in pearson_per_output],
        "per_output_spearman": [None if math.isnan(value) else float(value) for value in spearman_per_output],
    }
    return scalar, diagnostic

def _single_label_detailed_metrics(logits: torch.Tensor, targets: torch.Tensor, ece_bins: int = 15):
    """Compute classification metrics from the complete prediction set.

    Percent-valued metrics use the same 0--100 convention as Acc@1/Acc@5.
    Non-scalar diagnostics are returned separately for JSON reporting.
    """
    logits = logits.float()
    targets = targets.long().view(-1)
    num_classes = int(logits.shape[1])
    probabilities = logits.softmax(dim=1)
    confidence, predictions = probabilities.max(dim=1)
    encoded = targets * num_classes + predictions
    confusion = torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    confusion_f = confusion.float()
    support = confusion_f.sum(dim=1)
    predicted_count = confusion_f.sum(dim=0)
    true_positive = confusion_f.diag()
    recall = true_positive / support.clamp_min(1.0)
    precision = true_positive / predicted_count.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    valid = support > 0
    macro_precision = precision[valid].mean() if valid.any() else torch.tensor(0.0)
    macro_recall = recall[valid].mean() if valid.any() else torch.tensor(0.0)
    macro_f1 = f1[valid].mean() if valid.any() else torch.tensor(0.0)
    weighted_f1 = (f1 * support).sum() / support.sum().clamp_min(1.0)

    correctness = predictions.eq(targets).float()
    ece = torch.tensor(0.0)
    boundaries = torch.linspace(0.0, 1.0, int(ece_bins) + 1)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            ece += in_bin.float().mean() * (correctness[in_bin].mean() - confidence[in_bin].mean()).abs()

    one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
    per_class_ap = [_binary_average_precision(probabilities[:, c], one_hot[:, c]) for c in range(num_classes)]
    per_class_auc = [_binary_auroc(probabilities[:, c], one_hot[:, c]) for c in range(num_classes)]
    valid_ap = [value for value in per_class_ap if not math.isnan(value)]
    valid_auc = [value for value in per_class_auc if not math.isnan(value)]
    brier = (probabilities - one_hot).square().sum(dim=1).mean()
    nll = -torch.log(probabilities[torch.arange(targets.numel()), targets].clamp_min(1e-12)).mean()
    predictive_entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=1).mean()
    total = confusion_f.sum().clamp_min(1.0)
    observed_agreement = true_positive.sum() / total
    expected_agreement = (support * predicted_count).sum() / total.square()
    cohen_kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement).clamp_min(1e-12)
    covariance = true_positive.sum() * total - (support * predicted_count).sum()
    mcc_denominator = torch.sqrt(
        (total.square() - predicted_count.square().sum()).clamp_min(1e-12)
        * (total.square() - support.square().sum()).clamp_min(1e-12)
    )
    mcc = covariance / mcc_denominator
    scalar = {
        "macro_precision": float(macro_precision.item() * 100.0),
        "macro_recall": float(macro_recall.item() * 100.0),
        "macro_f1": float(macro_f1.item() * 100.0),
        "weighted_f1": float(weighted_f1.item() * 100.0),
        "balanced_accuracy": float(macro_recall.item() * 100.0),
        "macro_average_precision": 100.0 * sum(valid_ap) / max(1, len(valid_ap)),
        "macro_auroc_ovr": 100.0 * sum(valid_auc) / max(1, len(valid_auc)),
        "ece": float(ece.item() * 100.0),
        "mean_confidence": float(confidence.mean().item() * 100.0),
        "brier_score": float(brier.item()),
        "nll": float(nll.item()),
        "predictive_entropy": float(predictive_entropy.item()),
        "cohen_kappa": float(cohen_kappa.item()),
        "mcc": float(mcc.item()),
        "num_samples": int(targets.numel()),
        "num_classes": num_classes,
    }
    diagnostic = {
        "per_class_precision": [float(value * 100.0) for value in precision.tolist()],
        "per_class_average_precision": [None if math.isnan(value) else float(value * 100.0) for value in per_class_ap],
        "per_class_auroc_ovr": [None if math.isnan(value) else float(value * 100.0) for value in per_class_auc],
        "per_class_recall": [float(value * 100.0) for value in recall.tolist()],
        "per_class_f1": [float(value * 100.0) for value in f1.tolist()],
        "per_class_accuracy": [float(value * 100.0) for value in recall.tolist()],
        "per_class_support": [int(value) for value in support.tolist()],
        "confusion_matrix": confusion.tolist(),
    }
    return scalar, diagnostic


@torch.no_grad()
def _segmentation_metrics_from_confusion(confusion: torch.Tensor):
    confusion = confusion.double()
    true_positive = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    union = support + predicted - true_positive
    valid = support > 0
    iou = true_positive / union.clamp_min(1.0)
    dice = 2.0 * true_positive / (support + predicted).clamp_min(1.0)
    class_accuracy = true_positive / support.clamp_min(1.0)
    total = support.sum().clamp_min(1.0)
    pixel_accuracy = true_positive.sum() / total
    frequency = support / total
    scalar = {
        "pixel_accuracy": float(pixel_accuracy.item() * 100.0),
        "mean_class_accuracy": float(class_accuracy[valid].mean().item() * 100.0) if valid.any() else 0.0,
        "miou": float(iou[valid].mean().item() * 100.0) if valid.any() else 0.0,
        "mean_dice": float(dice[valid].mean().item() * 100.0) if valid.any() else 0.0,
        "frequency_weighted_iou": float((frequency[valid] * iou[valid]).sum().item() * 100.0) if valid.any() else 0.0,
        "num_classes_present": int(valid.sum().item()),
        "num_pixels": int(total.item()),
    }
    diagnostic = {
        "per_class_iou": [None if not bool(valid[i]) else float(iou[i].item() * 100.0) for i in range(len(iou))],
        "per_class_dice": [None if not bool(valid[i]) else float(dice[i].item() * 100.0) for i in range(len(dice))],
        "per_class_accuracy": [None if not bool(valid[i]) else float(class_accuracy[i].item() * 100.0) for i in range(len(class_accuracy))],
        "per_class_support_pixels": [int(value) for value in support.tolist()],
        "confusion_matrix": confusion.long().tolist(),
    }
    return scalar, diagnostic


def _depth_metrics_from_sums(values: dict[str, float]):
    count = max(float(values.get("count", 0.0)), 1.0)
    mean_log_error = float(values.get("log_error_sum", 0.0)) / count
    mean_log_square = float(values.get("log_error_square_sum", 0.0)) / count
    silog_variance = max(0.0, mean_log_square - mean_log_error * mean_log_error)
    return {
        "abs_rel": float(values.get("abs_rel_sum", 0.0) / count),
        "sq_rel": float(values.get("sq_rel_sum", 0.0) / count),
        "rmse": float(math.sqrt(max(0.0, values.get("square_error_sum", 0.0) / count))),
        "rmse_log": float(math.sqrt(max(0.0, mean_log_square))),
        "silog": float(100.0 * math.sqrt(silog_variance)),
        "mae": float(values.get("absolute_error_sum", 0.0) / count),
        "log10": float(values.get("log10_error_sum", 0.0) / count),
        "delta1": float(100.0 * values.get("delta1_sum", 0.0) / count),
        "delta2": float(100.0 * values.get("delta2_sum", 0.0) / count),
        "delta3": float(100.0 * values.get("delta3_sum", 0.0) / count),
        "valid_pixels": int(values.get("count", 0.0)),
    }


def _box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros(0)
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=1)
    area_a = (box[2:] - box[:2]).clamp_min(0).prod()
    area_b = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(dim=1)
    return intersection / (area_a + area_b - intersection).clamp_min(1e-12)


def _average_precision(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    if recalls.numel() == 0:
        return 0.0
    grid = torch.linspace(0.0, 1.0, 101)
    values = []
    for recall_level in grid:
        mask = recalls >= recall_level
        values.append(float(precisions[mask].max().item()) if mask.any() else 0.0)
    return float(sum(values) / len(values))


def _fallback_detection_metrics(predictions, targets):
    thresholds = [0.5 + 0.05 * index for index in range(10)]
    labels = sorted({int(value) for target in targets for value in target.get("labels", torch.empty(0, dtype=torch.long)).tolist()})
    ap_by_threshold = {threshold: [] for threshold in thresholds}
    for label in labels:
        gt_by_image = {}
        gt_count = 0
        detections = []
        for image_index, (prediction, target) in enumerate(zip(predictions, targets)):
            gt_mask = target["labels"] == label
            gt_boxes = target["boxes"][gt_mask].float()
            gt_by_image[image_index] = gt_boxes
            gt_count += int(gt_boxes.shape[0])
            pred_mask = prediction["labels"] == label
            for box, score in zip(prediction["boxes"][pred_mask], prediction["scores"][pred_mask]):
                detections.append((float(score), image_index, box.float()))
        if gt_count == 0:
            continue
        detections.sort(key=lambda row: row[0], reverse=True)
        for threshold in thresholds:
            matched = {index: torch.zeros(len(boxes), dtype=torch.bool) for index, boxes in gt_by_image.items()}
            tp, fp = [], []
            for _, image_index, box in detections:
                gt_boxes = gt_by_image[image_index]
                if gt_boxes.numel() == 0:
                    tp.append(0.0); fp.append(1.0); continue
                ious = _box_iou(box, gt_boxes)
                best = int(torch.argmax(ious).item()) if ious.numel() else -1
                if best >= 0 and float(ious[best]) >= threshold and not bool(matched[image_index][best]):
                    matched[image_index][best] = True; tp.append(1.0); fp.append(0.0)
                else:
                    tp.append(0.0); fp.append(1.0)
            if not tp:
                ap_by_threshold[threshold].append(0.0)
                continue
            tp_t = torch.tensor(tp).cumsum(0)
            fp_t = torch.tensor(fp).cumsum(0)
            recalls = tp_t / max(1, gt_count)
            precisions = tp_t / (tp_t + fp_t).clamp_min(1.0)
            ap_by_threshold[threshold].append(_average_precision(recalls, precisions))
    means = {threshold: (sum(values) / len(values) if values else 0.0) for threshold, values in ap_by_threshold.items()}

    def mean_recall(max_detections: int) -> float:
        recalls = []
        for threshold in thresholds:
            for prediction, target in zip(predictions, targets):
                order = torch.argsort(prediction["scores"], descending=True)[:max_detections]
                pred_boxes = prediction["boxes"][order]
                pred_labels = prediction["labels"][order]
                gt_boxes, gt_labels = target["boxes"], target["labels"]
                if len(gt_boxes) == 0:
                    continue
                matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
                for box, label in zip(pred_boxes, pred_labels):
                    candidates = torch.where((gt_labels == label) & ~matched)[0]
                    if candidates.numel() == 0:
                        continue
                    ious = _box_iou(box.float(), gt_boxes[candidates].float())
                    best_local = int(torch.argmax(ious).item())
                    if float(ious[best_local]) >= threshold:
                        matched[int(candidates[best_local])] = True
                recalls.append(float(matched.float().mean().item()))
        return 100.0 * sum(recalls) / max(1, len(recalls))

    return {
        "map": 100.0 * sum(means.values()) / max(1, len(means)),
        "map_50": 100.0 * means.get(0.5, 0.0),
        "map_75": 100.0 * means.get(0.75, 0.0),
        "mar_1": mean_recall(1),
        "mar_10": mean_recall(10),
        "mar_100": mean_recall(100),
        "num_images": len(targets),
        "num_classes_present": len(labels),
        "metric_backend": "internal_101_point",
    }


def _detection_metrics(predictions, targets):
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
        metric.update(predictions, targets)
        result = metric.compute()
        def scalar(key):
            value = result[key]
            return float(value.item() * 100.0) if torch.is_tensor(value) else float(value) * 100.0
        payload = {
            "map": scalar("map"), "map_50": scalar("map_50"), "map_75": scalar("map_75"),
            "mar_1": scalar("mar_1"), "mar_10": scalar("mar_10"), "mar_100": scalar("mar_100"),
            "num_images": len(targets), "metric_backend": "torchmetrics_coco",
        }
        diagnostic = {
            "per_class_map": [float(value * 100.0) for value in result.get("map_per_class", torch.empty(0)).tolist()],
            "classes": [int(value) for value in result.get("classes", torch.empty(0, dtype=torch.long)).tolist()],
        }
        return payload, diagnostic
    except Exception as exc:
        payload = _fallback_detection_metrics(predictions, targets)
        payload["metric_backend_note"] = f"torchmetrics unavailable: {type(exc).__name__}"
        return payload, {}


def evaluate(
    data_loader,
    model,
    device,
    use_amp: bool = False,
    measure_latency: bool = False,
    task_type: str = TASK_SINGLE_LABEL,
    criterion: Optional[torch.nn.Module] = None,
):
    if criterion is None and task_type != TASK_OBJECT_DETECTION:
        criterion = {
            TASK_SINGLE_LABEL: torch.nn.CrossEntropyLoss(),
            TASK_MULTILABEL: torch.nn.BCEWithLogitsLoss(),
            TASK_REGRESSION: torch.nn.MSELoss(),
            TASK_SEMANTIC_SEGMENTATION: torch.nn.CrossEntropyLoss(ignore_index=255),
            TASK_DEPTH_ESTIMATION: torch.nn.L1Loss(),
        }[task_type]
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    model.eval()

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    outputs, targets_all = [], []
    detection_outputs, detection_targets = [], []
    segmentation_confusion = None
    depth_sums = {key: 0.0 for key in (
        "count", "abs_rel_sum", "sq_rel_sum", "square_error_sum", "absolute_error_sum",
        "log_error_sum", "log_error_square_sum", "log10_error_sum", "delta1_sum", "delta2_sum", "delta3_sum",
    )}
    n_images, elapsed_forward = 0, 0.0
    for batch in metric_logger.log_every(data_loader, 10, header):
        if task_type == TASK_OBJECT_DETECTION:
            images = [image.to(device, non_blocking=True) for image in batch[0]]
            target = _move_detection_targets(batch[1], device)
            if measure_latency and device.type == "cuda" and torch.cuda.is_available(): torch.cuda.synchronize(device)
            start_forward = time.perf_counter() if measure_latency else 0.0
            with _amp_context(device, use_amp):
                prediction = model(images)
            if measure_latency and device.type == "cuda" and torch.cuda.is_available(): torch.cuda.synchronize(device)
            if measure_latency: elapsed_forward += time.perf_counter() - start_forward
            batch_size = len(images)
            for row in prediction:
                detection_outputs.append({key: value.detach().cpu() for key, value in row.items() if key in {"boxes", "scores", "labels"}})
            for row in target:
                detection_targets.append({key: value.detach().cpu() for key, value in row.items() if key in {"boxes", "labels", "area", "iscrowd", "image_id"}})
            n_images += batch_size
            continue

        images = batch[0].to(device, non_blocking=True)
        target = _prepare_target(batch[1].to(device, non_blocking=True), task_type)
        if measure_latency and device.type == "cuda" and torch.cuda.is_available(): torch.cuda.synchronize(device)
        start_forward = time.perf_counter() if measure_latency else 0.0
        with _amp_context(device, use_amp):
            output = _extract_output(model(images))
        if measure_latency and device.type == "cuda" and torch.cuda.is_available(): torch.cuda.synchronize(device)
        if measure_latency: elapsed_forward += time.perf_counter() - start_forward
        with _amp_context(device, use_amp):
            loss = criterion(output, target)
        batch_size = images.shape[0]
        metric_logger.meters["loss"].update(float(loss.item()), n=batch_size)

        if task_type == TASK_SINGLE_LABEL:
            topk = (1, 5) if output.shape[-1] >= 5 else (1,)
            values = accuracy(output, target, topk=topk)
            metric_logger.meters["acc1"].update(float(values[0].item()), n=batch_size)
            if len(values) > 1: metric_logger.meters["acc5"].update(float(values[1].item()), n=batch_size)
            outputs.append(output.detach().cpu()); targets_all.append(target.detach().cpu())
        elif task_type in {TASK_MULTILABEL, TASK_REGRESSION}:
            outputs.append(output.detach().cpu()); targets_all.append(target.detach().cpu())
        elif task_type == TASK_SEMANTIC_SEGMENTATION:
            predictions = output.argmax(dim=1)
            ignore_index = int(getattr(criterion, "ignore_index", 255))
            valid = (target != ignore_index) & (target >= 0) & (target < output.shape[1])
            encoded = target[valid] * output.shape[1] + predictions[valid]
            batch_confusion = torch.bincount(encoded, minlength=output.shape[1] ** 2).reshape(output.shape[1], output.shape[1]).cpu()
            segmentation_confusion = batch_confusion if segmentation_confusion is None else segmentation_confusion + batch_confusion
        elif task_type == TASK_DEPTH_ESTIMATION:
            valid = torch.isfinite(target) & torch.isfinite(output) & (target > 0)
            if valid.any():
                prediction = output[valid].float().clamp_min(1e-6)
                truth = target[valid].float().clamp_min(1e-6)
                error = prediction - truth
                log_error = torch.log(prediction) - torch.log(truth)
                ratio = torch.maximum(prediction / truth, truth / prediction)
                depth_sums["count"] += float(valid.sum().item())
                depth_sums["abs_rel_sum"] += float((error.abs() / truth).sum().item())
                depth_sums["sq_rel_sum"] += float((error.square() / truth).sum().item())
                depth_sums["square_error_sum"] += float(error.square().sum().item())
                depth_sums["absolute_error_sum"] += float(error.abs().sum().item())
                depth_sums["log_error_sum"] += float(log_error.sum().item())
                depth_sums["log_error_square_sum"] += float(log_error.square().sum().item())
                depth_sums["log10_error_sum"] += float((torch.log10(prediction) - torch.log10(truth)).abs().sum().item())
                depth_sums["delta1_sum"] += float((ratio < 1.25).sum().item())
                depth_sums["delta2_sum"] += float((ratio < 1.25 ** 2).sum().item())
                depth_sums["delta3_sum"] += float((ratio < 1.25 ** 3).sum().item())
        n_images += batch_size

    diagnostic_metrics = {}
    if outputs:
        prediction_tensor = _distributed_concat(torch.cat(outputs, dim=0))
        target_tensor = _distributed_concat(torch.cat(targets_all, dim=0))
        if task_type == TASK_SINGLE_LABEL:
            scalar_metrics, diagnostic_metrics = _single_label_detailed_metrics(prediction_tensor, target_tensor)
        elif task_type == TASK_MULTILABEL:
            scalar_metrics, diagnostic_metrics = _multilabel_metrics(prediction_tensor, target_tensor)
        else:
            scalar_metrics, diagnostic_metrics = _regression_metrics(prediction_tensor, target_tensor)
        for key, value in scalar_metrics.items(): metric_logger.update(**{key: value})
    elif task_type == TASK_SEMANTIC_SEGMENTATION and segmentation_confusion is not None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tensor = segmentation_confusion.to(device)
            torch.distributed.all_reduce(tensor)
            segmentation_confusion = tensor.cpu()
        scalar_metrics, diagnostic_metrics = _segmentation_metrics_from_confusion(segmentation_confusion)
        for key, value in scalar_metrics.items(): metric_logger.update(**{key: value})
    elif task_type == TASK_DEPTH_ESTIMATION:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            keys = list(depth_sums)
            tensor = torch.tensor([depth_sums[key] for key in keys], dtype=torch.float64, device=device)
            torch.distributed.all_reduce(tensor)
            depth_sums = dict(zip(keys, tensor.cpu().tolist()))
        scalar_metrics = _depth_metrics_from_sums(depth_sums)
        for key, value in scalar_metrics.items(): metric_logger.update(**{key: value})
    elif task_type == TASK_OBJECT_DETECTION:
        scalar_metrics, diagnostic_metrics = _detection_metrics(detection_outputs, detection_targets)
        for key, value in scalar_metrics.items():
            if isinstance(value, (int, float)): metric_logger.update(**{key: value})

    if measure_latency and n_images > 0:
        metric_logger.update(latency_ms_per_image=1000.0 * elapsed_forward / n_images)
        metric_logger.update(fps=n_images / max(elapsed_forward, 1e-12))
    if device.type == "cuda" and torch.cuda.is_available():
        metric_logger.update(peak_inference_memory_mb=torch.cuda.max_memory_allocated(device) / 1024**2)

    metric_logger.synchronize_between_processes()
    stats = {key: meter.global_avg for key, meter in metric_logger.meters.items()}
    stats.update(diagnostic_metrics)
    if task_type == TASK_SINGLE_LABEL:
        print(f"* Acc@1 {stats.get('acc1', 0):.3f} Acc@5 {stats.get('acc5', 0):.3f} macro-F1 {stats.get('macro_f1', 0):.3f} balanced-acc {stats.get('balanced_accuracy', 0):.3f} ECE {stats.get('ece', 0):.3f} loss {stats.get('loss', 0):.3f}")
    elif task_type == TASK_MULTILABEL:
        print(f"* mAP {stats.get('map', 0):.3f} micro-F1 {stats.get('micro_f1', 0):.3f} macro-F1 {stats.get('macro_f1', 0):.3f} subset-acc {stats.get('subset_accuracy', 0):.3f} ECE {stats.get('ece', 0):.3f} loss {stats.get('loss', 0):.3f}")
    elif task_type == TASK_REGRESSION:
        print(f"* MAE {stats.get('mae', 0):.5f} RMSE {stats.get('rmse', 0):.5f} R2 {stats.get('r2', 0):.5f} Pearson {stats.get('pearson', 0):.5f} loss {stats.get('loss', 0):.5f}")
    elif task_type == TASK_SEMANTIC_SEGMENTATION:
        print(f"* mIoU {stats.get('miou', 0):.3f} Dice {stats.get('mean_dice', 0):.3f} pixel-acc {stats.get('pixel_accuracy', 0):.3f} loss {stats.get('loss', 0):.4f}")
    elif task_type == TASK_DEPTH_ESTIMATION:
        print(f"* AbsRel {stats.get('abs_rel', 0):.5f} RMSE {stats.get('rmse', 0):.5f} delta1 {stats.get('delta1', 0):.3f} SILog {stats.get('silog', 0):.3f} loss {stats.get('loss', 0):.5f}")
    else:
        print(f"* box mAP {stats.get('map', 0):.3f} AP50 {stats.get('map_50', 0):.3f} AP75 {stats.get('map_75', 0):.3f} AR100 {stats.get('mar_100', 0):.3f}")
    return stats

