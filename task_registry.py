"""Task definitions shared by datasets, models, training, metrics, and runners."""
from __future__ import annotations

from dataclasses import dataclass

TASK_SINGLE_LABEL = "single_label"
TASK_MULTILABEL = "multilabel"
TASK_REGRESSION = "regression"
TASK_SEMANTIC_SEGMENTATION = "semantic_segmentation"
TASK_DEPTH_ESTIMATION = "depth_estimation"
TASK_OBJECT_DETECTION = "object_detection"

ALL_TASKS = frozenset({
    TASK_SINGLE_LABEL,
    TASK_MULTILABEL,
    TASK_REGRESSION,
    TASK_SEMANTIC_SEGMENTATION,
    TASK_DEPTH_ESTIMATION,
    TASK_OBJECT_DETECTION,
})
CLASSIFICATION_TASKS = frozenset({TASK_SINGLE_LABEL, TASK_MULTILABEL})
DENSE_PREDICTION_TASKS = frozenset({TASK_SEMANTIC_SEGMENTATION, TASK_DEPTH_ESTIMATION})
STRUCTURED_PREDICTION_TASKS = frozenset({
    TASK_SEMANTIC_SEGMENTATION,
    TASK_DEPTH_ESTIMATION,
    TASK_OBJECT_DETECTION,
})

ALIASES = {
    "classification": TASK_SINGLE_LABEL,
    "singlelabel": TASK_SINGLE_LABEL,
    "single_label_classification": TASK_SINGLE_LABEL,
    "multi_label": TASK_MULTILABEL,
    "multi_label_classification": TASK_MULTILABEL,
    "segmentation": TASK_SEMANTIC_SEGMENTATION,
    "semantic_seg": TASK_SEMANTIC_SEGMENTATION,
    "semantic_segmentation": TASK_SEMANTIC_SEGMENTATION,
    "depth": TASK_DEPTH_ESTIMATION,
    "monocular_depth": TASK_DEPTH_ESTIMATION,
    "depth_estimation": TASK_DEPTH_ESTIMATION,
    "detection": TASK_OBJECT_DETECTION,
    "object_detection": TASK_OBJECT_DETECTION,
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    primary_metric: str
    maximize: bool
    display_name: str
    required_metrics: tuple[str, ...]


TASK_SPECS = {
    TASK_SINGLE_LABEL: TaskSpec(
        TASK_SINGLE_LABEL, "acc1", True, "Acc@1",
        ("acc1", "acc5", "macro_f1", "weighted_f1", "balanced_accuracy", "macro_auroc_ovr", "macro_average_precision", "mcc", "cohen_kappa", "ece", "brier_score", "nll"),
    ),
    TASK_MULTILABEL: TaskSpec(
        TASK_MULTILABEL, "map", True, "mAP",
        ("map", "macro_auroc", "micro_auroc", "micro_f1", "macro_f1", "weighted_f1", "subset_accuracy", "hamming_accuracy", "ece", "brier_score"),
    ),
    TASK_REGRESSION: TaskSpec(
        TASK_REGRESSION, "mae", False, "MAE",
        ("mae", "median_absolute_error", "rmse", "r2", "pearson", "spearman"),
    ),
    TASK_SEMANTIC_SEGMENTATION: TaskSpec(
        TASK_SEMANTIC_SEGMENTATION, "miou", True, "mIoU",
        ("miou", "mean_dice", "pixel_accuracy", "mean_class_accuracy", "frequency_weighted_iou", "loss"),
    ),
    TASK_DEPTH_ESTIMATION: TaskSpec(
        TASK_DEPTH_ESTIMATION, "abs_rel", False, "AbsRel",
        ("abs_rel", "sq_rel", "rmse", "rmse_log", "silog", "delta1", "delta2", "delta3", "mae", "log10"),
    ),
    TASK_OBJECT_DETECTION: TaskSpec(
        TASK_OBJECT_DETECTION, "map", True, "box mAP",
        ("map", "map_50", "map_75", "mar_1", "mar_10", "mar_100", "loss"),
    ),
}


def normalize_task(task: str) -> str:
    value = str(task or "auto").strip().lower().replace("-", "_")
    return ALIASES.get(value, value)


def task_choices(include_auto: bool = True) -> list[str]:
    values = sorted(ALL_TASKS)
    return (["auto"] + values) if include_auto else values


def task_spec(task: str) -> TaskSpec:
    normalized = normalize_task(task)
    if normalized not in TASK_SPECS:
        raise KeyError(f"Unknown task {task!r}. Available: {sorted(TASK_SPECS)}")
    return TASK_SPECS[normalized]


__all__ = [
    "TASK_SINGLE_LABEL", "TASK_MULTILABEL", "TASK_REGRESSION",
    "TASK_SEMANTIC_SEGMENTATION", "TASK_DEPTH_ESTIMATION", "TASK_OBJECT_DETECTION",
    "ALL_TASKS", "CLASSIFICATION_TASKS", "DENSE_PREDICTION_TASKS", "STRUCTURED_PREDICTION_TASKS",
    "TASK_SPECS", "TaskSpec", "normalize_task", "task_choices", "task_spec",
]
