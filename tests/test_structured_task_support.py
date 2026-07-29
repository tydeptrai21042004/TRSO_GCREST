from __future__ import annotations

from argparse import Namespace

import torch
import torch.nn as nn

from datasets.build import build_dataset_split
from engine import (
    _depth_metrics_from_sums,
    _fallback_detection_metrics,
    _segmentation_metrics_from_confusion,
)
from main import set_trainability_policy
from models.model_support import method_compatibility
from models.structured_models import build_torchvision_segmentation
from task_registry import (
    TASK_DEPTH_ESTIMATION,
    TASK_OBJECT_DETECTION,
    TASK_SEMANTIC_SEGMENTATION,
    task_spec,
)


def _args(task: str) -> Namespace:
    return Namespace(
        dataset={
            TASK_SEMANTIC_SEGMENTATION: "fake_segmentation",
            TASK_DEPTH_ESTIMATION: "fake_depth",
            TASK_OBJECT_DETECTION: "fake_detection",
        }[task],
        task=task,
        data_path="./data",
        download=False,
        input_size=32,
        crop_ratio=0.875,
        train_interpolation="bicubic",
        imagenet_norm=True,
        imagenet_default_mean_and_std=True,
        train_aug="standard",
        color_jitter=0.0,
        aa="none",
        reprob=0.0,
        remode="pixel",
        recount=1,
        fake_train_size=4,
        fake_val_size=2,
        fake_test_size=2,
        segmentation_num_classes=4,
        segmentation_ignore_index=255,
        dense_train_aug="scale_crop_flip",
        dense_hflip_prob=0.5,
        detection_num_classes=3,
        nb_classes=4,
        seed=0,
        split_seed=42,
        val_ratio=0.1,
    )


def test_fake_semantic_segmentation_contract():
    args = _args(TASK_SEMANTIC_SEGMENTATION)
    dataset, classes = build_dataset_split(args, "train")
    image, target = dataset[0]
    assert classes == 4
    assert args.task == TASK_SEMANTIC_SEGMENTATION
    assert image.shape == (3, 32, 32)
    assert target.shape == (32, 32)
    assert target.dtype == torch.long


def test_fake_depth_contract():
    args = _args(TASK_DEPTH_ESTIMATION)
    dataset, outputs = build_dataset_split(args, "val")
    image, target = dataset[0]
    assert outputs == 1
    assert image.shape == (3, 32, 32)
    assert target.shape == (1, 32, 32)
    assert target.dtype == torch.float32
    assert torch.all(target > 0)


def test_fake_detection_contract():
    args = _args(TASK_OBJECT_DETECTION)
    dataset, classes = build_dataset_split(args, "test")
    image, target = dataset[0]
    assert classes == 3
    assert image.shape == (3, 32, 32)
    assert set(("boxes", "labels", "image_id", "area", "iscrowd")).issubset(target)
    assert target["boxes"].shape[-1] == 4
    assert target["labels"].dtype == torch.int64


def test_segmentation_metrics_are_exact_for_identity_confusion():
    confusion = torch.diag(torch.tensor([2.0, 3.0, 5.0]))
    scalar, diagnostic = _segmentation_metrics_from_confusion(confusion)
    assert scalar["miou"] == 100.0
    assert scalar["mean_dice"] == 100.0
    assert scalar["pixel_accuracy"] == 100.0
    assert scalar["mean_class_accuracy"] == 100.0
    assert scalar["frequency_weighted_iou"] == 100.0
    assert diagnostic["per_class_iou"] == [100.0, 100.0, 100.0]


def test_depth_metrics_are_exact_for_perfect_predictions():
    # The reducer receives sums over valid pixels.
    scalar = _depth_metrics_from_sums({
        "count": 4.0,
        "abs_rel_sum": 0.0,
        "sq_rel_sum": 0.0,
        "square_error_sum": 0.0,
        "absolute_error_sum": 0.0,
        "log10_error_sum": 0.0,
        "log_error_sum": 0.0,
        "log_error_square_sum": 0.0,
        "delta1_sum": 4.0,
        "delta2_sum": 4.0,
        "delta3_sum": 4.0,
    })
    assert scalar["abs_rel"] == 0.0
    assert scalar["rmse"] == 0.0
    assert scalar["silog"] == 0.0
    assert scalar["delta1"] == 100.0
    assert scalar["delta2"] == 100.0
    assert scalar["delta3"] == 100.0


def test_fallback_detection_metrics_reward_perfect_box():
    prediction = {
        "boxes": torch.tensor([[2.0, 2.0, 12.0, 12.0]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.99]),
    }
    target = {
        "boxes": torch.tensor([[2.0, 2.0, 12.0, 12.0]]),
        "labels": torch.tensor([1]),
    }
    scalar = _fallback_detection_metrics([prediction], [target])
    assert scalar["map"] == 100.0
    assert scalar["map_50"] == 100.0
    assert scalar["map_75"] == 100.0
    assert scalar["mar_100"] == 100.0


def test_torchvision_segmentation_head_has_requested_channels():
    model = build_torchvision_segmentation(
        "lraspp_mobilenet_v3_large", num_outputs=5, weights="none"
    )
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32))["out"]
    assert output.shape == (1, 5, 32, 32)


class _TinyDenseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1, bias=True),
            nn.ReLU(),
        )
        self.classifier = nn.Conv2d(8, 4, 1)

    def forward(self, x):
        return {"out": self.classifier(self.backbone(x))}


def _policy_args(method: str) -> Namespace:
    return Namespace(
        tuning_method=method,
        fair_protocol=True,
        bitfit_train_head=True,
        bitfit_bias_scope="all",
        piggyback_train_head=True,
    )


def test_generic_baseline_trainability_on_dense_model():
    for method in ("linear", "norm", "bias", "last_block", "full"):
        model = _TinyDenseModel()
        set_trainability_policy(model, _policy_args(method))
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        assert trainable, method
        if method != "full":
            assert any("classifier" in name for name in trainable), (method, trainable)


def test_task_method_compatibility_is_explicit():
    assert method_compatibility("trso", "cnn", TASK_SEMANTIC_SEGMENTATION)[0]
    assert method_compatibility("trso", "cnn", TASK_DEPTH_ESTIMATION)[0]
    assert not method_compatibility("trso", "cnn", TASK_OBJECT_DETECTION)[0]
    for method in ("full", "linear", "norm", "bias", "last_block"):
        assert method_compatibility(method, "cnn", TASK_OBJECT_DETECTION)[0]


def test_every_structured_task_declares_required_metrics():
    for task in (TASK_SEMANTIC_SEGMENTATION, TASK_DEPTH_ESTIMATION, TASK_OBJECT_DETECTION):
        spec = task_spec(task)
        assert spec.primary_metric in spec.required_metrics
        assert len(spec.required_metrics) >= 6


def test_fair_runner_auto_presets_and_explicit_detection_skip(tmp_path):
    from tools.run_fair_suite import build_suite, parser as fair_parser

    args = fair_parser().parse_args([
        "--dataset", "fake_detection",
        "--task", "object_detection",
        "--weights", "none",
        "--backbones", "auto",
        "--methods", "linear,trso,full,norm,bias,last_block",
        "--allow_nonpaper_controls", "True",
        "--seeds", "0",
        "--epochs", "1",
        "--input_size", "64",
        "--output_root", str(tmp_path / "out"),
        "--manifest", str(tmp_path / "manifest.json"),
    ])
    task, _, heads, comparisons, compatibility, _ = build_suite(args)
    assert task == TASK_OBJECT_DETECTION
    assert len(heads) == 3  # one linear head for each automatic detector backbone
    assert comparisons
    trso_rows = [row for row in compatibility if row["method"] == "trso"]
    assert trso_rows and all(row["status"] == "skipped" for row in trso_rows)
    for method in ("linear", "full", "norm", "bias", "last_block"):
        assert any(row["method"] == method and row["status"] == "scheduled" for row in compatibility)
