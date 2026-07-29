"""Regression tests for exact original-paper architecture/task contracts."""
from __future__ import annotations

import pytest

from main import canonicalize_args, get_args_parser
from models.model_support import (
    PAPER_BASELINE_METHODS,
    PAPER_ABLATION_METHODS,
    PAPER_REIMPLEMENTATION_CANDIDATES,
    STRICT_AUTO_METHODS,
    method_category,
    static_method_compatibility,
    validate_method_task,
)
from tools.run_fair_suite import parse_methods


def decision(method: str, backbone: str, task: str = "single_label", **kwargs):
    return static_method_compatibility(method, backbone, task, **kwargs)


def test_auto_suite_contains_only_strict_paper_baselines():
    assert tuple(parse_methods("auto")) == STRICT_AUTO_METHODS
    assert set(parse_methods("auto")) == set(PAPER_BASELINE_METHODS)
    for method in parse_methods("auto"):
        assert method_category(method) == "paper_baseline"


def test_nonbaseline_publication_groups_are_not_in_auto():
    auto = set(parse_methods("auto"))
    assert not auto.intersection({"full", "linear", "trso", "norm", "bias", "last_block", "lora", "bitfit", "sidetune"})
    assert not auto.intersection(PAPER_ABLATION_METHODS)
    assert not auto.intersection(PAPER_REIMPLEMENTATION_CANDIDATES)


@pytest.mark.parametrize(
    "method,valid_backbone,invalid_backbone",
    [
        ("prompt", "resnet18", "vit_b_16"),
        ("conv", "resnet50", "resnet18"),
        ("repadapter", "vit_b_16", "vit_tiny_patch16_224"),
        ("piggyback", "vgg16", "efficientnet_b0"),
        ("adaptformer", "vit_b_16", "deit_small_patch16_224"),
        ("vpt_shallow", "vit_b_16", "swin_t"),
        ("vpt_deep", "vit_b_16", "beit_base_patch16_224"),
        ("convpass", "vit_b_16", "resnet50"),
    ],
)
def test_exact_backbone_contracts(method, valid_backbone, invalid_backbone):
    assert decision(method, valid_backbone)[0], decision(method, valid_backbone)[1]
    assert not decision(method, invalid_backbone)[0]


@pytest.mark.parametrize("backbone", ["vit_b_16", "swin_t", "convnext_tiny"])
def test_ssf_accepts_only_implemented_paper_families(backbone):
    assert decision("ssf", backbone)[0]


def test_ssf_rejects_generic_cnn_family():
    assert not decision("ssf", "efficientnet_b0")[0]
    assert not decision("ssf", "mobilenet_v3_large")[0]


def test_every_strict_visual_paper_baseline_is_single_label_only():
    for method in PAPER_BASELINE_METHODS:
        validate_method_task(method, "single_label")
        for task in ("multilabel", "regression", "semantic_segmentation", "depth_estimation", "object_detection"):
            with pytest.raises(ValueError):
                validate_method_task(method, task)


@pytest.mark.parametrize(
    "method,backbone,task",
    [
        ("lora", "vit_b_16", "regression"),
        ("bitfit", "swin_t", "multilabel"),
        ("sidetune", "resnet50", "single_label"),
        ("norm", "resnet50", "single_label"),
    ],
)
def test_nonpaper_controls_are_blocked_by_default_and_explicitly_opt_in(method, backbone, task):
    ok, reason, _ = decision(method, backbone, task)
    assert not ok and "control" in reason
    ok, reason, _ = decision(method, backbone, task, allow_nonpaper_controls=True)
    assert ok, reason


def test_visual_prompting_strict_default_uses_frequency_mapping():
    args = get_args_parser().parse_args(["--tuning_method", "prompt", "--task", "single_label"])
    args = canonicalize_args(args)
    assert args.prompt_mapping == "frequency"


def test_paper_ablation_and_reimplementation_candidates_require_explicit_opt_in():
    for method in PAPER_ABLATION_METHODS:
        ok, reason, _ = decision(method, "vit_b_16")
        assert not ok and "ablation" in reason.lower()
        ok, reason, _ = decision(method, "vit_b_16", allow_paper_ablations=True)
        assert ok, reason
    for method in PAPER_REIMPLEMENTATION_CANDIDATES:
        ok, reason, _ = decision(method, "vit_b_16")
        assert not ok and "not certified" in reason.lower()
        ok, reason, _ = decision(method, "vit_b_16", allow_unverified_paper_reimplementations=True)
        assert ok, reason
