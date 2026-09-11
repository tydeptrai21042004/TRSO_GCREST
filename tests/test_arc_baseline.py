from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torchvision.models import resnet18
from torchvision.models.vision_transformer import VisionTransformer

from main import get_args_parser, set_trainability_policy
from models.model_support import method_category, method_compatibility
from models.tuning_modules.arc import (
    ARCProjectionBank,
    TorchvisionARCBlock,
    apply_arc,
    merge_arc_,
    set_arc_trainability,
)
from tools.run_fair_suite import method_variant


def tiny_vit() -> VisionTransformer:
    return VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=2,
        num_heads=2,
        hidden_dim=32,
        mlp_dim=64,
        num_classes=5,
        dropout=0.0,
        attention_dropout=0.0,
    )


def test_arc_is_a_strict_plain_vit_paper_baseline():
    assert method_category("arc") == "paper_baseline"
    assert method_compatibility("arc", "vit", "single_label")[0]
    assert not method_compatibility("arc", "resnet", "single_label")[0]
    assert not method_compatibility("arc", "vit", "semantic_segmentation")[0]


def test_arc_uses_two_shared_banks_and_layer_specific_coefficients():
    model = tiny_vit()
    records = apply_arc(model, adapter_dim=4, dropout=0.1)
    assert len(records) == 2
    assert isinstance(model.arc_projection_bank, ARCProjectionBank)
    assert sum(isinstance(module, ARCProjectionBank) for module in model.modules()) == 1
    wrappers = [module for module in model.modules() if isinstance(module, TorchvisionARCBlock)]
    assert len(wrappers) == 2
    assert all(wrapper.bank is model.arc_projection_bank for wrapper in wrappers)
    assert torch.count_nonzero(wrappers[0].att_adapter.adapter_rescale) == 0
    assert torch.count_nonzero(wrappers[0].mlp_adapter.adapter_rescale) > 0
    assert torch.count_nonzero(wrappers[0].att_adapter.adapter_bias) == 0
    assert torch.count_nonzero(wrappers[0].mlp_adapter.adapter_bias) == 0


def test_arc_trainability_and_gradients_are_restricted_to_arc_and_head():
    model = tiny_vit()
    apply_arc(model, adapter_dim=4, dropout=0.0)
    set_arc_trainability(model)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(
        name.startswith("arc_projection_bank.")
        or ".att_adapter." in name
        or ".mlp_adapter." in name
        or name.startswith("heads.")
        for name in trainable
    )
    output = model(torch.randn(2, 3, 32, 32))
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_arc_eval_merge_is_exact_and_idempotent():
    torch.manual_seed(9)
    model = tiny_vit()
    apply_arc(model, adapter_dim=4, dropout=0.0)
    # Exercise nonzero attention coefficients and biases, not only the official identity start.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, TorchvisionARCBlock):
                module.att_adapter.adapter_rescale.normal_(0.0, 0.1)
                module.att_adapter.adapter_bias.normal_(0.0, 0.1)
                module.mlp_adapter.adapter_bias.normal_(0.0, 0.1)
    model.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        before = model(images)
    assert merge_arc_(model) == 4
    assert merge_arc_(model) == 0
    with torch.no_grad():
        after = model(images)
    assert torch.allclose(before, after, atol=1e-6, rtol=1e-5)


def test_arc_rejects_non_vit_and_requires_eval_for_merge():
    with pytest.raises((TypeError, ValueError)):
        apply_arc(resnet18(weights=None), adapter_dim=4)
    model = tiny_vit()
    apply_arc(model, adapter_dim=4)
    with pytest.raises(RuntimeError, match="eval"):
        merge_arc_(model)


def test_arc_is_wired_through_main_and_fair_runner():
    args = get_args_parser().parse_args([])
    assert args.arc_dim == 50
    assert args.arc_dropout == 0.1
    assert args.arc_merge is True
    row = method_variant("arc", "vit", 0, "/tmp/head.pth", SimpleNamespace(head_init_policy="linear_probe"))
    assert row["arc_dim"] == 50
    assert row["arc_dropout"] == 0.1
    assert row["arc_merge"] is True
    assert row["head_from"] == "/tmp/head.pth"

    model = tiny_vit()
    apply_arc(model, adapter_dim=4, dropout=0.0)
    set_trainability_policy(model, SimpleNamespace(tuning_method="arc"))
    assert any(parameter.requires_grad for parameter in model.parameters())
