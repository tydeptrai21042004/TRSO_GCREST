from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.vision_transformer import VisionTransformer

from main import get_args_parser
from models.model_support import method_compatibility
from tools.run_fair_suite import method_variant
from types import SimpleNamespace
from models.tuning_modules.convpass_transformer import apply_convpass, set_convpass_trainability
from models.tuning_modules.arc import apply_arc, merge_arc_, set_arc_trainability
from models.tuning_modules.fact import apply_fact, merge_fact_, set_fact_trainability
from models.tuning_modules.spt import calibrate_spt, merge_spt_
from models.tuning_modules.vpt import VisualPromptTuning, set_vpt_trainability
from models.tuning_modules.vqt import VisualQueryTuning, set_vqt_trainability


def tiny_vit(num_classes: int = 5):
    return VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=2,
        num_heads=2,
        hidden_dim=32,
        mlp_dim=64,
        num_classes=num_classes,
        dropout=0.0,
        attention_dropout=0.0,
    )


def test_new_paper_baselines_are_conservatively_registered():
    for method in (
        "arc", "vpt_shallow", "vpt_deep", "convpass", "convpass_attn",
        "fact_tt", "fact_tk", "vqt", "spt_lora", "spt_adapter",
    ):
        ok, _ = method_compatibility(method, "vit", "single_label")
        assert ok
        ok, _ = method_compatibility(method, "resnet", "single_label")
        assert not ok
        ok, _ = method_compatibility(method, "vit", "semantic_segmentation")
        assert not ok


def test_vpt_shallow_and_deep_train_only_prompts_and_head():
    images = torch.randn(2, 3, 32, 32)
    for deep, expected_prompt_sets in ((False, 1), (True, 2)):
        model = VisualPromptTuning(tiny_vit(), num_tokens=3, deep=deep)
        set_vpt_trainability(model)
        output = model(images)
        assert output.shape == (2, 5)
        assert model.prompt_embeddings.shape[0] == expected_prompt_sets
        output.sum().backward()
        assert model.prompt_embeddings.grad is not None
        assert all(
            not parameter.requires_grad
            for name, parameter in model.named_parameters()
            if name.startswith("backbone.encoder") or name.startswith("backbone.conv_proj")
        )


def test_convpass_is_identity_initialized_and_only_bypasses_train():
    torch.manual_seed(1)
    model = tiny_vit().eval()
    images = torch.randn(2, 3, 32, 32)
    before = model(images).detach()
    records = apply_convpass(model, bottleneck=4, scale=1.0, dropout=0.0)
    set_convpass_trainability(model)
    after = model(images).detach()
    assert len(records) == 2
    assert torch.allclose(before, after, atol=1e-6, rtol=1e-5)
    assert any("attn_adapter" in name and parameter.requires_grad for name, parameter in model.named_parameters())
    assert not any(".block." in name and parameter.requires_grad for name, parameter in model.named_parameters())


def test_fact_tt_tk_are_identity_initialized_and_exactly_merge():
    images = torch.randn(2, 3, 32, 32)
    for variant in ("tt", "tk"):
        torch.manual_seed(2)
        model = tiny_vit().eval()
        before = model(images).detach()
        records = apply_fact(model, variant=variant, rank=2)
        set_fact_trainability(model)
        after = model(images).detach()
        assert len(records) == 8
        assert torch.equal(before, after)
        train_output = model(images)
        train_output.sum().backward()
        assert model.fact_bank.v.grad is not None
        merged = merge_fact_(model)
        assert merged == 8
        assert not hasattr(model, "fact_bank")
        merged_output = model(images).detach()
        assert torch.allclose(after, merged_output, atol=1e-6, rtol=1e-5)


def test_vqt_keeps_backbone_frozen_trains_queries_and_includes_final_cls():
    model = VisualQueryTuning(tiny_vit(), num_classes=5, query_length=1)
    set_vqt_trainability(model)
    assert model.head.in_features == (2 * 1 + 1) * 32
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 5)
    output.sum().backward()
    assert model.query_tokens.grad is not None
    assert model.head.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_spt_uses_squared_gradient_budget_and_mergeable_updates(tmp_path):
    torch.manual_seed(4)
    model = tiny_vit()
    loader = DataLoader(
        TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 5, (8,))),
        batch_size=4,
        shuffle=False,
    )
    report = calibrate_spt(
        model,
        loader,
        torch.device("cpu"),
        variant="lora",
        budget=100,
        sensitivity_samples=8,
        rank=2,
        output_path=tmp_path / "spt.json",
    )
    assert report["selected_connections"] == 100
    assert report["sensitivity_samples"] == 8
    assert report["actual_backbone_trainable_parameters"] <= 100
    assert (tmp_path / "spt.json").is_file()
    output = model(torch.randn(2, 3, 32, 32))
    output.sum().backward()
    assert any(parameter.requires_grad for parameter in model.parameters())
    assert merge_spt_(model) > 0


def test_fact_standard_vit_tensorizes_twelve_operations_per_block():
    model = VisionTransformer(
        image_size=32, patch_size=8, num_layers=2, num_heads=2,
        hidden_dim=32, mlp_dim=128, num_classes=5, dropout=0.0, attention_dropout=0.0,
    )
    records = apply_fact(model, variant="tt", rank=4)
    assert len(records) == 8
    assert model.fact_bank.operations == 24
    assert model.fact_bank.rank == 4


def test_spt_sensitivity_is_invariant_to_loader_batch_size(tmp_path):
    torch.manual_seed(12)
    images = torch.randn(6, 3, 32, 32)
    targets = torch.tensor([0, 1, 2, 3, 4, 0])
    base = tiny_vit().eval()
    state = base.state_dict()
    reports = []
    for batch_size in (1, 3):
        model = tiny_vit().eval()
        model.load_state_dict(state)
        loader = DataLoader(TensorDataset(images, targets), batch_size=batch_size, shuffle=False)
        reports.append(calibrate_spt(
            model, loader, torch.device("cpu"), variant="lora", budget=80,
            sensitivity_samples=6, rank=2, output_path=tmp_path / f"spt_{batch_size}.json",
        ))
    signature = lambda report: [
        (row["name"], row["selected"], row["allocation"], row["trainable_parameters"])
        for row in report["records"]
    ]
    assert signature(reports[0]) == signature(reports[1])


def test_paper_parameterizations_match_closed_form_counts():
    hidden, layers, bottleneck = 32, 2, 4
    convpass = tiny_vit()
    apply_convpass(convpass, bottleneck=bottleneck, scale=1.0, dropout=0.0)
    set_convpass_trainability(convpass)
    adapter_count = sum(
        p.numel() for name, p in convpass.named_parameters()
        if p.requires_grad and ("attn_adapter" in name or "mlp_adapter" in name)
    )
    per_adapter = (2 * bottleneck + 1) * hidden + 9 * bottleneck * bottleneck + 2 * bottleneck
    assert adapter_count == 2 * layers * per_adapter

    fact_tt = VisionTransformer(
        image_size=32, patch_size=8, num_layers=layers, num_heads=2,
        hidden_dim=hidden, mlp_dim=4 * hidden, num_classes=5, dropout=0.0, attention_dropout=0.0,
    )
    apply_fact(fact_tt, variant="tt", rank=4)
    assert sum(p.numel() for p in fact_tt.fact_bank.parameters()) == 2 * hidden * 4 + (12 * layers) * 4 * 4

    fact_tk = VisionTransformer(
        image_size=32, patch_size=8, num_layers=layers, num_heads=2,
        hidden_dim=hidden, mlp_dim=4 * hidden, num_classes=5, dropout=0.0, attention_dropout=0.0,
    )
    apply_fact(fact_tk, variant="tk", rank=8)
    assert sum(p.numel() for p in fact_tk.fact_bank.parameters()) == 2 * hidden * 8 + 8 ** 3 + 8 * (12 * layers)


def test_spt_adapter_keeps_torchvision_mha_out_projection_callable():
    torch.manual_seed(19)
    model = tiny_vit().eval()
    loader = DataLoader(
        TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([0, 1])),
        batch_size=2, shuffle=False,
    )
    report = calibrate_spt(
        model, loader, torch.device("cpu"), variant="adapter",
        budget=10 ** 9, sensitivity_samples=2, adapter_dim=2, rank=2,
    )
    for block in model.encoder.layers:
        assert isinstance(block.self_attention.out_proj, torch.nn.Linear)
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 5)
    assert report["structured_matrices"] > 0


def test_fair_runner_uses_paper_canonical_fact_and_spt_settings():
    args = SimpleNamespace(ra_pretrained_checkpoint="")
    tt = method_variant("fact_tt", "vit", 0, "/tmp/head.pth", args)
    tk = method_variant("fact_tk", "vit", 0, "/tmp/head.pth", args)
    spt = method_variant("spt_lora", "vit", 0, "/tmp/head.pth", args)
    vqt = method_variant("vqt", "vit", 0, "/tmp/head.pth", args)
    assert tt["fact_rank"] == 4
    assert tk["fact_rank"] == 8
    assert spt["spt_sensitivity_samples"] == 800
    assert "head_from" not in vqt


def test_cli_exposes_explicit_paper_hyperparameters_without_false_defaults():
    args = get_args_parser().parse_args([])
    assert args.fact_rank == 0  # resolves by variant: TT=4, TK=8
    assert args.spt_sensitivity_samples == 800
    assert args.vqt_query_length == 1
    assert args.convpass_dim == 8
    assert args.arc_dim == 50
    assert args.arc_dropout == 0.1
    assert args.arc_merge is True
