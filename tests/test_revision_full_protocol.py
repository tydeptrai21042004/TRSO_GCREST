from __future__ import annotations

from tools.revision_full_protocol import EXPECTED_MAIN_RUNS, SETTINGS, VIT_METHODS, command_for
from models.model_support import (
    PAPER_BASELINE_METHODS,
    ENGINEERING_CONTROL_METHODS,
    TRANSFERRED_CONTROL_METHODS,
    PAPER_ABLATION_METHODS,
    static_method_compatibility,
)


def _setting(key: str):
    return next(item for item in SETTINGS if item.key == key)


def test_revision_protocol_exact_204_run_contract():
    assert len(SETTINGS) == 11
    assert EXPECTED_MAIN_RUNS == 204
    assert sum(len(item.methods) * 3 for item in SETTINGS) == 204
    assert all("trso" in item.methods for item in SETTINGS)
    assert all(item.methods[0:2] == ("full", "linear") for item in SETTINGS)


def test_revision_protocol_exact_method_matrix():
    assert _setting("dtd_resnet50").methods == (
        "full", "linear", "prompt", "conv", "piggyback", "trso"
    )
    assert _setting("dtd_vit_b16").methods == VIT_METHODS
    assert _setting("flowers_vit_b16").methods == VIT_METHODS
    assert VIT_METHODS == (
        "full", "linear", "ssf", "adaptformer", "repadapter", "arc",
        "vpt_shallow", "vpt_deep", "convpass", "fact_tt", "fact_tk", "vqt",
        "spt_lora", "spt_adapter", "trso",
    )
    assert _setting("dtd_resnet18").methods == ("full", "linear", "prompt", "trso")
    assert _setting("dtd_swin_t").methods == ("full", "linear", "ssf", "trso")
    assert _setting("flowers_resnet18").methods == ("full", "linear", "prompt", "trso")
    assert _setting("flowers_swin_t").methods == ("full", "linear", "ssf", "trso")
    assert _setting("pet_resnet18").methods == ("full", "linear", "prompt", "trso")
    assert _setting("pet_swin_t").methods == ("full", "linear", "ssf", "trso")
    assert _setting("voc2007_mobilenetv3_small").methods == ("full", "linear", "ml_decoder", "trso")
    assert _setting("pet_segmentation_lraspp_mobilenetv3_large").methods == (
        "full", "linear", "segadapter", "trso"
    )


def test_revision_protocol_uses_manuscript_batch_sizes_and_seed_contract():
    expected = {
        "dtd_resnet50": 16,
        "dtd_vit_b16": 8,
        "dtd_resnet18": 32,
        "dtd_swin_t": 32,
        "flowers_resnet18": 32,
        "flowers_vit_b16": 16,
        "flowers_swin_t": 32,
        "pet_resnet18": 32,
        "pet_swin_t": 32,
        "voc2007_mobilenetv3_small": 32,
        "pet_segmentation_lraspp_mobilenetv3_large": 8,
    }
    assert {item.key: item.batch_size for item in SETTINGS} == expected
    for item in SETTINGS:
        cmd = command_for(item, output_root="out", data_path="data", download="auto")
        joined = " ".join(cmd)
        assert "--seeds 0,1,2" in joined
        assert "--split_seed 0" in joined
        assert "--epochs 30" in joined
        assert "--warmup_epochs 3" in joined
        assert "--input_size 224" in joined


def test_every_nonreference_main_method_is_compatible_or_proposal():
    for item in SETTINGS:
        backbone = item.backbone.split("@", 1)[0]
        for method in item.methods:
            if method in {"full", "linear", "trso"}:
                continue
            ok, reason, _ = static_method_compatibility(method, backbone, item.task)
            assert ok, f"{item.key}/{method}: {reason}"


def test_main_paper_registry_excludes_local_controls_and_includes_revision_baselines():
    assert {"fact_tt", "fact_tk", "vqt", "spt_lora", "spt_adapter", "ml_decoder", "segadapter"} <= set(PAPER_BASELINE_METHODS)
    assert not set(PAPER_BASELINE_METHODS).intersection(ENGINEERING_CONTROL_METHODS)
    assert not set(PAPER_BASELINE_METHODS).intersection(TRANSFERRED_CONTROL_METHODS)
    assert not set(PAPER_BASELINE_METHODS).intersection(PAPER_ABLATION_METHODS)
    assert "lora" not in PAPER_BASELINE_METHODS
    assert "bitfit" not in PAPER_BASELINE_METHODS
    assert "sidetune" not in PAPER_BASELINE_METHODS
    assert "convpass_attn" not in PAPER_BASELINE_METHODS
