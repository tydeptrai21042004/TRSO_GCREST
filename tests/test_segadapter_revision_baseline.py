from __future__ import annotations

import torch
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from models.structured_models import DenseOutputModel
from models.tuning_modules.segadapter import (
    SegAdapterBlock, SegAdapterStage, apply_segadapter_lraspp, set_segadapter_trainability,
)


def test_segadapter_block_paper_defaults_and_shape():
    block = SegAdapterBlock(16, 3, kernel_size=5, ffn_ratio=3.0, mu_init=1e-5)
    x = torch.randn(2, 16, 12, 12)
    y = block(x)
    assert y.shape == x.shape
    assert block.side_classifier(x).shape == (2, 3, 12, 12)
    assert torch.allclose(block.mu.detach(), torch.full_like(block.mu.detach(), 1e-5))
    assert block.ffn[0].out_channels == 48


def test_segadapter_lraspp_install_and_auxiliary_training_contract():
    tv = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=3)
    dense = DenseOutputModel(tv)
    report = apply_segadapter_lraspp(dense, num_classes=3, input_size=64)
    set_segadapter_trainability(dense)
    assert len(report.stage_names) == 4
    assert report.mu_init == (1e-5, 1e-4, 1e-3, 1e-2)
    assert report.kernel_size == 5
    assert report.ffn_ratio == 3.0
    assert report.aux_weight == 0.4
    assert sum(isinstance(m, SegAdapterStage) for m in dense.modules()) == 4
    assert all(p.requires_grad for p in dense.parameters())

    dense.train()
    out = dense(torch.randn(1, 3, 64, 64))
    assert out["out"].shape == (1, 3, 64, 64)
    assert out["aux"].shape == (1, 3, 64, 64)
    assert out["aux_weight"] == 0.4

    dense.eval()
    with torch.no_grad():
        eval_out = dense(torch.randn(1, 3, 64, 64))
    assert set(eval_out) == {"out"}
