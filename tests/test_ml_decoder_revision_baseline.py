from __future__ import annotations

import torch
from torchvision.models import mobilenet_v3_small

from models.tuning_modules.ml_decoder import MLDecoder, MLDecoderClassifier, set_ml_decoder_trainability


def test_ml_decoder_voc_output_and_fixed_queries():
    model = MLDecoderClassifier(
        mobilenet_v3_small(weights=None), 20,
        decoder_embedding=64, num_groups=-1, dropout=0.0,
    )
    set_ml_decoder_trainability(model)
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 20)
    decoder = next(module for module in model.modules() if isinstance(module, MLDecoder))
    assert decoder.num_queries == 20
    assert not decoder.query_embed.weight.requires_grad
    assert any(p.requires_grad for p in model.features.parameters())
    loss = out.square().mean()
    loss.backward()
    assert decoder.duplicate_pooling.grad is not None
    assert decoder.embed_standard.weight.grad is not None


def test_ml_decoder_uses_spatial_features_not_global_pooling():
    model = MLDecoderClassifier(
        mobilenet_v3_small(weights=None), 20,
        decoder_embedding=64, dropout=0.0,
    ).eval()
    seen = {}
    def _record_shape(_module, args):
        seen["shape"] = tuple(args[0].shape)
        return None
    handle = model.ml_decoder.register_forward_pre_hook(_record_shape)
    try:
        _ = model(torch.randn(1, 3, 64, 64))
    finally:
        handle.remove()
    assert len(seen["shape"]) == 4
    assert seen["shape"][2] > 1 and seen["shape"][3] > 1
