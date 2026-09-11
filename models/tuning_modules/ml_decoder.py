"""ML-Decoder baseline (Ridnik et al., WACV 2023).

This module is a compact PyTorch port of the public ML-Decoder release.  The
paper's non-ZSL path uses fixed query embeddings, one cross-attention decoder
layer, and grouped classifiers over spatial backbone embeddings.  The wrapper
below keeps that structure while exposing a normal image-classification model
interface for this repository.

Paper: "ML-Decoder: Scalable and Versatile Classification Head".
Official source: https://github.com/Alibaba-MIIL/ML_Decoder (MIT license).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TransformerDecoderLayerOptimal(nn.Module):
    """The cross-attention-only decoder layer used by the official ML-Decoder."""

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        if activation == "gelu":
            self.activation = F.gelu
        else:
            self.activation = F.relu

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        # The public implementation has no self-attention among queries.
        tgt = self.norm1(tgt + self.dropout1(tgt))
        tgt2 = self.multihead_attn(tgt, memory, memory, need_weights=False)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout3(tgt2))


class MLDecoder(nn.Module):
    """Non-zero-shot ML-Decoder classification head.

    The defaults mirror the public implementation: up to 100 fixed queries,
    decoder width 768, one decoder layer, and grouped fully-connected output.
    """

    def __init__(
        self,
        num_classes: int,
        initial_num_features: int,
        num_of_groups: int = -1,
        decoder_embedding: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        num_classes = int(num_classes)
        if num_classes <= 0:
            raise ValueError("ML-Decoder requires a positive number of classes.")
        embed_len = 100 if int(num_of_groups) < 0 else int(num_of_groups)
        embed_len = min(embed_len, num_classes)
        decoder_embedding = int(decoder_embedding)
        if decoder_embedding <= 0:
            decoder_embedding = 768
        nhead = 8
        if decoder_embedding % nhead != 0:
            # Keep multi-head attention valid for custom widths.
            nhead = max(1, math.gcd(decoder_embedding, 8))

        self.num_classes = num_classes
        self.num_queries = embed_len
        self.duplicate_factor = int(math.ceil(num_classes / embed_len))
        self.embed_standard = nn.Linear(int(initial_num_features), decoder_embedding)
        self.query_embed = nn.Embedding(embed_len, decoder_embedding)
        self.query_embed.requires_grad_(False)  # fixed queries in the public non-ZSL path
        self.decoder = TransformerDecoderLayerOptimal(
            decoder_embedding, nhead=nhead, dim_feedforward=2048, dropout=float(dropout)
        )
        self.duplicate_pooling = nn.Parameter(
            torch.empty(embed_len, decoder_embedding, self.duplicate_factor)
        )
        self.duplicate_pooling_bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.xavier_normal_(self.duplicate_pooling)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 4:  # B,C,H,W -> B,N,C
            spatial = x.flatten(2).transpose(1, 2)
        elif x.ndim == 3:
            spatial = x
        else:
            raise ValueError(f"ML-Decoder expects BxCxHxW or BxNxC features, got {tuple(x.shape)}")
        memory = F.relu(self.embed_standard(spatial), inplace=False)
        batch = int(memory.shape[0])
        queries = self.query_embed.weight.unsqueeze(1).expand(-1, batch, -1)
        hidden = self.decoder(queries, memory.transpose(0, 1)).transpose(0, 1)
        # Official GroupFC: each query owns a group of class classifiers.
        grouped = torch.einsum("bqd,qdk->bqk", hidden, self.duplicate_pooling)
        logits = grouped.flatten(1)[:, : self.num_classes]
        return logits + self.duplicate_pooling_bias


class MLDecoderClassifier(nn.Module):
    """Attach ML-Decoder to a CNN spatial feature extractor.

    For the revision protocol this wrapper is used with torchvision
    MobileNetV3-Small on VOC2007.  The baseline follows the paper's end-to-end
    training regime: the backbone and ML-Decoder are both trainable.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        *,
        decoder_embedding: int = 768,
        num_groups: int = -1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not hasattr(backbone, "features"):
            raise TypeError("MLDecoderClassifier requires a CNN exposing a .features module.")
        self.features = backbone.features
        feature_dim: Optional[int] = None
        classifier = getattr(backbone, "classifier", None)
        if isinstance(classifier, nn.Sequential):
            for module in classifier:
                if isinstance(module, nn.Linear):
                    feature_dim = int(module.in_features)
                    break
        if feature_dim is None:
            # Torchvision MobileNetV3 final feature stage exposes out_channels.
            for module in reversed(list(self.features.modules())):
                value = getattr(module, "out_channels", None)
                if value is not None:
                    feature_dim = int(value)
                    break
        if feature_dim is None:
            raise RuntimeError("Could not infer spatial feature dimension for ML-Decoder.")
        self.ml_decoder = MLDecoder(
            num_classes=num_classes,
            initial_num_features=feature_dim,
            num_of_groups=num_groups,
            decoder_embedding=decoder_embedding,
            dropout=dropout,
        )
        self.backbone_family = getattr(backbone, "backbone_family", "cnn")

    def forward(self, images: Tensor) -> Tensor:
        return self.ml_decoder(self.features(images))


def set_ml_decoder_trainability(model: nn.Module) -> None:
    """Paper-style ML-Decoder trains the feature extractor and decoder end-to-end."""
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    # Fixed query embeddings are explicitly non-learnable in the public release.
    for module in model.modules():
        if isinstance(module, MLDecoder):
            module.query_embed.requires_grad_(False)


__all__ = [
    "TransformerDecoderLayerOptimal", "MLDecoder", "MLDecoderClassifier",
    "set_ml_decoder_trainability",
]
