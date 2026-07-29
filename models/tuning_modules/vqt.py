"""Visual Query Tuning (CVPR 2023) for plain Vision Transformers.

A separate learnable query set is attached to every frozen Transformer layer.
The queries use that layer's frozen self-attention and MLP to summarize the
unchanged intermediate patch representation.  The per-layer query summaries
are concatenated and fed to a new downstream classifier.  Gradients do not
adapt or alter the backbone token stream.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .vit_paper_utils import (
    identify_plain_vit, timm_embed_tokens, torchvision_embed_tokens,
)


def _timm_query_attention(block: nn.Module, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    attn = block.attn
    normalized = block.norm1(tokens)
    query_normalized = block.norm1(query)
    hidden = normalized.shape[-1]
    heads = int(attn.num_heads)
    head_dim = hidden // heads
    qkv_weight = attn.qkv.weight
    qkv_bias = attn.qkv.bias
    q_bias = None if qkv_bias is None else qkv_bias[:hidden]
    q = F.linear(query_normalized, qkv_weight[:hidden], q_bias)
    kv = F.linear(normalized, qkv_weight, qkv_bias)
    kv = kv.reshape(tokens.shape[0], tokens.shape[1], 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    k, v = kv[1], kv[2]
    q = q.reshape(query.shape[0], query.shape[1], heads, head_dim).permute(0, 2, 1, 3)
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
    if hasattr(attn, "k_norm"):
        k = attn.k_norm(k)
    scale = float(getattr(attn, "scale", head_dim ** -0.5))
    weights = (q * scale) @ k.transpose(-2, -1)
    weights = weights.softmax(dim=-1)
    weights = getattr(attn, "attn_drop", nn.Identity())(weights)
    summary = (weights @ v).transpose(1, 2).reshape(query.shape[0], query.shape[1], hidden)
    summary = getattr(attn, "proj_drop", nn.Identity())(attn.proj(summary))
    query = query + getattr(block, "drop_path1", getattr(block, "drop_path", nn.Identity()))(
        getattr(block, "ls1", nn.Identity())(summary)
    )
    mlp = block.mlp(block.norm2(query))
    mlp = getattr(block, "ls2", nn.Identity())(mlp)
    mlp = getattr(block, "drop_path2", getattr(block, "drop_path", nn.Identity()))(mlp)
    return query + mlp


def _torchvision_query_attention(block: nn.Module, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    normalized = block.ln_1(tokens)
    query_normalized = block.ln_1(query)
    summary, _ = block.self_attention(query_normalized, normalized, normalized, need_weights=False)
    query = query + block.dropout(summary)
    return query + block.mlp(block.ln_2(query))


class VisualQueryTuning(nn.Module):
    def __init__(self, backbone: nn.Module, *, num_classes: int, query_length: int = 1) -> None:
        super().__init__()
        self.backbone = backbone
        self.layout = identify_plain_vit(backbone)
        self.query_length = int(query_length)
        if self.query_length <= 0:
            raise ValueError("VQT query length must be positive.")
        depth = len(self.layout.blocks)
        if self.layout.num_prefix_tokens != 1:
            raise ValueError("Strict VQT reproduction requires exactly one final class token.")
        self.query_tokens = nn.Parameter(
            torch.empty(depth, self.query_length, self.layout.hidden_dim)
        )
        nn.init.normal_(self.query_tokens, std=0.02)
        # Equation (5) of VQT concatenates all layer-wise query summaries with
        # the frozen backbone's final class token before the downstream head.
        feature_tokens = depth * self.query_length + 1
        self.head = nn.Linear(feature_tokens * self.layout.hidden_dim, int(num_classes))
        nn.init.zeros_(self.head.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.01)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        summaries = []
        if self.layout.kind == "timm":
            tokens = timm_embed_tokens(self.backbone, images)
            for index, block in enumerate(self.backbone.blocks):
                query = self.query_tokens[index].unsqueeze(0).expand(tokens.shape[0], -1, -1)
                summaries.append(_timm_query_attention(block, query, tokens))
                # Backbone features remain intact; no query token is inserted into this stream.
                with torch.no_grad():
                    tokens = block(tokens)
        else:
            tokens = torchvision_embed_tokens(self.backbone, images)
            for index, block in enumerate(self.backbone.encoder.layers):
                query = self.query_tokens[index].unsqueeze(0).expand(tokens.shape[0], -1, -1)
                summaries.append(_torchvision_query_attention(block, query, tokens))
                with torch.no_grad():
                    tokens = block(tokens)
        if self.layout.kind == "timm":
            final_tokens = getattr(self.backbone, "norm", nn.Identity())(tokens)
        else:
            final_tokens = self.backbone.encoder.ln(tokens)
        final_cls = final_tokens[:, 0].flatten(1)
        features = torch.cat([item.flatten(1) for item in summaries] + [final_cls], dim=1)
        return self.head(features)


def set_vqt_trainability(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("query_tokens") or name.startswith("head."))
