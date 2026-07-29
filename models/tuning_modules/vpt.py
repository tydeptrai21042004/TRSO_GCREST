"""Visual Prompt Tuning (ECCV 2022) for plain Vision Transformers.

This is a clean-room implementation of VPT-Shallow and VPT-Deep.  Prompt
vectors are inserted in token space while the pre-trained Transformer remains
frozen.  VPT-Deep replaces the prompt tokens before every Transformer block;
VPT-Shallow inserts one shared prompt set only at the input.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .vit_paper_utils import (
    identify_plain_vit, timm_embed_tokens, timm_forward_head,
    torchvision_embed_tokens, torchvision_forward_head,
)


class VisualPromptTuning(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_tokens: int = 10,
        deep: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(num_tokens) <= 0:
            raise ValueError("VPT requires at least one prompt token.")
        self.backbone = backbone
        self.layout = identify_plain_vit(backbone)
        self.num_tokens = int(num_tokens)
        self.deep = bool(deep)
        self.prompt_dropout = nn.Dropout(float(dropout))
        depth = len(self.layout.blocks) if self.deep else 1
        self.prompt_embeddings = nn.Parameter(
            torch.empty(depth, self.num_tokens, self.layout.hidden_dim)
        )
        # The VPT implementation initializes prompt vectors with a Xavier-like
        # uniform scale based on patch area and hidden width.
        patch = getattr(getattr(backbone, "patch_embed", None), "patch_size", None)
        if isinstance(patch, tuple):
            patch_area = int(patch[0]) * int(patch[1])
        elif patch is not None:
            patch_area = int(patch) ** 2
        else:
            conv = getattr(backbone, "conv_proj", None)
            kernel = getattr(conv, "kernel_size", (16, 16))
            patch_area = int(kernel[0]) * int(kernel[1])
        bound = math.sqrt(6.0 / float(3 * patch_area + self.layout.hidden_dim))
        nn.init.uniform_(self.prompt_embeddings, -bound, bound)

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self._head_parameters():
            parameter.requires_grad_(True)

    def _head_parameters(self):
        for name, parameter in self.backbone.named_parameters():
            if (
                name.startswith("head.") or name.startswith("heads.")
                or name.startswith("fc.") or name.startswith("classifier.")
            ):
                yield parameter

    def _prompts(self, layer: int, batch: int) -> torch.Tensor:
        index = layer if self.deep else 0
        return self.prompt_dropout(
            self.prompt_embeddings[index].unsqueeze(0).expand(batch, -1, -1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.layout.kind == "timm":
            tokens = timm_embed_tokens(self.backbone, images)
            prefix = self.layout.num_prefix_tokens
            prompts = self._prompts(0, tokens.shape[0])
            tokens = torch.cat((tokens[:, :prefix], prompts, tokens[:, prefix:]), dim=1)
            for index, block in enumerate(self.backbone.blocks):
                if self.deep and index > 0:
                    prompts = self._prompts(index, tokens.shape[0])
                    tokens = torch.cat(
                        (tokens[:, :prefix], prompts, tokens[:, prefix + self.num_tokens:]),
                        dim=1,
                    )
                tokens = block(tokens)
            norm = getattr(self.backbone, "norm", nn.Identity())
            tokens = norm(tokens)
            # Remove prompts before delegating pooling/head behavior to timm.
            tokens = torch.cat((tokens[:, :prefix], tokens[:, prefix + self.num_tokens:]), dim=1)
            return timm_forward_head(self.backbone, tokens)

        tokens = torchvision_embed_tokens(self.backbone, images)
        prompts = self._prompts(0, tokens.shape[0])
        tokens = torch.cat((tokens[:, :1], prompts, tokens[:, 1:]), dim=1)
        for index, block in enumerate(self.backbone.encoder.layers):
            if self.deep and index > 0:
                prompts = self._prompts(index, tokens.shape[0])
                tokens = torch.cat(
                    (tokens[:, :1], prompts, tokens[:, 1 + self.num_tokens:]), dim=1
                )
            tokens = block(tokens)
        tokens = torch.cat((tokens[:, :1], tokens[:, 1 + self.num_tokens:]), dim=1)
        return torchvision_forward_head(self.backbone, tokens)


def set_vpt_trainability(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        trainable = (
            name.startswith("prompt_embeddings")
            or name.startswith("backbone.head.")
            or name.startswith("backbone.heads.")
            or name.startswith("backbone.fc.")
            or name.startswith("backbone.classifier.")
        )
        parameter.requires_grad_(trainable)
