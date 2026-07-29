"""Offline controlled validation for G-CREST-TRSO.

This script compares the active proposal with a test-only historical fixed-rank
fixed-rank reference on deterministic source-to-target texture shifts. It is not used
by production training and does not expose proposal controls in ``main.py``.

Examples
--------
python -m tools.validate_gcrest_controlled --suite standard --seeds 0
python -m tools.validate_gcrest_controlled --suite hard --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, TensorDataset

from models.tuning_modules.mdl_tangent_core import calibrate_mdl_tangent_core


class FixedCore(nn.Module):
    def __init__(self, left: Tensor, right: Tensor, shape: Iterable[int]):
        super().__init__()
        self.shape = tuple(int(v) for v in shape)
        self.register_buffer("left", left.detach().contiguous())
        self.register_buffer("right", right.detach().contiguous())
        self.core = nn.Parameter(torch.zeros(left.shape[1], left.shape[1]))

    def forward(self, weight: Tensor) -> Tensor:
        return weight + (self.left @ self.core @ self.right.T).reshape(self.shape).to(weight.dtype)


def _is_head(name: str) -> bool:
    parts = name.lower().split(".")
    return "head" in parts[:-1] or (len(parts) == 2 and parts[0] == "fc")


def _eligible(model: nn.Module):
    norm_types = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm,
        nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        nn.Embedding,
    )
    for module_name, module in model.named_modules():
        if isinstance(module, norm_types):
            continue
        for pname, parameter in module.named_parameters(recurse=False):
            name = f"{module_name}.{pname}" if module_name else pname
            if parameter.ndim >= 2 and parameter.is_floating_point() and not _is_head(name):
                yield module, pname, parameter


def attach_fixed_reference(model: nn.Module, loader, rank: int, batches: int) -> int:
    candidates = list(_eligible(model))
    original = {id(p): p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for _, _, p in candidates:
        p.requires_grad_(True)
    sums = [torch.zeros_like(p) for _, _, p in candidates]
    count = 0
    model.train()
    for x, y in loader:
        if count >= batches:
            break
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
        for index, (_, _, p) in enumerate(candidates):
            if p.grad is not None:
                sums[index].add_(p.grad.detach())
        count += 1
    if count == 0:
        raise RuntimeError("fixed-reference calibration received no batches")
    attached = 0
    for (module, pname, parameter), gradient in zip(candidates, sums):
        matrix = (gradient / count).float().reshape(parameter.shape[0], -1)
        resolved = min(rank, matrix.shape[0], matrix.shape[1])
        left, _, right_h = torch.linalg.svd(matrix, full_matrices=False)
        parametrize.register_parametrization(
            module,
            pname,
            FixedCore(left[:, :resolved], right_h[:resolved].T, parameter.shape),
        )
        attached += resolved * resolved
    for p in model.parameters():
        p.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, FixedCore):
            module.core.requires_grad_(True)
    for name, p in model.named_parameters():
        if ".parametrizations." not in name and _is_head(name):
            p.requires_grad_(True)
    for p in model.parameters():
        if id(p) in original and not p.requires_grad:
            p.requires_grad_(False)
    return attached


def _set_frozen_batchnorm_eval(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            affine_trainable = any(
                p is not None and p.requires_grad for p in (module.weight, module.bias)
            )
            if not affine_trainable:
                module.eval()


def train(model: nn.Module, loader, epochs: int, lr: float, *, freeze_bn: bool = False) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        if freeze_bn:
            _set_frozen_batchnorm_eval(model)
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            optimizer.step()


def accuracy(model: nn.Module, loader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += int((model(x).argmax(1) == y).sum())
            total += int(y.numel())
    return 100.0 * correct / total


def freeze_except_head(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.head.parameters():
        p.requires_grad_(True)


class CNNBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.max_pool2d(F.gelu(self.norm(self.conv(x))), 2)


class TinyCNN(nn.Module):
    def __init__(self, classes: int, widths=(16, 32, 48)):
        super().__init__()
        blocks = []
        cin = 3
        for cout in widths:
            blocks.append(CNNBlock(cin, cout))
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(widths[-1], classes)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean((2, 3)))


class Attention(nn.Module):
    def __init__(self, dim=36, heads=3):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)).softmax(-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(b, n, c))


class TransformerBlock(nn.Module):
    def __init__(self, dim=36, heads=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 3 * dim), nn.GELU(), nn.Linear(3 * dim, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    def __init__(self, classes=6, dim=36, depth=3, size=24, patch=4):
        super().__init__()
        tokens = (size // patch) ** 2
        self.patch = nn.Conv2d(3, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, tokens + 1, dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0]))


def make_textures(n: int, seed: int, domain: str, *, classes: int, hard: bool):
    generator = torch.Generator().manual_seed(seed)
    size = 24
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    images, labels = [], []
    angles = torch.linspace(0, 180 - 180 / classes, classes)
    frequencies = torch.linspace(2.3, 4.4, classes)
    colors = torch.rand(classes, 3, generator=torch.Generator().manual_seed(991)) * 0.75 + 0.2
    for index in range(n):
        label = index % classes
        theta = math.radians(float(angles[label]))
        frequency = float(frequencies[label])
        phase = float(torch.rand((), generator=generator) * 2 * math.pi)
        x, y = xx.clone(), yy.clone()
        if domain == "target":
            theta += math.radians(17 if hard else 12)
            frequency *= 1.22 if hard else 1.18
            x = x + (0.18 if hard else 0.12) * torch.sin(2 * math.pi * y + phase)
            y = y + (0.12 if hard else 0.08) * torch.sin(2 * math.pi * x - phase)
        u = x * math.cos(theta) + y * math.sin(theta)
        v = -x * math.sin(theta) + y * math.cos(theta)
        pattern = torch.sin(2 * math.pi * frequency * u + phase)
        pattern += 0.5 * torch.cos(2 * math.pi * frequency * 0.6 * v - phase / 3)
        color = colors[label]
        if domain == "target":
            color = color[[2, 0, 1]]
        image = pattern.unsqueeze(0) * color[:, None, None]
        image += (0.32 if hard else 0.20) * torch.randn((3, size, size), generator=generator)
        if hard and domain == "target" and index % 3 == 0:
            x0 = int(torch.randint(0, 14, (), generator=generator))
            y0 = int(torch.randint(0, 14, (), generator=generator))
            image[:, y0:y0 + 8, x0:x0 + 8] *= 0.3
        image = (image - image.mean((1, 2), keepdim=True)) / (image.std((1, 2), keepdim=True) + 1e-5)
        images.append(image)
        labels.append(label)
    order = torch.randperm(n, generator=generator)
    return torch.stack(images)[order], torch.tensor(labels)[order]


def loaders(seed: int, hard: bool):
    classes = 8 if hard else 6
    sizes = (640, 240, 192, 800) if hard else (1200, 300, 600, 600)
    specs = [
        (sizes[0], seed, "source", True),
        (sizes[1], seed + 1, "source", False),
        (sizes[2], seed + 2, "target", True),
        (sizes[3], seed + 3, "target", False),
    ]
    batch = 32 if hard else 64
    return [
        DataLoader(
            TensorDataset(*make_textures(n, data_seed, domain, classes=classes, hard=hard)),
            batch_size=batch,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(seed + 100 + index),
        )
        for index, (n, data_seed, domain, shuffle) in enumerate(specs)
    ]


@dataclass
class Row:
    suite: str
    architecture: str
    seed: int
    fixed_accuracy: float
    gcrest_accuracy: float
    fixed_adapter_parameters: int
    gcrest_adapter_parameters: int
    gcrest_selected_tensors: int
    

def run(seed: int, suite: str, architecture: str) -> Row:
    hard = suite == "hard"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    source, _, warm_target, test = loaders(seed, hard)
    classes = 8 if hard else 6
    if hard:
        factory = lambda: TinyCNN(classes, widths=(10, 16, 24))
    elif architecture == "cnn":
        factory = lambda: TinyCNN(classes)
    else:
        factory = lambda: TinyViT(classes=classes)
    base = factory()
    train(base, source, 6 if hard else 8, 2e-3)
    freeze_except_head(base)
    train(base, warm_target, 2 if hard else 3, 5e-3)
    state = copy.deepcopy(base.state_dict())

    fixed_loader = loaders(seed, hard)[2]
    fixed = factory()
    fixed.load_state_dict(state)
    freeze_except_head(fixed)
    rank = 3 if hard else 2
    fixed_params = attach_fixed_reference(fixed, fixed_loader, rank=rank, batches=6)
    train(fixed, fixed_loader, 8 if hard else 7, 8e-3, freeze_bn=True)

    corrected_loader = loaders(seed, hard)[2]
    gcrest = factory()
    gcrest.load_state_dict(state)
    freeze_except_head(gcrest)
    report = calibrate_mdl_tangent_core(
        gcrest, corrected_loader, nn.CrossEntropyLoss(), device="cpu"
    )
    train(gcrest, corrected_loader, 8 if hard else 7, 8e-3, freeze_bn=True)
    return Row(
        suite=suite,
        architecture=architecture,
        seed=seed,
        fixed_accuracy=accuracy(fixed, test),
        gcrest_accuracy=accuracy(gcrest, test),
        fixed_adapter_parameters=fixed_params,
        gcrest_adapter_parameters=report.adapter_parameters,
        gcrest_selected_tensors=report.selected_tensors,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("standard", "hard"), default="standard")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    architectures = ("cnn",) if args.suite == "hard" else ("cnn", "vit")
    rows = [
        run(seed, args.suite, architecture)
        for seed in [int(v) for v in args.seeds.split(",") if v.strip()]
        for architecture in architectures
    ]
    payload = [asdict(row) for row in rows]
    print(json.dumps(payload, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
