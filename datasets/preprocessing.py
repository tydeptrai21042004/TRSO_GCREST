"""Shared preprocessing resolution for classification and structured tasks."""
from __future__ import annotations

from dataclasses import asdict, dataclass

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessingProfile:
    input_size: int
    mean: tuple[float, float, float] | None
    std: tuple[float, float, float] | None
    interpolation: str
    crop_ratio: float
    source: str = "repository_default"

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_normalization(args):
    use_norm = bool(getattr(args, "imagenet_norm", getattr(args, "imagenet_default_mean_and_std", True)))
    if not use_norm:
        return None
    mean = getattr(args, "preprocess_mean", None)
    std = getattr(args, "preprocess_std", None)
    if mean is not None and std is not None:
        return tuple(map(float, mean)), tuple(map(float, std))
    return IMAGENET_MEAN, IMAGENET_STD


def current_preprocessing_profile(args) -> PreprocessingProfile:
    norm = resolve_normalization(args)
    interpolation = str(getattr(args, "train_interpolation", "bicubic") or "bicubic").lower()
    return PreprocessingProfile(
        input_size=int(getattr(args, "input_size", 224)),
        mean=None if norm is None else norm[0],
        std=None if norm is None else norm[1],
        interpolation=interpolation,
        crop_ratio=float(getattr(args, "crop_ratio", 0.875) or 0.875),
        source=str(getattr(args, "preprocess_source", "repository_default")),
    )
