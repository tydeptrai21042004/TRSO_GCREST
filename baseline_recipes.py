"""Canonical baseline provenance and paper-faithful method defaults.

This module is deliberately independent of the training stack so that runners,
unit tests, documentation generators, and result auditors can share one source
of truth.

Two different notions of fairness are kept separate:

``controlled``
    All methods use the same outer training protocol.  Paper defaults are used
    only for method-internal structure (rank, bottleneck, prompt/query layout,
    etc.).  This is the recommended main-table comparison against TRSO.

``paper_paired``
    A baseline is run with the closest source-verified paper/official-repo
    training recipe available in this repository and TRSO is run again with
    the *same outer recipe*.  This answers whether a conclusion survives the
    baseline authors' optimization protocol without giving either method an
    optimizer/schedule advantage.

A recipe is never called exact when the public source exposes dataset-specific
hyperparameter files or a search procedure that is not fully encoded here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any, Mapping


@dataclass(frozen=True)
class BaselineRecipe:
    method: str
    paper: str
    official_source: str
    mechanism_fidelity: str
    architecture_fidelity: str
    training_fidelity: str
    original_scope: str
    method_defaults: Mapping[str, Any] = field(default_factory=dict)
    optimizer: str | None = None
    scheduler: str = "cosine"
    epochs: int | None = None
    batch_size: int | None = None
    base_lr: float | None = None
    scale_lr_by_batch_256: bool = False
    weight_decay: float | None = None
    warmup_epochs: int | None = None
    min_lr: float | None = None
    lr_search: tuple[float, ...] = ()
    wd_search: tuple[float, ...] = ()
    notes: str = ""

    @property
    def has_source_verified_outer_recipe(self) -> bool:
        return self.optimizer is not None and self.epochs is not None and (
            self.base_lr is not None or bool(self.lr_search)
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["has_source_verified_outer_recipe"] = self.has_source_verified_outer_recipe
        return payload


# Method-internal defaults.  These are used in *both* controlled and paired
# protocols so that a common optimizer recipe does not silently change the
# baseline mechanism/capacity.
PAPER_METHOD_DEFAULTS: dict[str, dict[str, Any]] = {
    "prompt": {
        "prompt_size": 30,
        "prompt_type": "padding",
        "prompt_mapping": "frequency",
        "prompt_mapping_batches": 0,
    },
    "conv": {
        "adapt_size": 8,
        "kernel_size": 3,
        "adapt_scale": 1.0,
        "conv_adapter_mode": "conv_parallel",
    },
    "piggyback": {
        "piggyback_threshold": 5e-3,
        "piggyback_mask_init": "ones",
    },
    "ssf": {
        "ssf_init_scale": 1.0,
        "ssf_init_shift": 0.0,
        "ssf_init_std": 0.02,
    },
    # Official AdaptFormer image code defaults to ffn_num=64 and scalar=0.1.
    "adaptformer": {
        "adaptformer_dim": 64,
        "adaptformer_scale": 0.1,
        "adaptformer_dropout": 0.0,
        "adaptformer_layernorm": "none",
    },
    "repadapter": {
        "repadapter_dim": 8,
        "repadapter_groups": 2,
        "repadapter_scale": 1.0,
        "repadapter_dropout": 0.1,
        "repadapter_merge": True,
    },
    "arc": {"arc_dim": 50, "arc_dropout": 0.1, "arc_merge": True},
    # VPT prompt length is dataset/config specific in the official release.
    # Ten is retained only as the repository fallback and is explicitly marked
    # non-exact in the recipe metadata below.
    "vpt_shallow": {"vpt_num_tokens": 10, "vpt_dropout": 0.0},
    "vpt_deep": {"vpt_num_tokens": 10, "vpt_dropout": 0.0},
    "convpass": {"convpass_dim": 8, "convpass_scale": 1.0, "convpass_dropout": 0.1},
    "convpass_attn": {"convpass_dim": 8, "convpass_scale": 1.0, "convpass_dropout": 0.1},
    "fact_tt": {"fact_rank": 4, "fact_scale": 1.0},
    "fact_tk": {"fact_rank": 8, "fact_scale": 1.0},
    "vqt": {"vqt_query_length": 1},
    "spt_lora": {
        "spt_budget": 400000,
        "spt_sensitivity_samples": 800,
        "spt_rank": 8,
        "spt_adapter_dim": 8,
        "spt_alpha": 8.0,
    },
    "spt_adapter": {
        "spt_budget": 400000,
        "spt_sensitivity_samples": 800,
        "spt_rank": 8,
        "spt_adapter_dim": 8,
        "spt_alpha": 8.0,
    },
    "ml_decoder": {
        "ml_decoder_embedding": 768,
        "ml_decoder_num_groups": -1,
        "ml_decoder_dropout": 0.1,
    },
    "segadapter": {
        "segadapter_kernel_size": 5,
        "segadapter_ffn_ratio": 3.0,
        "segadapter_aux_weight": 0.4,
    },
}


# VPT official tune_vtab.py searches these values and scales LR by batch/256.
# The exact prompt length and selected LR/WD are dataset-specific; therefore the
# recipe is correctly labelled ``source_search`` rather than ``exact_fixed``.
_VPT_LR_SEARCH = (50.0, 25.0, 10.0, 5.0, 2.5, 1.0, 0.5, 0.25, 0.1, 0.05)
_VPT_WD_SEARCH = (0.01, 0.001, 0.0001, 0.0)


RECIPES: dict[str, BaselineRecipe] = {
    "prompt": BaselineRecipe(
        "prompt",
        "Bahng et al., Exploring Visual Prompts for Adapting Large-Scale Models",
        "https://github.com/hjbahng/visual_prompting",
        "close", "paper_backbone_family", "source_audited_partial",
        "Single-label visual recognition with a frozen source classifier.",
        PAPER_METHOD_DEFAULTS["prompt"],
        notes="The prompt mechanism/mapping is source aligned; dataset-specific paper training settings must be supplied when reproducing a reported table exactly.",
    ),
    "conv": BaselineRecipe(
        "conv", "Chen et al., Conv-Adapter",
        "published implementation/paper configuration",
        "close", "paper_resnet50", "source_audited_partial",
        "ResNet-50 single-label classification.", PAPER_METHOD_DEFAULTS["conv"],
    ),
    "piggyback": BaselineRecipe(
        "piggyback", "Mallya et al., Piggyback", "https://github.com/arunmallya/piggyback",
        "close", "paper_cnn", "source_audited_partial",
        "Single-label CNN transfer with binary masks and a task classifier.", PAPER_METHOD_DEFAULTS["piggyback"],
    ),
    "ssf": BaselineRecipe(
        "ssf", "Lian et al., Scaling & Shifting Your Features", "https://github.com/dongzelian/SSF",
        "close", "paper_vit_swin_convnext", "official_script_required",
        "FGVC/VTAB/ImageNet transfer with the official pre-trained families.", PAPER_METHOD_DEFAULTS["ssf"],
        notes="Official repository contains dataset/backbone-specific scripts; do not collapse all of them into a single claimed paper-exact LR.",
    ),
    "adaptformer": BaselineRecipe(
        "adaptformer", "Chen et al., AdaptFormer", "https://github.com/ShoufaChen/AdaptFormer",
        "close", "paper_vit", "exact_official_default",
        "Image classification with ViT image code.", PAPER_METHOD_DEFAULTS["adaptformer"],
        optimizer="sgd", scheduler="cosine", epochs=100, batch_size=512,
        base_lr=0.1, scale_lr_by_batch_256=True, weight_decay=0.0,
        warmup_epochs=20, min_lr=0.0,
        notes="Official image-code defaults: ffn_num=64, scalar=0.1, base LR 0.1 scaled by effective batch/256, WD=0, 100 epochs, 20 warmup epochs.",
    ),
    "repadapter": BaselineRecipe(
        "repadapter", "Luo et al., RepAdapter", "https://github.com/luogen1996/RepAdapter",
        "close", "paper_vit_b16", "paper_search_required",
        "ViT-B/16 image transfer.", PAPER_METHOD_DEFAULTS["repadapter"],
        notes="The authors tune hyperparameters per task. The paired runner therefore refuses to call a generic fixed LR paper-exact.",
    ),
    "arc": BaselineRecipe(
        "arc", "Dong et al., ARC", "https://github.com/dongzelian/ARC",
        "close", "paper_vit", "official_script_required",
        "VTAB/FGVC visual recognition.", PAPER_METHOD_DEFAULTS["arc"],
    ),
    "vpt_shallow": BaselineRecipe(
        "vpt_shallow", "Jia et al., Visual Prompt Tuning", "https://github.com/KMnP/vpt",
        "close", "paper_vit", "source_search",
        "VTAB/FGVC; dataset-specific prompt length and LR/WD selection.", PAPER_METHOD_DEFAULTS["vpt_shallow"],
        optimizer="sgd", scheduler="cosine", epochs=30, batch_size=32,
        weight_decay=1e-4, warmup_epochs=5, min_lr=0.0,
        lr_search=_VPT_LR_SEARCH, wd_search=_VPT_WD_SEARCH,
        scale_lr_by_batch_256=True,
        notes="Official VPT code performs an 800/200 VTAB tuning split and searches LR/WD; prompt length is dataset specific.",
    ),
    "vpt_deep": BaselineRecipe(
        "vpt_deep", "Jia et al., Visual Prompt Tuning", "https://github.com/KMnP/vpt",
        "close", "paper_vit", "source_search",
        "VTAB/FGVC; dataset-specific prompt length and LR/WD selection.", PAPER_METHOD_DEFAULTS["vpt_deep"],
        optimizer="sgd", scheduler="cosine", epochs=30, batch_size=32,
        weight_decay=1e-4, warmup_epochs=5, min_lr=0.0,
        lr_search=_VPT_LR_SEARCH, wd_search=_VPT_WD_SEARCH,
        scale_lr_by_batch_256=True,
        notes="Official VPT code performs an 800/200 VTAB tuning split and searches LR/WD; prompt length is dataset specific.",
    ),
    "convpass": BaselineRecipe(
        "convpass", "Jie & Deng, Convolutional Bypasses Are Better Vision Transformer Adapters",
        "https://github.com/JieShibo/PETL-ViT",
        "close", "paper_vit_b16", "official_config_required",
        "ViT-B/16 VTAB visual recognition.", PAPER_METHOD_DEFAULTS["convpass"],
    ),
    "fact_tt": BaselineRecipe(
        "fact_tt", "Jie & Deng, FacT", "https://github.com/JieShibo/PETL-ViT",
        "close", "paper_vit_b16", "official_config_required",
        "ViT-B/16 VTAB visual recognition.", PAPER_METHOD_DEFAULTS["fact_tt"],
    ),
    "fact_tk": BaselineRecipe(
        "fact_tk", "Jie & Deng, FacT", "https://github.com/JieShibo/PETL-ViT",
        "close", "paper_vit_b16", "official_config_required",
        "ViT-B/16 VTAB visual recognition.", PAPER_METHOD_DEFAULTS["fact_tk"],
    ),
    "vqt": BaselineRecipe(
        "vqt", "Tu et al., Visual Query Tuning", "https://github.com/andytu28/VQT",
        "close", "paper_vit_b16", "optimizer_verified_config_required",
        "VTAB-1K with ViT-B/16; query length 1; Adam optimizer in official experiments.", PAPER_METHOD_DEFAULTS["vqt"],
        optimizer="adam",
        notes="Official README verifies Adam and query length 1. Dataset-specific LR/WD should come from the official config/tuning output rather than an invented universal value.",
    ),
    "spt_lora": BaselineRecipe(
        "spt_lora", "He et al., Sensitivity-Aware Visual PEFT", "https://github.com/ziplab/SPT",
        "close", "paper_vit_b16", "official_config_required",
        "VTAB-1K two-stage sensitivity then PEFT.", PAPER_METHOD_DEFAULTS["spt_lora"],
    ),
    "spt_adapter": BaselineRecipe(
        "spt_adapter", "He et al., Sensitivity-Aware Visual PEFT", "https://github.com/ziplab/SPT",
        "close", "paper_vit_b16", "official_config_required",
        "VTAB-1K two-stage sensitivity then PEFT.", PAPER_METHOD_DEFAULTS["spt_adapter"],
    ),
    "ml_decoder": BaselineRecipe(
        "ml_decoder", "Ridnik et al., ML-Decoder", "https://github.com/Alibaba-MIIL/ML_Decoder",
        "close", "transferred_backbone", "transferred_controlled",
        "Published head mechanism; repository revision evaluates VOC2007/MobileNetV3-S rather than an original reported setup.", PAPER_METHOD_DEFAULTS["ml_decoder"],
        notes="Do not label the VOC2007/MobileNetV3-S row an exact paper reproduction.",
    ),
    "segadapter": BaselineRecipe(
        "segadapter", "Peng & Kameyama, SegAdapter", "paper-equation reimplementation",
        "paper_equation_reimplementation", "transferred_backbone", "transferred_controlled",
        "Published segmentation adapter equations transferred to LR-ASPP/MobileNetV3-L.", PAPER_METHOD_DEFAULTS["segadapter"],
        notes="The mechanism is paper derived, but the revision backbone/dataset combination is a transfer evaluation.",
    ),
}


def paper_method_defaults(method: str) -> dict[str, Any]:
    """Return a copy of the source-audited method-internal defaults."""
    return dict(PAPER_METHOD_DEFAULTS.get(str(method), {}))


def recipe_for(method: str) -> BaselineRecipe | None:
    return RECIPES.get(str(method))


def recipe_manifest() -> dict[str, dict[str, Any]]:
    return {method: recipe.to_dict() for method, recipe in sorted(RECIPES.items())}


def paper_outer_trials(
    method: str,
    *,
    batch_size: int | None = None,
    search_mode: str = "full",
) -> list[dict[str, Any]]:
    """Return source-verified outer optimization trials for ``method``.

    ``search_mode='full'`` reproduces an encoded official search grid.
    ``search_mode='compact'`` keeps the endpoints/central values for a cheaper
    diagnostic but is explicitly tagged non-exact.  Fixed recipes return one
    trial.  Methods whose exact outer recipe is not encoded return an empty list
    instead of silently inventing paper hyperparameters.
    """
    recipe = recipe_for(method)
    if recipe is None or recipe.optimizer is None or recipe.epochs is None:
        return []

    effective_batch = int(batch_size or recipe.batch_size or 32)
    common = {
        "optimizer": recipe.optimizer,
        "scheduler": recipe.scheduler,
        "epochs": int(recipe.epochs),
        "batch_size": effective_batch,
        "warmup_epochs": int(recipe.warmup_epochs or 0),
        "min_lr": float(recipe.min_lr or 0.0),
    }

    def scaled_lr(value: float) -> float:
        if recipe.scale_lr_by_batch_256:
            return float(value) * effective_batch / 256.0
        return float(value)

    if recipe.lr_search:
        lrs = list(recipe.lr_search)
        wds = list(recipe.wd_search or ((recipe.weight_decay if recipe.weight_decay is not None else 0.0),))
        if search_mode == "compact":
            # Deterministic, transparent diagnostic subset. Never call exact.
            lrs = list(dict.fromkeys((lrs[0], lrs[len(lrs) // 2], lrs[-1])))
            wds = list(dict.fromkeys((wds[0], wds[len(wds) // 2], wds[-1])))
        elif search_mode != "full":
            raise ValueError("search_mode must be 'full' or 'compact'")
        trials = []
        for index, (lr, wd) in enumerate(product(lrs, wds)):
            trials.append({
                **common,
                "lr": scaled_lr(float(lr)),
                "weight_decay": float(wd),
                "weight_decay_adapter": float(wd),
                "paper_trial_index": index,
                "paper_search_mode": search_mode,
                "paper_base_lr": float(lr),
            })
        return trials

    if recipe.base_lr is None:
        return []
    wd = float(recipe.weight_decay or 0.0)
    return [{
        **common,
        "lr": scaled_lr(float(recipe.base_lr)),
        "weight_decay": wd,
        "weight_decay_adapter": wd,
        "paper_trial_index": 0,
        "paper_search_mode": "fixed",
        "paper_base_lr": float(recipe.base_lr),
    }]


__all__ = [
    "BaselineRecipe",
    "PAPER_METHOD_DEFAULTS",
    "RECIPES",
    "paper_method_defaults",
    "paper_outer_trials",
    "recipe_for",
    "recipe_manifest",
]
