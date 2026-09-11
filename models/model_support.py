"""Backbone-family detection and baseline compatibility contracts.

The compatibility table is deliberately conservative. A baseline is not
silently rewritten to operate on a different representation family. Unsupported
method/backbone pairs fail before training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Optional

import torch.nn as nn

from task_registry import (
    ALL_TASKS, CLASSIFICATION_TASKS, TASK_DEPTH_ESTIMATION,
    TASK_OBJECT_DETECTION, TASK_SEMANTIC_SEGMENTATION, TASK_SINGLE_LABEL, normalize_task,
)


ALL_FAMILIES = frozenset({"resnet", "cnn", "vit", "swin", "transformer", "mlp", "clip_cnn", "clip_transformer"})
DENSE_TASKS = frozenset({TASK_SEMANTIC_SEGMENTATION, TASK_DEPTH_ESTIMATION})
NON_DETECTION_TASKS = frozenset(set(ALL_TASKS) - {TASK_OBJECT_DETECTION})
VECTOR_TASKS = frozenset({"single_label", "multilabel", "regression"})
SINGLE_LABEL_TASKS = frozenset({TASK_SINGLE_LABEL})
CNN_FAMILIES = frozenset({"resnet", "cnn", "clip_cnn"})
TRANSFORMER_FAMILIES = frozenset({"vit", "swin", "transformer", "clip_transformer"})


@dataclass(frozen=True)
class MethodSupport:
    families: FrozenSet[str]
    paper_scope: str
    implementation_scope: str
    tasks: FrozenSet[str] = ALL_TASKS
    task_scope: str = "Task-head agnostic implementation."


METHOD_SUPPORT: Dict[str, MethodSupport] = {
    "full": MethodSupport(ALL_FAMILIES, "Architecture-agnostic optimization baseline.", "Any supported classifier/regressor."),
    "linear": MethodSupport(ALL_FAMILIES, "Architecture-agnostic frozen-feature baseline.", "Any backbone with a replaceable task head."),
    "norm": MethodSupport(
        ALL_FAMILIES,
        "Normalization tuning trains normalization affine parameters and the downstream task head.",
        "Architecture-agnostic baseline for any supported task model.",
    ),
    "bias": MethodSupport(
        ALL_FAMILIES,
        "Bias-only tuning trains bias terms and the downstream task head.",
        "Architecture-agnostic bias baseline for any supported task model.",
    ),
    "last_block": MethodSupport(
        ALL_FAMILIES,
        "Partial fine-tuning trains the final backbone stage/block plus the downstream task head.",
        "Architecture-aware final-stage baseline with conservative structural discovery.",
    ),
    "prompt": MethodSupport(
        frozenset({"resnet"}),
        "Visual prompting learns an image-space border while the pretrained classifier remains frozen.",
        "Single-label classification with an unchanged pretrained output head and fixed label mapping.",
        tasks=frozenset({"single_label"}),
        task_scope="The source-to-target label mapping is defined only for single-label classification."
    ),
    "conv": MethodSupport(
        frozenset({"resnet"}),
        "Conv-Adapter is reproduced in ResNet-50 Bottleneck blocks.",
        "ResNet-50 only; four paper insertion schemes are available.",
        tasks=SINGLE_LABEL_TASKS,
        task_scope="The implemented paper route is single-label image classification on ResNet-50."
    ),
    "adapter": MethodSupport(
        frozenset({"resnet"}),
        "Alias of the strict Conv-Adapter reproduction.",
        "ResNet-50 only; do not report conv and adapter as separate rows.",
        tasks=SINGLE_LABEL_TASKS,
        task_scope="Alias of the single-label ResNet-50 Conv-Adapter route."
    ),
    "trso": MethodSupport(
        ALL_FAMILIES,
        "The proposal adapts generic matrix-shaped weights with cross-fitted evidence-selected two-sided tangent cores and a full task head.",
        "Architecture-agnostic Conv/Linear weights; normalization and embedding tensors are conservatively excluded.",
        tasks=NON_DETECTION_TASKS,
        task_scope="Classification, regression, semantic segmentation, and depth use tensor-output losses without changing the proposal core. Detection is excluded because torchvision detectors require target-aware model forwards during calibration."
    ),
    "ssf": MethodSupport(
        frozenset({"cnn", "vit", "swin"}),
        "SSF inserts affine scale/shift after published internal operations.",
        "torchvision ConvNeXt, VisionTransformer, and SwinTransformer only.",
        tasks=SINGLE_LABEL_TASKS,
        task_scope="The strict paper suite uses the original visual-recognition setting: single-label image classification."
    ),
    "lora": MethodSupport(
        frozenset({"vit", "swin", "transformer"}),
        "LoRA adapts Transformer query/value matrices.",
        "Recognized Transformer Q/V or packed QKV attention projections.",
        tasks=VECTOR_TASKS,
        task_scope="This is a transferred vision control, not an original-paper task reproduction."
    ),
    "bitfit": MethodSupport(
        TRANSFORMER_FAMILIES,
        "BitFit trains Transformer bias terms and the task classifier.",
        "Transformer biases plus task head.",
        tasks=VECTOR_TASKS,
        task_scope="This is a transferred vision control, not an original-paper task reproduction."
    ),
    "adaptformer": MethodSupport(
        frozenset({"vit"}),
        "AdaptFormer adds a parallel bottleneck branch beside each frozen ViT MLP.",
        "Recognized timm/torchvision Vision Transformer blocks; Swin is rejected rather than approximated.",
        tasks=SINGLE_LABEL_TASKS,
        task_scope="The strict paper suite uses the original visual-recognition setting: single-label image classification."
    ),
    "repadapter": MethodSupport(
        frozenset({"vit"}),
        "RepAdapter RepBlock inserts structurally re-parameterizable affine adapters before ViT attention and MLP projections.",
        "ViT-B/16 only; the trained adapter branches are exactly folded into QKV and first-MLP projections for deployment.",
        tasks=SINGLE_LABEL_TASKS,
        task_scope="The strict route reproduces the paper's ViT-B/16 single-label image-classification setting."
    ),
    "arc": MethodSupport(
        frozenset({"vit"}),
        "ARC re-composes layer-specific attention and FFN adapters from two cross-layer shared symmetric projection banks.",
        "Plain timm/torchvision Vision Transformers; the main ARC route inserts sequential adapters before MHA and FFN and supports exact evaluation-time folding.",
        tasks=SINGLE_LABEL_TASKS,
        task_scope="The strict route follows the ARC single-label visual-recognition formulation on plain ViTs."
    ),
    "piggyback": MethodSupport(
        frozenset({"resnet", "cnn"}),
        "Piggyback learns binary masks over frozen pretrained CNN weights.",
        "CNN Conv2d weights with a trainable downstream classifier; deployment storage is reported in bits.",
        tasks=SINGLE_LABEL_TASKS,
        task_scope="The strict paper suite uses the original visual-recognition setting: single-label image classification."
    ),
    "sidetune": MethodSupport(
        frozenset({"resnet"}),
        "Side-Tuning combines a frozen base with a copied trainable side network.",
        "ResNet only; side is initialized from the pretrained base.",
        tasks=CLASSIFICATION_TASKS,
        task_scope="Current side-network wrapper produces classification logits."
    ),
    "vpt_shallow": MethodSupport(
        frozenset({"vit"}),
        "VPT-Shallow inserts learnable prompt tokens once at the input of a frozen ViT.",
        "Plain timm/torchvision Vision Transformers only; image-border prompting remains a separate baseline.",
        tasks=SINGLE_LABEL_TASKS,
        task_scope="Faithful VPT reproduction is restricted to single-label recognition with a plain ViT."
    ),
    "vpt_deep": MethodSupport(
        frozenset({"vit"}),
        "VPT-Deep introduces a separate learnable prompt set at every frozen ViT block.",
        "Plain timm/torchvision Vision Transformers only.",
        tasks=SINGLE_LABEL_TASKS,
        task_scope="Faithful VPT reproduction is restricted to single-label recognition with a plain ViT."
    ),
    "convpass": MethodSupport(
        frozenset({"vit"}),
        "ConvPass adds convolutional bypasses in parallel to ViT attention and MLP branches.",
        "Plain timm/torchvision Vision Transformers with square patch grids.",
        tasks=SINGLE_LABEL_TASKS,
        task_scope="Published ConvPass route: single-label recognition with a plain ViT."
    ),
    "convpass_attn": MethodSupport(
        frozenset({"vit"}),
        "ConvPass-Attn adds the published convolutional bypass only beside attention.",
        "Plain timm/torchvision Vision Transformers with square patch grids.",
        tasks=SINGLE_LABEL_TASKS,
        task_scope="Published ConvPass attention-only ablation on single-label recognition."
    ),
    "fact_tt": MethodSupport(
        frozenset({"vit"}),
        "FacT-TT learns shared two-sided factors and one rank-by-rank core slice for each tensorized ViT operation.",
        "Plain timm/torchvision Vision Transformers with standard QKV/MLP projections.",
        tasks=SINGLE_LABEL_TASKS,
        task_scope="Published FacT single-label visual-recognition formulation."
    ),
    "fact_tk": MethodSupport(
        frozenset({"vit"}),
        "FacT-TK generates operation cores from a shared third-order core and operation factors.",
        "Plain timm/torchvision Vision Transformers with standard QKV/MLP projections.",
        tasks=SINGLE_LABEL_TASKS,
        task_scope="Published FacT single-label visual-recognition formulation."
    ),
    "vqt": MethodSupport(
        frozenset({"vit"}),
        "VQT learns per-layer query tokens that summarize intact intermediate ViT representations.",
        "Plain timm/torchvision Vision Transformers; frozen backbone and concatenated query features.",
        tasks=SINGLE_LABEL_TASKS,
        task_scope="Published VQT single-label visual-recognition formulation."
    ),
    "spt_lora": MethodSupport(
        frozenset({"vit"}),
        "SPT uses accumulated squared gradients for one-shot sensitivity and combines sparse tuning with LoRA.",
        "Plain timm/torchvision Vision Transformers; explicit paper budget/rank/sample hyperparameters; the fair runner uses 800 sensitivity samples as in the main experiments.",
        tasks=SINGLE_LABEL_TASKS,
        task_scope="Published SPT-LoRA single-label visual-recognition formulation."
    ),
    "spt_adapter": MethodSupport(
        frozenset({"vit"}),
        "SPT uses accumulated squared gradients for one-shot sensitivity and combines sparse tuning with sequential Adapters.",
        "Plain timm/torchvision Vision Transformers; explicit paper budget/dimension/sample hyperparameters; the fair runner uses 800 sensitivity samples as in the main experiments.",
        tasks=SINGLE_LABEL_TASKS,
        task_scope="Published SPT-Adapter single-label visual-recognition formulation."
    ),
    "ml_decoder": MethodSupport(
        frozenset({"cnn"}),
        "ML-Decoder replaces global-pooling classification heads with fixed queries, cross-attention, and grouped classifiers.",
        "Revision baseline is the published ML-Decoder head on MobileNetV3-Small spatial features.",
        tasks=frozenset({"multilabel"}),
        task_scope="Task-specific published multi-label classification baseline; the revision route is VOC2007/MobileNetV3-Small."
    ),
    "segadapter": MethodSupport(
        frozenset({"cnn"}),
        "SegAdapter augments segmentation backbone stages with HSA, FFN, scaled residual injection, and an auxiliary coarse segmentation loss.",
        "Revision reimplementation targets torchvision LR-ASPP with MobileNetV3-Large using the published SegAdapter equations/defaults.",
        tasks=frozenset({TASK_SEMANTIC_SEGMENTATION}),
        task_scope="Published semantic-segmentation adapter baseline."
    ),
}


# Publication taxonomy. Only methods in PAPER_BASELINE_METHODS are allowed to
# appear as literature-baseline rows. Reference controls, the TRSO proposal,
# paper-internal ablations, transferred controls, engineering controls, and
# implementations that still differ from the official release are separate.
REFERENCE_CONTROL_METHODS = frozenset({"full", "linear"})
ENGINEERING_CONTROL_METHODS = frozenset({"norm", "bias", "last_block"})
PROPOSAL_METHODS = frozenset({"trso"})
PAPER_BASELINE_METHODS = frozenset({
    "prompt", "conv", "ssf", "adaptformer", "repadapter", "arc", "piggyback",
    "vpt_shallow", "vpt_deep", "convpass",
    # Paper-derived implementations validated against the authors' public releases.
    "fact_tt", "fact_tk", "vqt", "spt_lora", "spt_adapter",
    # Task-specific published baselines added for the revision.
    "ml_decoder", "segadapter",
})
PAPER_ABLATION_METHODS = frozenset({"convpass_attn"})
PAPER_REIMPLEMENTATION_CANDIDATES = frozenset()
TRANSFERRED_CONTROL_METHODS = frozenset({"lora", "bitfit", "sidetune"})

# ``--methods auto`` is deliberately baseline-only. Comparison, reference,
# proposal and ablation suites are selected explicitly by the runners.
STRICT_AUTO_METHODS = (
    "prompt", "conv", "ssf", "adaptformer", "repadapter", "arc", "piggyback",
    "vpt_shallow", "vpt_deep", "convpass", "fact_tt", "fact_tk", "vqt",
    "spt_lora", "spt_adapter", "ml_decoder", "segadapter",
)
BENCHMARK_COMPARISON_METHODS = (
    "full", "linear", "trso", *STRICT_AUTO_METHODS,
)

ORIGINAL_PAPER_REFERENCES = {
    "prompt": "Bahng et al., Exploring Visual Prompts for Adapting Large-Scale Models (2022)",
    "conv": "Chen et al., Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets (2022/2024)",
    "ssf": "Lian et al., Scaling & Shifting Your Features (NeurIPS 2022)",
    "adaptformer": "Chen et al., AdaptFormer (NeurIPS 2022)",
    "repadapter": "Luo et al., Towards Efficient Visual Adaption via Structural Re-parameterization (2023)",
    "arc": "Dong et al., Efficient Adaptation of Large Vision Transformer via Adapter Re-Composing (NeurIPS 2023)",
    "piggyback": "Mallya et al., Piggyback (ECCV 2018)",
    "vpt_shallow": "Jia et al., Visual Prompt Tuning (ECCV 2022)",
    "vpt_deep": "Jia et al., Visual Prompt Tuning (ECCV 2022)",
    "convpass": "Jie and Deng, Convolutional Bypasses Are Better Vision Transformer Adapters (2022)",
    "convpass_attn": "Jie and Deng, ConvPass attention-only paper ablation (2022)",
    "fact_tt": "Jie and Deng, FacT: Factor-Tuning for Lightweight Adaptation on Vision Transformer (AAAI 2023)",
    "fact_tk": "Jie and Deng, FacT: Factor-Tuning for Lightweight Adaptation on Vision Transformer (AAAI 2023)",
    "vqt": "Tu et al., Visual Query Tuning (CVPR 2023)",
    "spt_lora": "He et al., Sensitivity-Aware Visual Parameter-Efficient Fine-Tuning (ICCV 2023)",
    "spt_adapter": "He et al., Sensitivity-Aware Visual Parameter-Efficient Fine-Tuning (ICCV 2023)",
    "lora": "Hu et al., LoRA (ICLR 2022; original NLP/LLM task)",
    "bitfit": "Ben-Zaken et al., BitFit (ACL 2022; original NLP task)",
    "sidetune": "Zhang et al., Side-Tuning (ECCV 2020; current wrapper is not an exact reproduction)",
    "ml_decoder": "Ridnik et al., ML-Decoder: Scalable and Versatile Classification Head (WACV 2023)",
    "segadapter": "Peng and Kameyama, Simple and Efficient Vision Backbone Adapter for Image Semantic Segmentation (ACML 2023)",
}


def method_category(method: str) -> str:
    method = canonical_method(method)
    if method == "adapter":
        method = "conv"
    if method in REFERENCE_CONTROL_METHODS:
        return "reference_control"
    if method in ENGINEERING_CONTROL_METHODS:
        return "engineering_control"
    if method in PROPOSAL_METHODS:
        return "proposal"
    if method in PAPER_BASELINE_METHODS:
        return "paper_baseline"
    if method in PAPER_ABLATION_METHODS:
        return "paper_ablation"
    if method in PAPER_REIMPLEMENTATION_CANDIDATES:
        return "paper_reimplementation_candidate"
    if method in TRANSFERRED_CONTROL_METHODS:
        return "transferred_control"
    return "unknown"


def validate_method_status(
    method: str,
    *,
    allow_nonpaper_controls: bool = False,
    allow_paper_ablations: bool = False,
    allow_unverified_paper_reimplementations: bool = False,
) -> None:
    """Prevent a method from being silently mislabeled as a paper baseline."""
    method = canonical_method(method)
    category = method_category(method)
    reference = ORIGINAL_PAPER_REFERENCES.get(method, "no original paper baseline")
    if category in {"engineering_control", "transferred_control"} and not allow_nonpaper_controls:
        raise ValueError(
            f"Method '{method}' is classified as {category}, not as an original-paper "
            f"vision baseline in this repository ({reference}). Use it only in a separately "
            "labeled control experiment with --allow_nonpaper_controls True."
        )
    if category == "paper_ablation" and not allow_paper_ablations:
        raise ValueError(
            f"Method '{method}' is an ablation inside another baseline paper, not a standalone "
            f"literature baseline ({reference}). Use --allow_paper_ablations True only in a "
            "separate paper-ablation study."
        )
    if category == "paper_reimplementation_candidate" and not allow_unverified_paper_reimplementations:
        raise ValueError(
            f"Method '{method}' is paper-derived but the local implementation is not certified "
            f"as an exact official-paper reproduction ({reference}). It is excluded from the "
            "strict baseline table. Use --allow_unverified_paper_reimplementations True only "
            "for a separately labeled reproduction study."
        )



def canonical_method(name: str) -> str:
    value = str(name or "").strip().lower().replace("-", "_")
    aliases = {
        "task_response": "trso",
        "task_response_adapter": "trso",
        "trso_adapter": "trso",
        "conv_adapter": "conv",
        "conv_adapt": "conv",
        "ssf_adapter": "ssf",
        "lora_conv2d": "lora_conv",
        "side_tuning": "sidetune",
        "sidetuning": "sidetune",
        "side_tune": "sidetune",
        "adapt_former": "adaptformer",
        "piggy_back": "piggyback",
        "linear_probe": "linear",
        "head_only": "linear",
        "norm_tuning": "norm",
        "bn_tuning": "norm",
        "ln_tuning": "norm",
        "bias_only": "bias",
        "partial": "last_block",
        "last_stage": "last_block",
        "finetune": "full",
        "vpt": "vpt_deep",
        "vpt_shallow_prompt": "vpt_shallow",
        "vpt_deep_prompt": "vpt_deep",
        "conv_pass": "convpass",
        "adapter_re_composing": "arc",
        "arc_adapter": "arc",
        "fact": "fact_tk",
        "fact_tt": "fact_tt",
        "fact_tk": "fact_tk",
        "visual_query_tuning": "vqt",
        "spt": "spt_lora",
        "spt_lora": "spt_lora",
        "spt_adapter": "spt_adapter",
        "mldecoder": "ml_decoder",
        "ml_decoder": "ml_decoder",
        "seg_adapter": "segadapter",
        "segadapter": "segadapter",
    }
    return aliases.get(value, value)


def detect_backbone_family(model: nn.Module, backbone_name: str = "", source: str = "") -> str:
    """Infer a conservative representation family from type/name metadata."""
    name = str(backbone_name or "").lower()
    cls_name = model.__class__.__name__.lower()
    module_name = model.__class__.__module__.lower()
    text = " ".join((name, cls_name, module_name, str(source).lower()))

    if "clip" in text:
        if any(token in text for token in ("vit", "visiontransformer")):
            return "clip_transformer"
        return "clip_cnn"
    if any(token in text for token in ("swin", "shiftedwindow")):
        return "swin"
    if any(token in text for token in ("visiontransformer", "vision_transformer", "vit_", "deit", "beit", "eva", "cait")):
        return "vit"
    if any(token in text for token in ("transformer", "maxvit")):
        return "transformer"
    if "resnet" in text or all(hasattr(model, attr) for attr in ("layer1", "layer2", "layer3", "layer4")):
        return "resnet"
    if any(token in text for token in (
        "convnext", "efficientnet", "mobilenet", "densenet", "regnet", "vgg", "alexnet",
        "mnasnet", "shufflenet", "squeezenet", "inception", "googlenet", "nasnet",
    )):
        return "cnn"
    if any(token in text for token in ("mlpmixer", "mixer", "resmlp", "gmlp")):
        return "mlp"

    # Structural fallback: presence of spatial convolutions is not enough to
    # classify hybrid Transformers, so use it only after name/type checks.
    if any(isinstance(module, nn.Conv2d) for module in model.modules()):
        return "cnn"
    return "transformer" if any(isinstance(module, nn.MultiheadAttention) for module in model.modules()) else "unknown"


def validate_method_backbone(method: str, family: str) -> None:
    method = canonical_method(method)
    if method not in METHOD_SUPPORT:
        raise ValueError(f"Unknown tuning method '{method}'.")
    support = METHOD_SUPPORT[method]
    if family not in support.families:
        allowed = ", ".join(sorted(support.families))
        raise ValueError(
            f"Method '{method}' is not supported on backbone family '{family}'. "
            f"Allowed families: {allowed}. Paper scope: {support.paper_scope} "
            f"Implementation scope: {support.implementation_scope}"
        )




def validate_method_task(method: str, task: str, *, allow_auto: bool = True) -> None:
    method = canonical_method(method)
    task = normalize_task(task)
    if allow_auto and task == "auto":
        return
    if method not in METHOD_SUPPORT:
        raise ValueError(f"Unknown tuning method '{method}'.")
    support = METHOD_SUPPORT[method]
    if task not in support.tasks:
        allowed = ", ".join(sorted(support.tasks))
        raise ValueError(
            f"Method '{method}' is not supported for task '{task}'. "
            f"Allowed tasks: {allowed}. Task scope: {support.task_scope}"
        )


def method_compatibility(method: str, family: str, task: str) -> tuple[bool, str]:
    """Return a non-throwing capability decision for experiment planners."""
    try:
        validate_method_backbone(method, family)
        validate_method_task(method, task)
    except ValueError as exc:
        return False, str(exc)
    return True, "supported"


def infer_backbone_family_name(backbone_name: str, source: str = "") -> str:
    """Conservative family inference without constructing/downloading a model."""
    name = str(backbone_name or "").lower()
    text = f"{name} {str(source or '').lower()}"
    if "clip" in text:
        return "clip_transformer" if any(token in text for token in ("vit", "transformer")) else "clip_cnn"
    if "swin" in text:
        return "swin"
    if any(token in text for token in ("vit", "deit", "beit", "eva", "cait", "vision_transformer")):
        return "vit"
    if any(token in text for token in ("maxvit", "transformer")):
        return "transformer"
    if "resnet" in text or "resnext" in text or "wide_resnet" in text:
        return "resnet"
    if any(token in text for token in (
        "convnext", "efficientnet", "mobilenet", "densenet", "regnet", "vgg",
        "alexnet", "mnasnet", "shufflenet", "squeezenet", "inception",
        "googlenet", "nasnet", "xception", "rexnet", "coatnet",
    )):
        return "cnn"
    if any(token in text for token in ("mixer", "resmlp", "gmlp")):
        return "mlp"
    return "unknown"


def static_method_compatibility(
    method: str, backbone_name: str, task: str, *, source: str = "auto",
    allow_nonpaper_controls: bool = False,
    allow_paper_ablations: bool = False,
    allow_unverified_paper_reimplementations: bool = False,
) -> tuple[bool, str, str]:
    """Plan-time compatibility including paper-specific backbone contracts."""
    method = canonical_method(method)
    family = infer_backbone_family_name(backbone_name, source)
    try:
        validate_method_status(
            method,
            allow_nonpaper_controls=allow_nonpaper_controls,
            allow_paper_ablations=allow_paper_ablations,
            allow_unverified_paper_reimplementations=allow_unverified_paper_reimplementations,
        )
    except ValueError as exc:
        return False, str(exc), family
    ok, reason = method_compatibility(method, family, task)
    if not ok:
        return False, reason, family
    normalized = str(backbone_name or "").lower().replace("-", "_")
    if method in {"conv", "adapter"} and "resnet50" not in normalized:
        return False, f"{method} strict reproduction requires ResNet-50, got {backbone_name!r}.", family
    if method == "adaptformer" and family != "vit":
        return False, "AdaptFormer requires a recognized plain ViT backbone.", family

    # Exact original-paper architecture contracts. Broad family matches are not
    # sufficient because they can silently turn a reproduction into a transfer.
    plain_vit_methods = {
        "adaptformer", "arc", "vpt_shallow", "vpt_deep", "convpass", "convpass_attn",
        "fact_tt", "fact_tk", "vqt", "spt_lora", "spt_adapter",
    }
    if method in plain_vit_methods:
        disallowed_hybrid = any(token in normalized for token in ("swin", "deit", "beit", "eva", "cait", "maxvit"))
        if family != "vit" or "vit" not in normalized or disallowed_hybrid:
            return False, f"{method} strict reproduction requires a plain Vision Transformer (ViT), got {backbone_name!r}.", family
    if method == "ml_decoder" and "mobilenet_v3_small" not in normalized:
        return False, f"ML-Decoder revision route requires MobileNetV3-Small, got {backbone_name!r}.", family
    if method == "segadapter" and "lraspp_mobilenet_v3_large" not in normalized:
        return False, f"SegAdapter revision route requires LR-ASPP/MobileNetV3-Large, got {backbone_name!r}.", family
    if method == "repadapter":
        allowed = ("vit_b_16", "vit_base_patch16_224", "vit_base_patch16_224_in21k")
        if family != "vit" or not any(token in normalized for token in allowed):
            return False, (
                "RepAdapter strict reproduction is limited to the paper's ViT-B/16 image-classification route; "
                f"got {backbone_name!r}."
            ), family
    if method == "prompt":
        allowed_prompt_models = ("resnet18", "resnet50", "resnet101", "resnet152", "resnext101_32x8d")
        if not any(token in normalized for token in allowed_prompt_models):
            return False, (
                "The implemented visual-prompting paper route keeps the original classifier "
                "and is restricted to ResNet-18/50/101/152 or ResNeXt-101-32x8d. "
                f"Got {backbone_name!r}."
            ), family
    if method == "ssf":
        if not (family in {"vit", "swin"} or "convnext" in normalized):
            return False, "SSF strict reproduction supports ViT, Swin, or ConvNeXt only.", family
    if method == "piggyback":
        if not any(token in normalized for token in ("resnet50", "vgg16")):
            return False, "Piggyback strict reproduction supports the paper CNN routes VGG-16 and ResNet-50 only.", family
    return True, "supported", family

def compatibility_rows():
    for method, support in METHOD_SUPPORT.items():
        yield {
            "method": method,
            "families": sorted(support.families),
            "paper_scope": support.paper_scope,
            "implementation_scope": support.implementation_scope,
            "tasks": sorted(support.tasks),
            "task_scope": support.task_scope,
            "category": method_category(method),
            "strict_default": method in STRICT_AUTO_METHODS,
            "original_paper": ORIGINAL_PAPER_REFERENCES.get("conv" if method == "adapter" else method, ""),
        }


__all__ = [
    "ALL_FAMILIES",
    "ALL_TASKS",
    "CLASSIFICATION_TASKS",
    "CNN_FAMILIES",
    "TRANSFORMER_FAMILIES",
    "METHOD_SUPPORT",
    "REFERENCE_CONTROL_METHODS",
    "ENGINEERING_CONTROL_METHODS",
    "PROPOSAL_METHODS",
    "PAPER_BASELINE_METHODS",
    "PAPER_ABLATION_METHODS",
    "PAPER_REIMPLEMENTATION_CANDIDATES",
    "TRANSFERRED_CONTROL_METHODS",
    "STRICT_AUTO_METHODS",
    "BENCHMARK_COMPARISON_METHODS",
    "ORIGINAL_PAPER_REFERENCES",
    "canonical_method",
    "method_category",
    "validate_method_status",
    "detect_backbone_family",
    "infer_backbone_family_name",
    "static_method_compatibility",
    "validate_method_backbone",
    "validate_method_task",
    "method_compatibility",
    "normalize_task",
    "compatibility_rows",
]
