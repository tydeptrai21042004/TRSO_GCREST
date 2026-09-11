# main.py — TRSO parameter-efficient fine-tuning entry point
"""
Revision-focused training script.

Main fixes for reviewer/editor comments:
1. Adds full fine-tuning and linear-probe baselines.
2. Adds Task-Response Spatial Operator (TRSO) calibration and layer selection.
3. Uses train/val/test semantics: best checkpoint is selected on validation, then
   evaluated once on test.
4. Adds profiling hooks for trainable params, total params, FLOPs, latency, and memory.
5. Logs per-epoch history and convergence summaries for mean/std aggregation.
6. Fixes side-tuning constructor and safer adapter trainability rules.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import time
import random
import sys
import platform
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

# Keep repository runs bytecode-cache free. Release packaging also removes any
# third-party/tool caches that may still be created.
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
try:
    from timm.data.mixup import Mixup
    from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
    from timm.utils import ModelEma
except Exception:
    from compat.timm_compat import Mixup, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, ModelEma

from datasets import build_dataset, current_preprocessing_profile, get_dataset_spec, parse_download_mode
try:
    from datasets.build import available_datasets, build_dataset_split
except Exception:  # backward fallback
    build_dataset_split = None
    available_datasets = lambda: []
from task_registry import (
    TASK_DEPTH_ESTIMATION, TASK_MULTILABEL, TASK_OBJECT_DETECTION,
    TASK_REGRESSION, TASK_SEMANTIC_SEGMENTATION, TASK_SINGLE_LABEL,
    normalize_task, task_choices, task_spec,
)

from engine import evaluate, train_one_epoch
from memory_utils import profile_memory_cost
from models import build_model
from models.model_support import (
    ORIGINAL_PAPER_REFERENCES, canonical_method, compatibility_rows,
    detect_backbone_family, method_category, static_method_compatibility,
    validate_method_backbone, validate_method_status, validate_method_task,
)
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils


def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.lower()
        if s in ("yes", "true", "t", "y", "1"):
            return True
        if s in ("no", "false", "f", "n", "0"):
            return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args_parser():
    parser = argparse.ArgumentParser("Parameter Efficient Tuning", add_help=False)

    # Backbone
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--model_source", type=str, default="auto", choices=["auto", "torchvision", "timm", "hub", "smp"])
    parser.add_argument("--weights", type=str, default="DEFAULT")
    parser.add_argument("--list_backbones", action="store_true")
    parser.add_argument("--list_compatibility", action="store_true")
    parser.add_argument("--experiment_suite", type=str, default="", help="Manifest suite metadata.")
    parser.add_argument("--experiment_name", type=str, default="", help="Ablation or sweep variant metadata.")
    parser.add_argument("--experiment_run_id", type=str, default="", help="Deterministic manifest run identifier.")
    parser.add_argument("--legacy_auto_hparams", type=str2bool, default=False, help="Opt into the original repository hyperparameter override table.")
    parser.add_argument("--pretrained", type=str2bool, default=None)
    parser.add_argument("--keep_pretrained_head", type=str2bool, default=True)
    parser.add_argument("--cifar_hub", type=str, default="auto", choices=["auto", "chenyaofo", "akamaster"])

    # CLIP linear probe branch
    parser.add_argument("--clip_model", type=str, default=None, help="OpenAI CLIP/OpenCLIP visual backbone name, e.g. RN50 or ViT-B/16.")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--freeze_backbone", type=str2bool, default=True)

    # Methods
    parser.add_argument(
        "--tuning_method",
        type=str,
        default="trso",
        help=(
            "full | linear | norm | bias | last_block | prompt | conv | adapter | trso | "
            "ssf | lora | bitfit | adaptformer | repadapter | arc | piggyback | sidetune | "
            "vpt_shallow | vpt_deep | convpass | convpass_attn | fact_tt | fact_tk | vqt | spt_lora | spt_adapter"
        ),
    )

    parser.add_argument(
        "--allow_nonpaper_controls", type=str2bool, default=False,
        help=(
            "Opt in to engineering or cross-domain controls (norm/bias/last_block, "
            "vision-transferred LoRA/BitFit, and the repository Side-Tuning approximation). "
            "They are excluded from strict original-paper comparisons by default."
        ),
    )
    parser.add_argument(
        "--allow_paper_ablations", type=str2bool, default=False,
        help="Opt in to paper-internal ablations such as ConvPass-Attn; never label them as standalone baselines.",
    )
    parser.add_argument(
        "--allow_unverified_paper_reimplementations", type=str2bool, default=False,
        help="Opt in to qualified FacT/VQT/SPT reproduction candidates outside the strict baseline table.",
    )

    # Side-tuning
    parser.add_argument("--sidetune_alpha", type=float, default=0.5)
    parser.add_argument("--sidetune_learn_alpha", type=str2bool, default=True)
    parser.add_argument("--sidetune_width", type=int, default=64)
    parser.add_argument("--sidetune_depth", type=int, default=4)
    parser.add_argument("--sidetune_arch", type=str, default="lightweight", choices=["lightweight", "copy"])
    parser.add_argument("--sidetune_checkpoint", type=str, default="", help="Optional distilled/lightweight side-network checkpoint.")

    # Prompt / Conv-Adapter
    parser.add_argument("--prompt_size", default=30, type=int)
    parser.add_argument("--prompt_type", default="padding", choices=["padding", "fixed_patch", "random_patch"])
    parser.add_argument("--prompt_output_indices", default="", type=str, help="Comma-separated fixed pretrained output indices. Explicit indices override --prompt_mapping.")
    parser.add_argument("--prompt_mapping", default="frequency", choices=["identity", "frequency"], help="Training-derived frequency mapping is the strict paper default; identity is an explicit ablation.")
    parser.add_argument("--prompt_mapping_batches", default=0, type=int, help="Number of train batches for frequency mapping; 0 uses the full train loader.")
    parser.add_argument("--kernel_size", default=3, type=int)
    parser.add_argument("--adapt_size", default=8, type=float)
    parser.add_argument("--adapt_scale", default=1.0, type=float)
    parser.add_argument("--conv_adapter_mode", default="conv_parallel", choices=["conv_parallel", "conv_sequential", "residual_parallel", "residual_sequential"])

    # Cross-Fitted Variance-Calibrated Evidence-Spectral Tangent Core: no method-specific rank,
    # budget, layer list, threshold, head policy, core mode, calibration-batch
    # count, gain, or adapter-capacity argument.
    parser.add_argument(
        "--trso_fast_inference", type=str2bool, default=True,
        help="Exactly merge the learned G-CREST tangent-core updates before evaluation.",
    )
    parser.add_argument(
        "--trso_ablation", type=str, default="full",
        choices=["full", "diagonal_only", "no_sampling_variance", "no_crossfit", "head_only"],
        help="Fixed structural ablation for analysis only; full is the proposal used in benchmark comparisons.",
    )
    # Reviewer-requested sensitivity controls. Defaults exactly reproduce the
    # proposed method; non-default values are ablations and must be reported as such.
    parser.add_argument("--trso_mode_count_rule", default="geometric",
                        choices=["geometric", "shannon", "arithmetic", "harmonic"],
                        help="Global R rule; geometric is the paper proposal, others are reviewer ablations.")
    parser.add_argument("--trso_r_scale", type=float, default=1.0,
                        help="Scale the automatic R before top-R selection; 1.0 is the proposal.")
    parser.add_argument("--trso_fixed_r", type=int, default=0,
                        help="If >0, override automatic R for sensitivity analysis only.")
    parser.add_argument("--trso_calibration_fraction", type=float, default=1.0,
                        help="Fraction of calibration batches to use (reviewer sensitivity study).")
    parser.add_argument("--trso_calibration_max_batches", type=int, default=0,
                        help="Optional hard cap on calibration batches; 0 means no extra cap.")
    parser.add_argument("--trso_partition_mode", default="alternating",
                        choices=["alternating", "seeded_random"],
                        help="Deterministic calibration partition construction.")
    parser.add_argument("--trso_partition_seed", type=int, default=0,
                        help="Seed used only by seeded_random calibration partitions.")
    parser.add_argument("--trso_svd_oversampling", type=int, default=0,
                        help="Randomized-SVD oversampling; 0 keeps the deterministic automatic rule.")
    parser.add_argument("--trso_svd_power_iterations", type=int, default=2,
                        help="Randomized-SVD power iterations (reported for reproducibility).")
    parser.add_argument("--trso_svd_seed", type=int, default=0,
                        help="Base seed for deterministic randomized-SVD sketches.")


    # SSF / LoRA / BitFit
    parser.add_argument("--ssf_init_scale", type=float, default=1.0)
    parser.add_argument("--ssf_init_shift", type=float, default=0.0)
    parser.add_argument("--ssf_init_std", type=float, default=0.02)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_merge_weights", type=str2bool, default=True)
    parser.add_argument("--lora_target", type=str, default="all", choices=["all", "1x1", "3x3", "dw"])
    parser.add_argument("--bitfit_train_head", type=str2bool, default=True)
    parser.add_argument("--bitfit_bias_scope", type=str, default="all", choices=["all", "transformer", "attention"])

    # AdaptFormer (NeurIPS 2022)
    parser.add_argument("--adaptformer_dim", type=int, default=16)
    parser.add_argument("--adaptformer_scale", type=float, default=0.1)
    parser.add_argument("--adaptformer_dropout", type=float, default=0.0)
    parser.add_argument("--adaptformer_layernorm", type=str, default="none", choices=["none", "in", "out"])

    # RepAdapter RepBlock (ViT-B/16 classification route)
    parser.add_argument("--repadapter_dim", type=int, default=8)
    parser.add_argument("--repadapter_groups", type=int, default=2)
    parser.add_argument("--repadapter_scale", type=float, default=1.0)
    parser.add_argument("--repadapter_dropout", type=float, default=0.1)
    parser.add_argument("--repadapter_merge", type=str2bool, default=True)

    # ARC main configuration (NeurIPS 2023 official ViT defaults)
    parser.add_argument("--arc_dim", type=int, default=50)
    parser.add_argument("--arc_dropout", type=float, default=0.1)
    parser.add_argument("--arc_merge", type=str2bool, default=True)

    # Transformer paper baselines
    parser.add_argument("--vpt_num_tokens", type=int, default=10)
    parser.add_argument("--vpt_dropout", type=float, default=0.0)
    parser.add_argument("--convpass_dim", type=int, default=8)
    parser.add_argument("--convpass_scale", type=float, default=1.0)
    parser.add_argument("--convpass_dropout", type=float, default=0.1)
    parser.add_argument(
        "--fact_rank", type=int, default=0,
        help="FacT rank; 0 uses the paper canonical setting (TT=4, TK=8).",
    )
    parser.add_argument("--fact_scale", type=float, default=1.0)
    parser.add_argument("--vqt_query_length", type=int, default=1)
    parser.add_argument("--spt_budget", type=int, default=400000, help="Desired SPT backbone parameter budget; paper sweeps 0.2M--1.0M.")
    parser.add_argument(
        "--spt_sensitivity_samples", type=int, default=800,
        help="SPT sensitivity samples; the paper main experiments use 800 (400 is an ablation setting).",
    )
    parser.add_argument("--spt_rank", type=int, default=8)
    parser.add_argument("--spt_adapter_dim", type=int, default=8)
    parser.add_argument("--spt_alpha", type=float, default=8.0)

    # Piggyback (ECCV 2018)
    parser.add_argument("--piggyback_threshold", type=float, default=5e-3)
    parser.add_argument("--piggyback_mask_init", type=str, default="ones", choices=["ones", "near_threshold"])
    parser.add_argument("--piggyback_mask_scale", type=float, default=1e-2)
    parser.add_argument("--piggyback_mask_linear", type=str2bool, default=False)
    parser.add_argument("--piggyback_train_head", type=str2bool, default=True)

    # Batch / epochs
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--update_freq", default=1, type=int)
    parser.add_argument("--fs_shot", default=16, type=int)

    # Generic model params
    parser.add_argument("--model", default="resnet50_clip", type=str)
    parser.add_argument("--drop_path", type=float, default=0)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--crop_ratio", default=0.875, type=float)

    # EMA
    parser.add_argument("--model_ema", type=str2bool, default=False)
    parser.add_argument("--model_ema_decay", type=float, default=0.9999)
    parser.add_argument("--model_ema_force_cpu", type=str2bool, default=False)
    parser.add_argument("--model_ema_eval", type=str2bool, default=False)

    # Optimization
    parser.add_argument("--optimizer", default="auto", choices=["auto", "adamw", "sgd"])
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--paper_hparams", type=str2bool, default=False, help="Apply the original paper default optimizer schedule where a single canonical setting exists.")
    parser.add_argument("--opt_eps", default=1e-8, type=float)
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--weight_decay_adapter", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--weight_decay_end", type=float, default=None)

    # Controlled fairness protocol. All PEFT methods share one optimizer, LR,
    # scheduler, warm-up and decay; full tuning and linear probing may use
    # separate learning rates as explicitly allowed by the experiment design.
    parser.add_argument("--fair_protocol", type=str2bool, default=False)
    parser.add_argument("--fair_optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--fair_peft_lr", type=float, default=1e-3)
    parser.add_argument("--fair_full_lr", type=float, default=1e-4)
    parser.add_argument("--fair_linear_lr", type=float, default=1e-3)
    parser.add_argument("--fair_weight_decay", type=float, default=1e-4)
    parser.add_argument("--fair_warmup_epochs", type=int, default=5)
    parser.add_argument("--fair_min_lr", type=float, default=1e-6)

    # Augmentation / preprocessing
    parser.add_argument("--color_jitter", type=float, default=0.0)
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--regression_loss", type=str, default="mse", choices=["mse", "smooth_l1"])
    parser.add_argument("--train_interpolation", type=str, default="bicubic")
    parser.add_argument("--crop_pct", type=float, default=None)
    parser.add_argument("--reprob", type=float, default=0.0)
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)
    parser.add_argument("--resplit", type=str2bool, default=False)
    parser.add_argument("--imagenet_norm", type=str2bool, default=True)
    parser.add_argument("--train_aug", type=str, default="standard", choices=["standard", "resize"])

    # Mixup/Cutmix
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--cutmix_minmax", type=float, nargs="+", default=None)
    parser.add_argument("--mixup_prob", type=float, default=1.0)
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5)
    parser.add_argument("--mixup_mode", type=str, default="batch")

    # Finetuning/checkpoint params
    parser.add_argument("--finetune", default="")
    parser.add_argument("--head_from", default="", type=str)
    parser.add_argument("--head_init_scale", default=1.0, type=float)
    parser.add_argument("--peft_freeze_head", type=str2bool, default=False, help="Freeze the shared task-aware head during PEFT adaptation; jointly adapting it is the recommended default.")
    parser.add_argument("--peft_head_lr_scale", type=float, default=0.5, help="Head learning-rate multiplier for PEFT methods; linear/full tuning use 1.0.")
    parser.add_argument("--no_decay_bias_norm", type=str2bool, default=True, help="Exclude biases, normalization parameters, and scalar gates from weight decay.")
    parser.add_argument("--model_key", default="model|module", type=str)
    parser.add_argument("--model_prefix", default="", type=str)

    # Dataset
    parser.add_argument("--is_tuning", default=False, type=str2bool)
    parser.add_argument("--dataset", default="dtd", type=str)
    parser.add_argument("--task", default="auto", choices=task_choices(include_auto=True))
    parser.add_argument("--download", type=parse_download_mode, default="no", help="Dataset download policy: auto, yes/true, or no/false. Legacy booleans remain accepted.")
    parser.add_argument("--preprocess", default="auto", choices=["auto", "repository"], help="Resolve normalization/interpolation from pretrained metadata when available.")
    parser.add_argument("--allow_val_as_test", type=str2bool, default=False)
    parser.add_argument("--val_ratio", default=0.1, type=float)
    parser.add_argument("--dtd_partition", default=1, type=int)
    parser.add_argument("--places_small", type=str2bool, default=True)
    parser.add_argument("--inat_target_type", default="full", type=str)
    parser.add_argument("--coco_task", default="multilabel", choices=["multilabel", "majority"])
    parser.add_argument("--celeba_task", default="attributes", choices=["attributes", "landmarks"], help="CelebA attributes is multi-label; landmarks is 10-output regression.")
    parser.add_argument("--segmentation_num_classes", default=2, type=int)
    parser.add_argument("--segmentation_ignore_index", default=255, type=int)
    parser.add_argument("--dense_train_aug", default="scale_crop_flip", choices=["scale_crop_flip", "resize"])
    parser.add_argument("--dense_hflip_prob", default=0.5, type=float)
    parser.add_argument("--depth_scale", default=1.0, type=float, help="Divide stored depth values by this scale.")
    parser.add_argument("--depth_min", default=1e-3, type=float)
    parser.add_argument("--depth_max", default=80.0, type=float)
    parser.add_argument("--depth_loss", default="silog_l1", choices=["l1", "smooth_l1", "silog_l1"])
    parser.add_argument("--detection_num_classes", default=2, type=int, help="Includes background class.")
    parser.add_argument("--voc_year", default="2007", choices=["2007", "2012"])
    parser.add_argument("--train_csv", default="", type=str)
    parser.add_argument("--val_csv", default="", type=str)
    parser.add_argument("--test_csv", default="", type=str)
    parser.add_argument("--image_column", default="image", type=str)
    parser.add_argument("--label_column", default="label", type=str)
    parser.add_argument("--label_separator", default=";", type=str)
    parser.add_argument("--data_path", default="./data", type=str)
    parser.add_argument("--eval_data_path", default=None, type=str)
    parser.add_argument("--nb_classes", default=1000, type=int)
    parser.add_argument("--fake_train_size", default=32, type=int)
    parser.add_argument("--fake_val_size", default=16, type=int)
    parser.add_argument("--fake_test_size", default=16, type=int)
    parser.add_argument("--imagenet_default_mean_and_std", type=str2bool, default=True)
    parser.add_argument("--output_dir", default="./experiments/")
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--split_seed", default=42, type=int, help="Dataset split seed independent of optimization seed.")
    parser.add_argument("--deterministic", type=str2bool, default=False, help="Use deterministic PyTorch algorithms when available.")

    parser.add_argument("--resume", default="")
    parser.add_argument("--auto_resume", type=str2bool, default=False)
    parser.add_argument("--save_ckpt", type=str2bool, default=True)
    parser.add_argument("--save_ckpt_freq", default=1, type=int)
    parser.add_argument("--save_ckpt_num", default=2, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", type=str2bool, default=False)
    parser.add_argument("--dist_eval", type=str2bool, default=True)
    parser.add_argument("--disable_eval", type=str2bool, default=False)
    parser.add_argument("--final_test", type=str2bool, default=True)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", type=str2bool, default=True)

    # distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", type=str2bool, default=False)
    parser.add_argument("--dist_url", default="env://")

    # AMP / profiling / logging
    parser.add_argument("--use_amp", type=str2bool, default=True)
    parser.add_argument("--profile_efficiency", type=str2bool, default=True)
    parser.add_argument("--profile_batch_size", type=int, default=32)
    parser.add_argument("--save_history", type=str2bool, default=True)
    parser.add_argument("--measure_eval_latency", type=str2bool, default=False)
    parser.add_argument("--evaluate_before_training", type=str2bool, default=False)
    parser.add_argument("--report_wall_clock", type=str2bool, default=True)

    # W&B kept for compatibility. Actual logging is optional.
    parser.add_argument("--enable_wandb", type=str2bool, default=False)
    parser.add_argument("--project", default="parameter_efficient_tuning_cv", type=str)
    parser.add_argument("--wandb_ckpt", type=str2bool, default=False)

    return parser


def canonicalize_args(args):
    args.tuning_method = canonical_method(args.tuning_method)
    validate_method_status(
        args.tuning_method,
        allow_nonpaper_controls=bool(getattr(args, "allow_nonpaper_controls", False)),
        allow_paper_ablations=bool(getattr(args, "allow_paper_ablations", False)),
        allow_unverified_paper_reimplementations=bool(
            getattr(args, "allow_unverified_paper_reimplementations", False)
        ),
    )

    if args.tuning_method == "full":
        args.freeze_backbone = False
    else:
        args.freeze_backbone = True

    if args.fair_protocol:
        # One controlled optimization recipe for every PEFT baseline and TRSO.
        # Only full fine-tuning and linear probing use their explicitly allowed LRs.
        args.paper_hparams = False
        args.legacy_auto_hparams = False
        args.optimizer = args.fair_optimizer
        if args.tuning_method == "full":
            args.lr = float(args.fair_full_lr)
        elif args.tuning_method == "linear":
            args.lr = float(args.fair_linear_lr)
        else:
            args.lr = float(args.fair_peft_lr)
        args.weight_decay = float(args.fair_weight_decay)
        args.weight_decay_adapter = float(args.fair_weight_decay)
        args.warmup_epochs = int(args.fair_warmup_epochs)
        args.min_lr = float(args.fair_min_lr)
    elif args.optimizer == "auto":
        args.optimizer = "sgd" if args.tuning_method in {"prompt", "sidetune"} else "adamw"
    if args.paper_hparams:
        if args.tuning_method == "prompt":
            args.optimizer = "sgd"
            args.lr = 40.0
            args.weight_decay = 0.0
            args.weight_decay_adapter = 0.0
            args.epochs = 1000
            args.warmup_steps = 1000
            args.prompt_size = 30

    if args.peft_head_lr_scale <= 0:
        raise ValueError("--peft_head_lr_scale must be positive.")
    if args.adaptformer_dim <= 0:
        raise ValueError("--adaptformer_dim must be positive.")
    if args.adaptformer_dropout < 0 or args.adaptformer_dropout >= 1:
        raise ValueError("--adaptformer_dropout must be in [0, 1).")
    if args.repadapter_dim <= 0:
        raise ValueError("--repadapter_dim must be positive.")
    if args.repadapter_groups <= 0:
        raise ValueError("--repadapter_groups must be positive.")
    if args.repadapter_dim % args.repadapter_groups != 0:
        raise ValueError("--repadapter_groups must divide --repadapter_dim.")
    if args.repadapter_dropout < 0 or args.repadapter_dropout >= 1:
        raise ValueError("--repadapter_dropout must be in [0, 1).")
    if args.arc_dim <= 0:
        raise ValueError("--arc_dim must be positive.")
    if args.arc_dropout < 0 or args.arc_dropout >= 1:
        raise ValueError("--arc_dropout must be in [0, 1).")
    if args.piggyback_mask_scale < 0:
        raise ValueError("--piggyback_mask_scale cannot be negative.")
    if not (0.0 < float(args.val_ratio) < 1.0):
        raise ValueError("--val_ratio must be in (0, 1).")
    if args.task in {"multilabel", "regression"} and (args.mixup > 0 or args.cutmix > 0 or args.cutmix_minmax is not None):
        raise ValueError("Mixup/CutMix is currently supported only for single-label classification.")
    # Task compatibility is capability-based. Most feature/weight adapters are
    # task-head agnostic and can be evaluated on single-label, multi-label, or
    # regression targets. Visual Prompting remains single-label only because it
    # requires a source-to-target class mapping in the frozen source logit space.
    validate_method_task(args.tuning_method, args.task, allow_auto=True)
    return args


def save_json_on_master(obj: Dict, path: str):
    if utils.is_main_process():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)


def resolve_input_size_before_dataset(args) -> int:
    """Resolve --input_size=0 from pretrained metadata before transforms exist.

    Dataset transforms are constructed before the model in this repository, so
    changing input_size only inside the backbone factory is too late. This
    resolver keeps every method on a given backbone at the same valid spatial
    resolution and avoids accidental 96/224 mismatches.
    """
    requested = int(getattr(args, "input_size", 224))
    if requested > 0:
        return requested
    if getattr(args, "clip_model", None):
        return 224  # replaced by the actual CLIP preprocessing metadata later

    source = str(getattr(args, "model_source", "auto") or "auto").lower()
    backbone = str(getattr(args, "backbone", ""))
    if source in {"auto", "torchvision"}:
        try:
            weights, _ = _resolve_weights_multiapi(backbone, getattr(args, "weights", "DEFAULT"))
            if weights is not None and weights != "legacy_pretrained":
                crop = weights.transforms().crop_size
                size = int(crop[0] if isinstance(crop, (tuple, list)) else crop)
                if size > 0:
                    args.input_size = size
                    return size
        except Exception:
            pass
    if source in {"auto", "timm"}:
        try:
            import timm
            cfg = timm.models.get_pretrained_cfg(backbone)
            input_shape = getattr(cfg, "input_size", None) if cfg is not None else None
            if input_shape and len(input_shape) >= 3:
                size = int(input_shape[-1])
                if size > 0:
                    args.input_size = size
                    return size
        except Exception:
            pass
    args.input_size = 224
    print(f"[Warn] Could not infer native input size for {backbone!r}; using 224.")
    return 224


def resolve_preprocessing_before_dataset(args) -> dict:
    """Resolve preprocessing metadata without changing the submitted-paper defaults silently.

    Explicit repository mode preserves the historical ImageNet normalization. In
    auto mode, pretrained checkpoint metadata is used when it can be read safely.
    The resolved values are stored on ``args`` and written to the run manifest.
    """
    resolve_input_size_before_dataset(args)
    args.preprocess_source = "repository_default"
    args.preprocess_mean = None
    args.preprocess_std = None
    if str(getattr(args, "preprocess", "auto")).lower() != "auto":
        return current_preprocessing_profile(args).to_dict()

    source = str(getattr(args, "model_source", "auto") or "auto").lower()
    backbone = str(getattr(args, "backbone", ""))
    if source in {"auto", "torchvision"}:
        try:
            weights, _ = _resolve_weights_multiapi(backbone, getattr(args, "weights", "DEFAULT"))
            if weights is not None and weights != "legacy_pretrained":
                transform = weights.transforms()
                mean = getattr(transform, "mean", None)
                std = getattr(transform, "std", None)
                if mean is not None and std is not None:
                    args.preprocess_mean = tuple(float(x) for x in mean)
                    args.preprocess_std = tuple(float(x) for x in std)
                    args.preprocess_source = "torchvision_weights"
                    return current_preprocessing_profile(args).to_dict()
        except Exception:
            pass
    if source in {"auto", "timm"}:
        try:
            import timm
            cfg = timm.models.get_pretrained_cfg(backbone)
            if cfg is not None:
                mean = getattr(cfg, "mean", None)
                std = getattr(cfg, "std", None)
                if mean is not None and std is not None:
                    args.preprocess_mean = tuple(float(x) for x in mean)
                    args.preprocess_std = tuple(float(x) for x in std)
                    interpolation = getattr(cfg, "interpolation", None)
                    if interpolation:
                        args.train_interpolation = str(interpolation)
                    crop_pct = getattr(cfg, "crop_pct", None)
                    if crop_pct:
                        args.crop_ratio = float(crop_pct)
                    args.preprocess_source = "timm_pretrained_cfg"
        except Exception:
            pass
    return current_preprocessing_profile(args).to_dict()


def _resolve_weights_multiapi(backbone: str, weights_str: str):
    try:
        from torchvision.models import get_model_weights, get_weight  # noqa: F401
        has_new_api = True
    except Exception:
        has_new_api = False

    if not weights_str or str(weights_str).lower() in ("none", "scratch", "random"):
        return None, has_new_api
    if not has_new_api:
        return "legacy_pretrained", False
    if "." in weights_str:
        from torchvision.models import get_weight
        return get_weight(weights_str), True
    from torchvision.models import get_model_weights
    try:
        enum_cls = get_model_weights(backbone)
        member = weights_str.upper()
        if member == "DEFAULT":
            return enum_cls.DEFAULT, True
        if hasattr(enum_cls, member):
            return getattr(enum_cls, member), True
    except Exception:
        pass
    return None, True


def _infer_clip_input_size(preprocess):
    size = None
    try:
        ts = getattr(preprocess, "transforms", None) or []
        for t in ts:
            name = t.__class__.__name__.lower()
            if "centercrop" in name and hasattr(t, "size"):
                size = t.size[0] if isinstance(t.size, (tuple, list)) else int(t.size)
        if size is None:
            for t in ts:
                name = t.__class__.__name__.lower()
                if "resize" in name and hasattr(t, "size"):
                    val = t.size
                    size = max(val) if isinstance(val, (tuple, list)) else int(val)
    except Exception:
        size = None
    return size or 224


def _strip_wrapper_prefixes(name: str) -> str:
    prefixes = ("module.", "backbone.", "visual.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
    return name


def _is_head_key(k: str) -> bool:
    name = _strip_wrapper_prefixes(str(k)).lower()
    tokens = (
        "head.", "heads.", "fc.", "classifier.", "linear.", "aux_classifier.",
        "segmentation_head.", "decode_head.", "roi_heads.box_predictor.",
        "roi_heads.mask_predictor.", "roi_heads.keypoint_predictor.", "rpn.head.",
        "box_head.", "box_predictor.", "mask_head.", "mask_predictor.",
    )
    return name.startswith(tokens) or any(f".{token}" in name for token in tokens)


def _is_head_param(name: str) -> bool:
    return _is_head_key(name)


def _extract_checkpoint_model(ckpt: dict, model_key: str):
    for mk in str(model_key).split("|"):
        if mk in ckpt:
            return ckpt[mk]
    return ckpt


def load_matching_head(model: nn.Module, checkpoint_path: str, model_key: str = "model|module", strict: bool = True) -> int:
    """Load a shared task head before method calibration/optimizer construction.

    Fair suites use one head-only checkpoint per backbone and seed so all PEFT
    methods begin from the same task-aware classifier.  This is especially
    important for response-based TRSO calibration.
    """
    if not checkpoint_path:
        return 0
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    source = _extract_checkpoint_model(checkpoint, model_key) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(source, dict):
        raise TypeError(f"Head checkpoint does not contain a state dictionary: {checkpoint_path}")
    target = model.state_dict()
    matched = {}
    source_heads = {
        _strip_wrapper_prefixes(name): value
        for name, value in source.items()
        if _is_head_key(name) and hasattr(value, "shape")
    }
    target_heads = {name: value for name, value in target.items() if _is_head_key(name)}
    for raw_name, value in source_heads.items():
        for name in (raw_name, _strip_wrapper_prefixes(raw_name)):
            if name in target_heads and target_heads[name].shape == value.shape:
                matched[name] = value
                break
    # Wrapper baselines may rename the same classifier (for example fc -> head).
    # Match remaining tensors only when shape and weight/bias suffix identify a
    # unique source tensor, avoiding accidental loading into unrelated layers.
    for target_name, target_value in target_heads.items():
        if target_name in matched:
            continue
        suffix = target_name.rsplit(".", 1)[-1]
        candidates = [
            value for source_name, value in source_heads.items()
            if source_name.rsplit(".", 1)[-1] == suffix and value.shape == target_value.shape
        ]
        if len(candidates) == 1:
            matched[target_name] = candidates[0]
    if not matched:
        message = f"No compatible task-head parameters found in {checkpoint_path}"
        if strict:
            raise RuntimeError(message)
        print(f"[Warn] {message}")
        return 0
    model.load_state_dict(matched, strict=False)
    print(f"[Fair Head] Loaded {len(matched)} task-head tensors from {checkpoint_path}")
    return len(matched)


def safe_torch_load(path, map_location="cpu"):
    """Load a trusted PyTorch checkpoint across PyTorch versions.

    PyTorch 2.6 changed the default behavior of torch.load to
    weights_only=True. The checkpoints saved by this training script include
    argparse.Namespace and sometimes NumPy scalar objects in `args`, so loading
    them with the new default can fail with WeightsUnpickler errors.

    Use this only for checkpoints you created or otherwise trust.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not expose the weights_only argument.
        return torch.load(path, map_location=map_location)


class CLIPLinearProbe(nn.Module):
    def __init__(self, visual_module: nn.Module, feat_dim: int, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.visual = visual_module
        if freeze_backbone:
            for p in self.visual.parameters():
                p.requires_grad = False
            self.visual.eval()
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feats = self.visual(x)
        return self.head(feats)


class PixelPromptWrapper(nn.Module):
    """Apply the repository's original border prompt before any image backbone."""

    def __init__(self, backbone: nn.Module, prompt_size: int, image_size: int):
        super().__init__()
        from models.tuning_modules.prompter import PadPrompter
        self.tuning_module = PadPrompter(prompt_size=prompt_size, image_size=image_size)
        self.backbone = backbone
        self.backbone_family = getattr(backbone, "backbone_family", None)

    def forward(self, x):
        return self.backbone(self.tuning_module(x))


def _replace_classifier_head(model_backbone: nn.Module, num_classes: int, keep_pretrained_head: bool = True):
    """Replace a classification head across torchvision, timm, and local models.

    Returns True when a compatible head is found, including when its output
    dimension already matches. A missing head is an error at the call site.
    """
    num_classes = int(num_classes)

    if hasattr(model_backbone, "reset_classifier") and callable(model_backbone.reset_classifier):
        try:
            current = model_backbone.get_classifier() if hasattr(model_backbone, "get_classifier") else None
            if not (keep_pretrained_head and isinstance(current, nn.Linear) and current.out_features == num_classes):
                model_backbone.reset_classifier(num_classes)
            return True
        except Exception:
            pass

    def replace_attr(parent, name) -> bool:
        layer = getattr(parent, name, None)
        if isinstance(layer, nn.Linear):
            if not (keep_pretrained_head and layer.out_features == num_classes):
                setattr(parent, name, nn.Linear(layer.in_features, num_classes, bias=layer.bias is not None))
            return True
        if isinstance(layer, nn.Conv2d):
            if not (keep_pretrained_head and layer.out_channels == num_classes):
                setattr(parent, name, nn.Conv2d(layer.in_channels, num_classes, layer.kernel_size, layer.stride, layer.padding, bias=layer.bias is not None))
            return True
        if isinstance(layer, nn.Sequential):
            items = list(layer)
            for index in reversed(range(len(items))):
                child = items[index]
                if isinstance(child, nn.Linear):
                    if not (keep_pretrained_head and child.out_features == num_classes):
                        items[index] = nn.Linear(child.in_features, num_classes, bias=child.bias is not None)
                        setattr(parent, name, nn.Sequential(*items))
                    return True
                if isinstance(child, nn.Conv2d):
                    if not (keep_pretrained_head and child.out_channels == num_classes):
                        items[index] = nn.Conv2d(child.in_channels, num_classes, child.kernel_size, child.stride, child.padding, bias=child.bias is not None)
                        setattr(parent, name, nn.Sequential(*items))
                    return True
        return False

    for attribute in ("fc", "classifier", "linear", "head"):
        if replace_attr(model_backbone, attribute):
            return True

    # Torchvision VisionTransformer stores the classifier at heads.head.
    heads = getattr(model_backbone, "heads", None)
    if heads is not None:
        if replace_attr(heads, "head"):
            return True
        if isinstance(heads, nn.Sequential):
            items = list(heads)
            for index in reversed(range(len(items))):
                if isinstance(items[index], nn.Linear):
                    child = items[index]
                    if not (keep_pretrained_head and child.out_features == num_classes):
                        items[index] = nn.Linear(child.in_features, num_classes, bias=child.bias is not None)
                        model_backbone.heads = nn.Sequential(*items)
                    return True
    return False


def _freeze_batchnorm(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            if m.affine:
                if m.weight is not None:
                    try:
                        m.weight.requires_grad_(False)
                    except RuntimeError:
                        if torch.nn.utils.parametrize.is_parametrized(m, "weight"):
                            m.parametrizations.weight.original.requires_grad_(False)
                if m.bias is not None:
                    try:
                        m.bias.requires_grad_(False)
                    except RuntimeError:
                        if torch.nn.utils.parametrize.is_parametrized(m, "bias"):
                            m.parametrizations.bias.original.requires_grad_(False)




def _torchvision_classification_backbones():
    """Return torchvision image-classification builders only."""
    import torchvision
    blocked_modules = (".detection.", ".segmentation.", ".video.", ".optical_flow.", ".quantization.")
    names = []
    try:
        from torchvision.models import get_model_builder
        for name in torchvision.models.list_models():
            try:
                module_name = get_model_builder(name).__module__
            except Exception:
                continue
            if not any(token in module_name for token in blocked_modules):
                names.append(name)
    except Exception:
        names = [name for name in torchvision.models.list_models() if not name.startswith("video")]
    return sorted(set(names))

def _build_torchvision_or_hub_backbone(args):
    """Build from torchvision, timm, or supported CIFAR hubs."""
    import torchvision

    if args.pretrained is None:
        pretrained_flag = args.weights is not None and str(args.weights).lower() not in ("none", "scratch", "random")
    else:
        pretrained_flag = bool(args.pretrained)

    source = str(getattr(args, "model_source", "auto")).lower()
    hub_pattern = re.match(r"^cifar(10|100)_.+$", args.backbone) or args.backbone in (
        "cifar_resnet56", "resnet56_cifar", "akamaster_resnet56", "resnet56_cifar10"
    ) or re.match(r"^akamaster_resnet(20|32|44|56|110)$", args.backbone)

    if source in ("auto", "hub") and hub_pattern:
        if re.match(r"^cifar(10|100)_.+$", args.backbone) and args.cifar_hub in ("auto", "chenyaofo"):
            provider, entry = "chenyaofo/pytorch-cifar-models", args.backbone
            model_backbone = torch.hub.load(provider, entry, pretrained=pretrained_flag)
        else:
            provider = "akamaster/pytorch_resnet_cifar10"
            entry = args.backbone.replace("akamaster_", "") if args.backbone.startswith("akamaster_") else "resnet56"
            model_backbone = torch.hub.load(provider, entry)
        args.input_size = 32
        args.resolved_model_source = "hub"
        print(f"[Info] Loaded {entry} from {provider} (input_size=32).")
        return model_backbone
    if source == "hub":
        raise RuntimeError(f"Backbone '{args.backbone}' is not registered in the supported hub patterns.")

    try:
        tv_names = set(_torchvision_classification_backbones())
    except Exception:
        tv_names = {name for name in dir(torchvision.models) if callable(getattr(torchvision.models, name, None))}

    if source in ("auto", "torchvision") and args.backbone in tv_names:
        # An explicit --pretrained False must override a default weight enum;
        # otherwise offline and scratch experiments unexpectedly download weights.
        resolved_weight_request = args.weights if pretrained_flag else "none"
        tv_weights, has_new_api = _resolve_weights_multiapi(args.backbone, resolved_weight_request)
        kwargs = {}
        # Randomly initialized torchvision ViTs can be constructed for a custom image size.
        if tv_weights is None and any(token in args.backbone.lower() for token in ("vit_", "vision_transformer")):
            kwargs["image_size"] = int(args.input_size)
        if has_new_api:
            from torchvision.models import get_model
            try:
                model_backbone = get_model(args.backbone, weights=tv_weights, **kwargs)
            except TypeError:
                model_backbone = get_model(args.backbone, weights=tv_weights)
        else:
            function = getattr(torchvision.models, args.backbone)
            try:
                model_backbone = function(pretrained=tv_weights == "legacy_pretrained", **kwargs)
            except TypeError:
                model_backbone = function(pretrained=tv_weights == "legacy_pretrained")
        if tv_weights is not None:
            try:
                crop = tv_weights.transforms().crop_size
                args.input_size = int(crop[0] if isinstance(crop, (tuple, list)) else crop)
            except Exception:
                pass
        args.resolved_model_source = "torchvision"
        print(f"[Info] Loaded {args.backbone} from torchvision with weights={tv_weights}.")
        return model_backbone
    if source == "torchvision":
        raise RuntimeError(f"Backbone '{args.backbone}' is not available in torchvision.")

    if source in ("auto", "timm"):
        try:
            import timm
        except Exception as exc:
            raise RuntimeError(
                f"Backbone '{args.backbone}' was not found in torchvision and timm is unavailable. "
                "Install the requirements or choose --model_source torchvision."
            ) from exc
        if args.backbone not in set(timm.list_models(pretrained=False)):
            raise RuntimeError(f"Backbone '{args.backbone}' is not available in timm.")
        create_kwargs = {"pretrained": pretrained_flag}
        # Original visual prompting keeps the pretrained source classifier and
        # applies a fixed source-to-target label mapping. Other methods replace
        # the task head with the downstream class count.
        if args.tuning_method != "prompt":
            create_kwargs["num_classes"] = int(args.nb_classes)
        try:
            model_backbone = timm.create_model(args.backbone, img_size=int(args.input_size), **create_kwargs)
        except TypeError:
            model_backbone = timm.create_model(args.backbone, **create_kwargs)
        args.resolved_model_source = "timm"
        print(f"[Info] Loaded {args.backbone} from timm (pretrained={pretrained_flag}).")
        return model_backbone

    raise RuntimeError(f"Unable to resolve backbone '{args.backbone}' from source '{source}'.")


def _add_adapters(model_backbone: nn.Module, args):
    method = args.tuning_method
    adapter_param_ids = set()

    if method in ("full", "linear", "norm", "bias", "last_block", "bitfit", "prompt", "sidetune", "vpt_shallow", "vpt_deep", "vqt", "spt_lora", "spt_adapter"):
        return model_backbone, adapter_param_ids

    if method in ("conv", "adapter"):
        if "resnet50" not in str(args.backbone).lower():
            raise ValueError("Conv-Adapter paper reproduction requires ResNet-50.")
        from models.tuning_modules.conv_adapter import apply_conv_adapter_resnet50
        count = apply_conv_adapter_resnet50(
            model_backbone,
            mode=args.conv_adapter_mode,
            kernel_size=args.kernel_size,
            stages=(1, 2, 3, 4),
            reduction=args.adapt_size,
            adapt_scale=args.adapt_scale,
        )
        print(f"[Conv-Adapter] inserted {count} adapters using {args.conv_adapter_mode}.")

    elif method == "trso":
        print("[G-CREST-TRSO] Deferred evidence calibration until the training loader is available.")
        return model_backbone, adapter_param_ids

    elif method == "ssf":
        from models.tuning_modules.ssf import apply_ssf
        records = apply_ssf(model_backbone, init_std=args.ssf_init_std)
        print(f"[SSF] inserted {len(records)} paper-specified affine modules.")

    elif method == "lora":
        from models.tuning_modules.lora_transformer import apply_lora_transformer
        count = apply_lora_transformer(
            model_backbone,
            rank=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            merge_weights=bool(args.lora_merge_weights),
        )
        print(f"[LoRA] wrapped {count} Transformer Q/V attention projections.")

    elif method in {"convpass", "convpass_attn"}:
        from models.tuning_modules.convpass_transformer import apply_convpass
        records = apply_convpass(
            model_backbone, bottleneck=args.convpass_dim, scale=args.convpass_scale,
            dropout=args.convpass_dropout, attn_only=method == "convpass_attn",
        )
        print(f"[ConvPass] inserted {len(records)} convolutional bypass blocks ({method}).")

    elif method in {"fact_tt", "fact_tk"}:
        from models.tuning_modules.fact import apply_fact
        variant = method.split("_")[-1]
        resolved_rank = int(args.fact_rank) if int(args.fact_rank) > 0 else (4 if variant == "tt" else 8)
        records = apply_fact(
            model_backbone, variant=variant, rank=resolved_rank, scale=args.fact_scale
        )
        model_backbone._fact_rank = resolved_rank
        print(
            f"[FacT] registered {len(records)} factorized weight updates "
            f"({method}, rank={resolved_rank})."
        )

    elif method == "adaptformer":
        from models.tuning_modules.adaptformer import apply_adaptformer
        records = apply_adaptformer(
            model_backbone,
            bottleneck=args.adaptformer_dim,
            dropout=args.adaptformer_dropout,
            scale=args.adaptformer_scale,
            layernorm_option=args.adaptformer_layernorm,
        )
        print(f"[AdaptFormer] inserted {len(records)} parallel FFN adapters.")

    elif method == "repadapter":
        from models.tuning_modules.repadapter import apply_repadapter
        records = apply_repadapter(
            model_backbone,
            hidden_dim=args.repadapter_dim,
            groups=args.repadapter_groups,
            scale=args.repadapter_scale,
            dropout=args.repadapter_dropout,
        )
        print(f"[RepAdapter] inserted {2 * len(records)} RepBlock affine branches across {len(records)} ViT blocks.")

    elif method == "arc":
        from models.tuning_modules.arc import apply_arc
        records = apply_arc(model_backbone, adapter_dim=args.arc_dim, dropout=args.arc_dropout)
        print(
            f"[ARC] inserted {2 * len(records)} re-composed branches across {len(records)} ViT blocks "
            f"with adapter_dim={args.arc_dim}."
        )

    elif method == "piggyback":
        from models.tuning_modules.piggyback import apply_piggyback, piggyback_storage
        records = apply_piggyback(
            model_backbone,
            threshold=args.piggyback_threshold,
            mask_init=args.piggyback_mask_init,
            mask_scale=args.piggyback_mask_scale,
            mask_linear=bool(args.piggyback_mask_linear),
            exclude_task_head=True,
            backbone_name=str(getattr(args, "backbone", "")),
        )
        storage = piggyback_storage(model_backbone)
        print(
            f"[Piggyback] masked {len(records)} operations / {storage.masked_weights:,} weights; "
            f"deployed mask={storage.deployed_mask_megabytes:.3f} MiB."
        )

    else:
        raise ValueError(
            f"Unsupported strict paper baseline: {method}. "
            "LoRA-Conv and former hook approximations were removed from the paper-reproduction path."
        )

    return model_backbone, adapter_param_ids


def _last_block_prefixes(model: nn.Module) -> tuple[str, ...]:
    """Find the final reusable backbone stage without architecture-specific knobs."""
    module_names = [name for name, module in model.named_modules() if name and any(True for _ in module.parameters(recurse=False))]
    preferred_suffixes = (
        "layer4", "stages.3", "stage4", "features.7", "features.8", "features.12",
        "encoder.layers.11", "encoder.layers.encoder_layer_11", "backbone.body.layer4",
        "backbone.layer4", "model.backbone.body.layer4", "model.backbone.layer4",
    )
    matches = [name for name in module_names if any(name.endswith(suffix) for suffix in preferred_suffixes)]
    if matches:
        return tuple(sorted(set(matches), key=len))

    # Transformer/CNN block lists: select the highest numeric child within the
    # deepest repeated container (blocks.N, layers.N, stages.N, features.N).
    import re as _re
    candidates = []
    for name in module_names:
        match = _re.search(r"^(.*(?:blocks|layers|stages|features))\.(\d+)(?:\.|$)", name)
        if match:
            candidates.append((match.group(1), int(match.group(2))))
    if candidates:
        best_container = max(candidates, key=lambda row: (row[1], len(row[0])))[0]
        best_index = max(index for container, index in candidates if container == best_container)
        return (f"{best_container}.{best_index}",)

    # Conservative fallback: the last parameterized non-head module.
    for name in reversed(module_names):
        if not _is_head_key(name + ".weight"):
            return (name,)
    raise RuntimeError("Could not identify a final backbone block for partial fine-tuning.")


def set_trainability_policy(model: nn.Module, args, extra_adapter_param_ids: Optional[set] = None):
    method = args.tuning_method
    extra_adapter_param_ids = extra_adapter_param_ids or set()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if method == "full":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return model

    if method == "linear":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(_is_head_param(name))
        _freeze_batchnorm(model)
        return model

    if method == "norm":
        normalization_types = (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm,
            nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        )
        norm_parameter_ids = {
            id(parameter)
            for module in model.modules() if isinstance(module, normalization_types)
            for parameter in module.parameters(recurse=False)
        }
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(_is_head_param(name) or id(parameter) in norm_parameter_ids)
        return model

    if method == "bias":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(_is_head_param(name) or name.lower().endswith(".bias"))
        _freeze_batchnorm(model)
        return model

    if method == "last_block":
        prefixes = _last_block_prefixes(model)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(_is_head_param(name) or any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes))
        print(f"[Last-Block] trainable backbone prefixes: {prefixes}")
        return model

    if method == "prompt":
        # Original visual prompting optimizes only the prompt. The pretrained
        # output classifier remains frozen and is accessed through fixed labels.
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("tuning_module."))
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("Visual prompting produced no trainable prompt parameters")
        model.backbone.eval()
        if not getattr(args, "fair_protocol", False):
            args.weight_decay = 0.0
        return model

    if method in ("conv", "adapter"):
        from models.tuning_modules.conv_adapter import set_conv_adapter_trainability
        set_conv_adapter_trainability(model)
        _freeze_batchnorm(model)
        return model


    if method == "ssf":
        from models.tuning_modules.ssf import set_ssf_trainability
        set_ssf_trainability(model)
        return model

    if method == "lora":
        from models.tuning_modules.lora_transformer import mark_only_lora_as_trainable
        mark_only_lora_as_trainable(model, train_bias="none")
        for name, parameter in model.named_parameters():
            if _is_head_param(name):
                parameter.requires_grad_(True)
        return model

    if method == "bitfit":
        from models.tuning_modules.bitfit import set_bitfit_trainability
        set_bitfit_trainability(
            model,
            train_head=bool(args.bitfit_train_head),
            bias_scope=getattr(args, "bitfit_bias_scope", "all"),
        )
        if not getattr(args, "fair_protocol", False):
            args.weight_decay = 0.0
        return model

    if method in {"vpt_shallow", "vpt_deep"}:
        from models.tuning_modules.vpt import set_vpt_trainability
        set_vpt_trainability(model)
        return model

    if method == "vqt":
        from models.tuning_modules.vqt import set_vqt_trainability
        set_vqt_trainability(model)
        return model

    if method in {"convpass", "convpass_attn"}:
        from models.tuning_modules.convpass_transformer import set_convpass_trainability
        set_convpass_trainability(model)
        return model

    if method in {"fact_tt", "fact_tk"}:
        from models.tuning_modules.fact import set_fact_trainability
        set_fact_trainability(model)
        return model

    if method in {"spt_lora", "spt_adapter"}:
        # Allocation is data-derived after the loader is available.
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(_is_head_param(name))
        return model

    if method == "adaptformer":
        from models.tuning_modules.adaptformer import set_adaptformer_trainability
        set_adaptformer_trainability(model)
        return model

    if method == "repadapter":
        from models.tuning_modules.repadapter import set_repadapter_trainability
        set_repadapter_trainability(model)
        return model

    if method == "arc":
        from models.tuning_modules.arc import set_arc_trainability
        set_arc_trainability(model)
        return model

    if method == "piggyback":
        from models.tuning_modules.piggyback import set_piggyback_trainability
        set_piggyback_trainability(model, train_head=bool(args.piggyback_train_head))
        _freeze_batchnorm(model)
        return model

    if method == "sidetune":
        for name, parameter in model.named_parameters():
            trainable = name.startswith("side.") or name.startswith("head.") or name == "alpha_logit"
            parameter.requires_grad_(trainable)
        model.base.eval()
        return model

    if method == "trso":
        # The shared task head is the only warm-start coordinate before the
        # calibration data select the final cross-fitted tangent coordinates; the full task head remains trainable.
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(_is_head_param(name))
        _freeze_batchnorm(model)
        return model

    raise ValueError(f"Unsupported strict paper baseline: {method}")


def apply_shared_head_freeze(model: nn.Module, args) -> None:
    """Apply the same shared-head policy to every compatible PEFT method."""
    if not bool(getattr(args, "peft_freeze_head", False)) or not getattr(args, "head_from", ""):
        return
    if args.tuning_method in {"full", "linear", "prompt"}:
        return
    for name, parameter in model.named_parameters():
        if _is_head_param(name):
            parameter.requires_grad_(False)


class MaskedDepthLoss(nn.Module):
    """Depth loss that ignores invalid pixels and optionally adds SILog."""

    def __init__(self, mode: str = "silog_l1", min_depth: float = 1e-3, max_depth: float = 80.0):
        super().__init__()
        self.mode = str(mode)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

    def forward(self, prediction, target):
        prediction = prediction.float()
        target = target.float()
        valid = torch.isfinite(target) & torch.isfinite(prediction) & (target > 0)
        valid &= target >= self.min_depth
        if self.max_depth > self.min_depth:
            valid &= target <= self.max_depth
        if not valid.any():
            return prediction.sum() * 0.0
        pred = prediction[valid].clamp(self.min_depth, self.max_depth)
        truth = target[valid].clamp(self.min_depth, self.max_depth)
        if self.mode == "smooth_l1":
            return nn.functional.smooth_l1_loss(pred, truth)
        l1 = nn.functional.l1_loss(pred, truth)
        if self.mode == "l1":
            return l1
        log_error = torch.log(pred) - torch.log(truth)
        silog = torch.sqrt(torch.clamp(log_error.square().mean() - 0.85 * log_error.mean().square(), min=0.0) + 1e-12)
        return l1 + silog


def build_task_criterion(args, mixup_active: bool = False):
    task_type = getattr(args, "task_type", TASK_SINGLE_LABEL)
    if task_type == TASK_MULTILABEL:
        return nn.BCEWithLogitsLoss()
    if task_type == TASK_REGRESSION:
        return nn.SmoothL1Loss() if getattr(args, "regression_loss", "mse") == "smooth_l1" else nn.MSELoss()
    if task_type == TASK_SEMANTIC_SEGMENTATION:
        return nn.CrossEntropyLoss(ignore_index=int(getattr(args, "segmentation_ignore_index", 255)))
    if task_type == TASK_DEPTH_ESTIMATION:
        return MaskedDepthLoss(
            getattr(args, "depth_loss", "silog_l1"),
            getattr(args, "depth_min", 1e-3),
            getattr(args, "depth_max", 80.0),
        )
    if task_type == TASK_OBJECT_DETECTION:
        return None
    if mixup_active:
        return SoftTargetCrossEntropy()
    smoothing = float(getattr(args, "smoothing", 0.0) or 0.0)
    if smoothing > 0.0:
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    return nn.CrossEntropyLoss()


def primary_metric(task_type: str):
    spec = task_spec(task_type)
    return spec.primary_metric, spec.maximize


def format_primary(stats: Dict, task_type: str) -> str:
    spec = task_spec(task_type)
    value = float(stats.get(spec.primary_metric, float("nan")))
    return f"{spec.display_name}={value:.5f}"


def build_optimizer_parameter_groups(
    model: nn.Module,
    args,
    adapter_param_ids: Optional[set[int]] = None,
) -> list[dict]:
    """Create stable adapter/head/backbone groups with LR scaling and no-decay.

    Small PEFT tensors, classifier biases, normalization affine parameters and
    scalar gates are harmed by indiscriminate decay.  The head receives a lower
    LR after linear-probe initialization while all PEFT methods keep the same
    group policy for a controlled comparison.
    """
    adapter_param_ids = adapter_param_ids or set()
    buckets: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
    method = str(getattr(args, "tuning_method", ""))
    head_scale = (
        float(getattr(args, "peft_head_lr_scale", 1.0))
        if method not in {"full", "linear"}
        else 1.0
    )

    adapter_tokens = (
        "pet_adapter", "trso", "basis_atoms", "coefficients", "gate",
        "parametrizations", "mdl_tangent", "tangent_core", "core",
        "ssf", "lora", "adaptformer", "repadapter", "arc_projection_bank", "arc_", "piggyback", "mask_scores",
        "adapter", "side", "tuning_module", "prompt", "prompt_embeddings",
        "convpass", "attn_adapter", "mlp_adapter", "fact_bank", "query_tokens",
        "spt", "operation_cores", "operation_factors", "core_tensor",
    )

    def no_decay(name: str, parameter: torch.nn.Parameter) -> bool:
        if not bool(getattr(args, "no_decay_bias_norm", True)):
            return False
        lower = name.lower()
        return (
            parameter.ndim <= 1
            or lower.endswith(".bias")
            or "norm" in lower
            or ".bn" in lower
            or "gate" in lower
            or "scale" in lower
            or "shift" in lower
        )

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _is_head_param(name):
            kind = "head"
        elif any(token in name for token in adapter_tokens) or id(parameter) in adapter_param_ids:
            kind = "adapter"
        else:
            kind = "backbone"
        buckets.setdefault((kind, no_decay(name, parameter)), []).append(parameter)

    groups: list[dict] = []
    for (kind, exclude_decay), parameters in sorted(buckets.items()):
        if kind == "adapter":
            weight_decay = 0.0 if exclude_decay else float(args.weight_decay_adapter)
            lr_scale = 1.0
        elif kind == "head":
            weight_decay = 0.0 if exclude_decay else float(args.weight_decay)
            lr_scale = head_scale
        else:
            weight_decay = 0.0 if exclude_decay else float(args.weight_decay)
            lr_scale = 1.0
        groups.append({
            "params": parameters,
            "lr": float(args.lr) * lr_scale,
            "lr_scale": lr_scale,
            "weight_decay": weight_decay,
            "group_name": f"{kind}_{'no_decay' if exclude_decay else 'decay'}",
        })
    return groups


def _classification_logits(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if isinstance(output, dict):
        for key in ("logits", "out", "pred"):
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
    raise TypeError(f"Cannot extract classification logits from output type {type(output)!r}.")


def _move_training_batch(batch, device):
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise TypeError("TRSO calibration expects data-loader batches shaped as (images, labels, ...).")
    images, labels = batch[0], batch[1]
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def calibrate_trso_model(model: nn.Module, data_loader, device: torch.device, args) -> list[str]:
    """Attach Cross-Fitted Variance-Calibrated Evidence-Spectral Tangent Core."""
    from models.tuning_modules.mdl_tangent_core import calibrate_mdl_tangent_core

    was_training = model.training
    criterion = build_task_criterion(args, mixup_active=False)
    report = calibrate_mdl_tangent_core(
        model,
        data_loader,
        lambda logits, labels: criterion(logits, labels),
        device=device,
        logits_fn=_classification_logits,
        is_head=_is_head_param,
        batch_to_device=_move_training_batch,
        ablation=getattr(args, "trso_ablation", "full"),
        mode_count_rule=getattr(args, "trso_mode_count_rule", "geometric"),
        r_scale=getattr(args, "trso_r_scale", 1.0),
        fixed_r=getattr(args, "trso_fixed_r", 0),
        calibration_fraction=getattr(args, "trso_calibration_fraction", 1.0),
        calibration_max_batches=getattr(args, "trso_calibration_max_batches", 0),
        partition_mode=getattr(args, "trso_partition_mode", "alternating"),
        partition_seed=getattr(args, "trso_partition_seed", 0),
        svd_oversampling=getattr(args, "trso_svd_oversampling", 0),
        svd_power_iterations=getattr(args, "trso_svd_power_iterations", 2),
        svd_seed=getattr(args, "trso_svd_seed", 0),
    )
    _freeze_batchnorm(model)
    model.train(was_training)
    payload = getattr(model, "_mdl_tangent_report", report.to_dict())
    print(
        "[G-CREST-TRSO] "
        f"ablation={report.ablation}; selected={report.selected_tensors}/{report.candidate_tensors}; "
        f"adapter_parameters={report.adapter_parameters}; "
        f"head_parameters={report.head_trainable_parameters}; "
        f"basis_values={report.frozen_basis_values}; "
        f"D0={report.global_numerical_support}; D1={report.global_shannon_effective_modes:.2f}; "
        f"R={report.global_selected_modes}; rule={report.mode_count_rule}; "
        f"rank[min/med/max]={report.rank_min}/{report.rank_median:g}/{report.rank_max}"
    )
    if report.fallback_reason:
        print(f"[G-CREST-TRSO] fallback: {report.fallback_reason}")
    if args.output_dir:
        save_json_on_master(payload, os.path.join(args.output_dir, "mdl_tangent_calibration.json"))
    return [record.name for record in report.records if record.rank > 0]


def build_datasets(args):
    if build_dataset_split is None:
        dataset_train, args.nb_classes = build_dataset(args=args, is_train=True)
        dataset_val, _ = build_dataset(args=args, is_train=False) if not args.disable_eval else (None, args.nb_classes)
        dataset_test = None
        return dataset_train, dataset_val, dataset_test

    dataset_train, args.nb_classes = build_dataset_split(args=args, split="train")
    dataset_val = None
    dataset_test = None
    if not args.disable_eval:
        dataset_val, _ = build_dataset_split(args=args, split="val")
    if args.final_test:
        try:
            dataset_test, _ = build_dataset_split(args=args, split="test")
        except Exception as e:
            if args.allow_val_as_test:
                print(f"[Warn] test split unavailable for {args.dataset}: {e}. --allow_val_as_test=True, so validation is reused.")
                dataset_test = dataset_val
            else:
                raise RuntimeError(
                    f"A distinct test split is unavailable for dataset '{args.dataset}'. "
                    "Provide a test split or explicitly set --allow_val_as_test True."
                ) from e
    return dataset_train, dataset_val, dataset_test


def _seed_data_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_samplers(args, dataset_train, dataset_val, dataset_test):
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    if getattr(args, "distributed", False):
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed)
    else:
        sampler_generator = torch.Generator().manual_seed(int(args.seed) + int(global_rank))
        sampler_train = torch.utils.data.RandomSampler(dataset_train, generator=sampler_generator)

    def eval_sampler(ds):
        if ds is None:
            return None
        if getattr(args, "distributed", False) and args.dist_eval:
            return torch.utils.data.DistributedSampler(ds, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        return torch.utils.data.SequentialSampler(ds)

    return sampler_train, eval_sampler(dataset_val), eval_sampler(dataset_test)



@torch.no_grad()
def calibrate_prompt_frequency_mapping(
    model: nn.Module,
    data_loader,
    device: torch.device,
    num_classes: int,
    max_batches: int = 0,
) -> dict:
    """Estimate a one-to-one source/target label map from training data.

    Counts are collected from the frozen pretrained classifier without an
    input prompt. A maximum-weight rectangular assignment then selects one
    unique source logit for every downstream class. Only training examples are
    used; validation and test labels never participate.
    """
    from models.tuning_modules.prompter import VisualPromptingClassifier

    if not isinstance(model, VisualPromptingClassifier):
        raise TypeError("Frequency mapping requires VisualPromptingClassifier")
    backbone = model.backbone
    was_training = backbone.training
    backbone.eval()
    counts = None
    seen = torch.zeros(int(num_classes), dtype=torch.long, device=device)
    for batch_index, batch in enumerate(data_loader):
        if max_batches > 0 and batch_index >= int(max_batches):
            break
        images, targets = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True).long()
        logits = backbone(images)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim != 2:
            raise RuntimeError(f"Prompt source classifier must return [B,K], got {tuple(logits.shape)}")
        if logits.shape[1] < int(num_classes):
            raise RuntimeError("Source classifier has fewer outputs than downstream classes")
        if counts is None:
            counts = torch.zeros(int(num_classes), int(logits.shape[1]), dtype=torch.long, device=device)
        predictions = logits.argmax(dim=1)
        valid = (targets >= 0) & (targets < int(num_classes))
        flat = targets[valid] * counts.shape[1] + predictions[valid]
        counts.view(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=counts.dtype))
        seen.scatter_add_(0, targets[valid], torch.ones_like(targets[valid], dtype=seen.dtype))

    if counts is None:
        raise RuntimeError("Prompt frequency mapping received no training batches")
    if utils.is_dist_avail_and_initialized():
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(seen, op=torch.distributed.ReduceOp.SUM)
    if (seen == 0).any():
        missing = torch.where(seen == 0)[0].tolist()
        raise RuntimeError(f"Prompt mapping did not observe target classes {missing}")

    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(-counts.cpu().numpy())
    mapping = torch.empty(int(num_classes), dtype=torch.long)
    mapping[torch.as_tensor(rows, dtype=torch.long)] = torch.as_tensor(cols, dtype=torch.long)
    model.output_indices.copy_(mapping.to(model.output_indices.device))
    backbone.train(was_training)
    selected_counts = counts.cpu()[torch.arange(int(num_classes)), mapping]
    return {
        "strategy": "frequency_hungarian",
        "output_indices": mapping.tolist(),
        "class_support": seen.cpu().tolist(),
        "selected_frequency": selected_counts.tolist(),
        "mapping_precision": float(selected_counts.sum().item() / max(1, seen.sum().item())),
    }

def _parse_prompt_indices(value: str, num_classes: int):
    text = str(value or "").strip()
    if not text:
        return list(range(int(num_classes)))
    indices = [int(item.strip()) for item in text.split(",") if item.strip()]
    if len(indices) != int(num_classes):
        raise ValueError("--prompt_output_indices must contain exactly nb_classes indices")
    return indices


def build_model_for_experiment(args, clip_visual=None, clip_feat_dim=None):
    task_type = normalize_task(getattr(args, "task_type", getattr(args, "task", "single_label")))
    if task_type in {TASK_SEMANTIC_SEGMENTATION, TASK_DEPTH_ESTIMATION}:
        from models.structured_models import build_smp_model, build_torchvision_segmentation
        depth = task_type == TASK_DEPTH_ESTIMATION
        if args.model_source == "smp":
            encoder_weights = None if str(args.weights).lower() in {"none", "scratch", "random"} else "imagenet"
            model_backbone = build_smp_model(args.backbone, args.nb_classes, depth=depth, encoder_weights=encoder_weights)
            args.resolved_model_source = "smp"
        else:
            model_backbone = build_torchvision_segmentation(args.backbone, args.nb_classes, args.weights, depth=depth)
            args.resolved_model_source = "torchvision_segmentation"
        family = getattr(model_backbone, "backbone_family", detect_backbone_family(model_backbone, args.backbone, args.resolved_model_source))
        validate_method_backbone(args.tuning_method, family)
        args.backbone_family = family
        model_backbone.backbone_family = family
        print(f"[Compatibility] method={args.tuning_method} | task={task_type} | family={family} | source={args.resolved_model_source}")
        model_backbone, adapter_param_ids = _add_adapters(model_backbone, args)
        return model_backbone, adapter_param_ids

    if task_type == TASK_OBJECT_DETECTION:
        from models.structured_models import build_torchvision_detection
        if args.model_source not in {"auto", "torchvision"}:
            raise ValueError("Object detection currently uses torchvision detection models.")
        model = build_torchvision_detection(args.backbone, args.nb_classes, args.weights, args.input_size)
        family = getattr(model, "backbone_family", "cnn")
        validate_method_backbone(args.tuning_method, family)
        args.backbone_family = family
        args.resolved_model_source = "torchvision_detection"
        print(f"[Compatibility] method={args.tuning_method} | task={task_type} | family={family} | source=torchvision_detection")
        model, adapter_param_ids = _add_adapters(model, args)
        return model, adapter_param_ids

    # The original visual-prompting baseline requires a frozen pretrained
    # classifier output space. A feature-only CLIP visual encoder is therefore
    # not a faithful implementation and is rejected by the compatibility table.
    if args.clip_model:
        if clip_visual is None or clip_feat_dim is None:
            raise RuntimeError("CLIP requested but not initialized.")
        family = "clip_transformer" if str(args.clip_model).upper().startswith("VIT") else "clip_cnn"
        validate_method_backbone(args.tuning_method, family)
        if args.tuning_method not in ("full", "linear", "bitfit"):
            raise ValueError("Strict CLIP branch supports full, linear, and BitFit only.")
        model = CLIPLinearProbe(
            clip_visual,
            int(clip_feat_dim),
            args.nb_classes,
            freeze_backbone=bool(args.freeze_backbone),
        )
        model.backbone_family = family
        return model, set()


    model_backbone = _build_torchvision_or_hub_backbone(args)
    family = detect_backbone_family(
        model_backbone,
        backbone_name=args.backbone,
        source=getattr(args, "resolved_model_source", args.model_source),
    )
    if family == "unknown":
        raise RuntimeError(
            f"Could not determine the architecture family of '{args.backbone}'. "
            "Use a registered torchvision/timm model or add an explicit family detector."
        )
    ok, reason, _ = static_method_compatibility(
        args.tuning_method,
        args.backbone,
        task_type,
        source=getattr(args, "resolved_model_source", args.model_source),
        allow_nonpaper_controls=bool(getattr(args, "allow_nonpaper_controls", False)),
        allow_paper_ablations=bool(getattr(args, "allow_paper_ablations", False)),
        allow_unverified_paper_reimplementations=bool(
            getattr(args, "allow_unverified_paper_reimplementations", False)
        ),
    )
    if not ok:
        raise ValueError(reason)
    validate_method_backbone(args.tuning_method, family)
    args.backbone_family = family
    model_backbone.backbone_family = family
    print(f"[Compatibility] method={args.tuning_method} | family={family} | source={getattr(args, 'resolved_model_source', 'unknown')}")

    if args.tuning_method == "prompt":
        from models.tuning_modules.prompter import VisualPromptingClassifier
        indices = _parse_prompt_indices(args.prompt_output_indices, args.nb_classes)
        return VisualPromptingClassifier(
            backbone=model_backbone,
            num_classes=args.nb_classes,
            prompt_size=args.prompt_size,
            image_size=args.input_size,
            output_indices=indices,
            prompt_type=args.prompt_type,
        ), set()

    if args.tuning_method == "sidetune":
        from models.tuning_modules.side_tuning import SideTuningClassifier
        model = SideTuningClassifier(
            base_model=model_backbone,
            num_classes=args.nb_classes,
            learn_alpha=bool(args.sidetune_learn_alpha),
            alpha_init=float(args.sidetune_alpha),
            side_arch=args.sidetune_arch,
            side_width=args.sidetune_width,
            side_depth=args.sidetune_depth,
            side_checkpoint=args.sidetune_checkpoint,
        )
        model.backbone_family = family
        return model, set()

    if not _replace_classifier_head(model_backbone, args.nb_classes, keep_pretrained_head=args.keep_pretrained_head):
        raise RuntimeError(
            f"No replaceable task head was found for backbone '{args.backbone}'. "
            "This run is stopped to avoid training with an incorrect output dimension."
        )

    if args.tuning_method in {"vpt_shallow", "vpt_deep"}:
        from models.tuning_modules.vpt import VisualPromptTuning
        wrapped = VisualPromptTuning(
            model_backbone, num_tokens=args.vpt_num_tokens,
            deep=args.tuning_method == "vpt_deep", dropout=args.vpt_dropout,
        )
        wrapped.backbone_family = family
        return wrapped, set()

    if args.tuning_method == "vqt":
        from models.tuning_modules.vqt import VisualQueryTuning
        wrapped = VisualQueryTuning(
            model_backbone, num_classes=args.nb_classes, query_length=args.vqt_query_length
        )
        wrapped.backbone_family = family
        return wrapped, set()

    model_backbone, adapter_param_ids = _add_adapters(model_backbone, args)
    return model_backbone, adapter_param_ids


def main(args):
    run_wall_start = time.perf_counter()
    proposal_calibration_time_sec = 0.0
    profiling_time_sec = 0.0
    final_evaluation_time_sec = 0.0
    args = canonicalize_args(args)

    if args.list_compatibility:
        for row in compatibility_rows():
            print(
                f"{row['method']:12s} | families={','.join(row['families']):38s} "
                f"| tasks={','.join(row['tasks']):34s} | category={row['category']:19s} "
                f"| {row['implementation_scope']}"
            )
        return

    if args.list_backbones:
        import torchvision
        print("[torchvision]")
        for name in _torchvision_classification_backbones():
            print(name)
        try:
            import timm
            print("[timm]")
            for name in timm.list_models(pretrained=False):
                print(name)
        except Exception:
            print("[timm unavailable: install timm to enable its backbone catalogue]")
        from models.structured_models import available_structured_backbones
        for group, names in available_structured_backbones().items():
            print(f"[{group}]")
            for name in names:
                print(name)
        print("[datasets]")
        for name in available_datasets():
            print(name)
        return

    utils.init_distributed_mode(args)

    if str(args.device).lower().startswith("cuda") and not torch.cuda.is_available():
        print("[Info] CUDA not available — falling back to CPU.")
        args.device = "cpu"
        args.use_amp = False
    elif str(args.device).lower() == "cpu":
        args.use_amp = False
        args.pin_mem = False

    device = torch.device(args.device)
    print(args)
    print(f"[Info] Using device: {device}  (AMP={'on' if args.use_amp else 'off'})")

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = device.type == "cuda"

    # Resolve automatic spatial size and checkpoint-consistent preprocessing before datasets exist.
    preprocessing_profile = resolve_preprocessing_before_dataset(args)

    clip_preprocess = None
    clip_visual = None
    clip_feat_dim = None
    if args.clip_model:
        ok = False
        try:
            import clip
            clip_model_full, clip_preprocess = clip.load(args.clip_model, device="cpu", jit=False)
            clip_visual = clip_model_full.visual
            clip_feat_dim = getattr(clip_visual, "output_dim", None)
            if clip_feat_dim is None:
                size = _infer_clip_input_size(clip_preprocess)
                with torch.no_grad():
                    clip_feat_dim = clip_visual(torch.zeros(1, 3, size, size)).shape[1]
            ok = True
            print(f"[CLIP] Loaded OpenAI {args.clip_model}")
        except Exception as e:
            print(f"[CLIP] OpenAI CLIP load failed ({e}). Trying OpenCLIP ...")
            try:
                import open_clip
                model_full, _, clip_preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrained)
                clip_visual = model_full.visual
                clip_feat_dim = getattr(clip_visual, "output_dim", None)
                if clip_feat_dim is None:
                    size = _infer_clip_input_size(clip_preprocess)
                    with torch.no_grad():
                        clip_feat_dim = clip_visual(torch.zeros(1, 3, size, size)).shape[1]
                ok = True
                print(f"[CLIP] Loaded OpenCLIP {args.clip_model}")
            except Exception as e2:
                raise RuntimeError(f"Failed to load CLIP via OpenAI and OpenCLIP: {e2}")
        if ok:
            args.input_size = _infer_clip_input_size(clip_preprocess)

    dataset_train, dataset_val, dataset_test = build_datasets(args)
    # Dataset construction resolves --task=auto and output dimensionality. Check
    # the actual task before model construction so unsupported combinations do
    # not fail halfway through an experiment.
    validate_method_task(args.tuning_method, args.task_type, allow_auto=False)

    if args.output_dir and utils.is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        try:
            dataset_spec = get_dataset_spec(args.dataset).to_dict()
        except Exception:
            dataset_spec = {"name": str(args.dataset), "source": "unknown", "download_policy": "unknown"}
        dataset_protocol = {
            "dataset": args.dataset,
            "dataset_spec": dataset_spec,
            "task_type": args.task_type,
            "output_dim": int(args.nb_classes),
            "split_seed": int(args.split_seed),
            "train_samples": len(dataset_train),
            "validation_samples": 0 if dataset_val is None else len(dataset_val),
            "test_samples": 0 if dataset_test is None else len(dataset_test),
            "allow_val_as_test": bool(args.allow_val_as_test),
            "download_mode": str(args.download),
            "preprocessing": preprocessing_profile,
            "input_size": int(args.input_size),
        }
        save_json_on_master(dataset_protocol, os.path.join(args.output_dir, "dataset_protocol.json"))
        save_json_on_master({
            "dataset": args.dataset,
            "task_type": args.task_type,
            "backbone": args.backbone,
            "model_source": args.model_source,
            "weights": args.weights,
            "seed": int(args.seed),
            "split_seed": int(args.split_seed),
            "tuning_method": args.tuning_method,
            "preprocessing": preprocessing_profile,
            "dataset_spec": dataset_spec,
        }, os.path.join(args.output_dir, "run_manifest.json"))

    if args.clip_model and clip_preprocess is not None:
        for ds in (dataset_train, dataset_val, dataset_test):
            if ds is None:
                continue
            target = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
            if hasattr(target, "transform"):
                target.transform = clip_preprocess
            if hasattr(target, "transforms"):
                target.transforms = clip_preprocess

    sampler_train, sampler_val, sampler_test = build_samplers(args, dataset_train, dataset_val, dataset_test)

    if utils.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    loader_generator = torch.Generator().manual_seed(seed)
    drop_last = len(dataset_train) >= args.batch_size
    collate_fn = None
    if args.task_type == TASK_OBJECT_DETECTION:
        from datasets.structured import detection_collate
        collate_fn = detection_collate
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=drop_last,
        worker_init_fn=_seed_data_worker,
        generator=loader_generator,
        collate_fn=collate_fn,
    )
    data_loader_val = None
    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=min(max(1, args.batch_size), 256),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            worker_init_fn=_seed_data_worker,
            generator=torch.Generator().manual_seed(seed + 1000),
            collate_fn=collate_fn,
        )
    data_loader_test = None
    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=min(max(1, args.batch_size), 256),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            worker_init_fn=_seed_data_worker,
            generator=torch.Generator().manual_seed(seed + 2000),
            collate_fn=collate_fn,
        )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active and args.task_type != TASK_SINGLE_LABEL:
        raise ValueError("Mixup/CutMix is supported only for single-label classification.")
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    model, adapter_param_ids = build_model_for_experiment(args, clip_visual=clip_visual, clip_feat_dim=clip_feat_dim)
    model = set_trainability_policy(model, args, extra_adapter_param_ids=adapter_param_ids)
    model.to(device)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        save_json_on_master(vars(args), os.path.join(args.output_dir, "args.json"))
        environment = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "device_requested": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" and torch.cuda.is_available() else None,
            "gpu_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" and torch.cuda.is_available() else None,
        }
        try:
            import torchvision
            environment["torchvision"] = torchvision.__version__
        except Exception:
            environment["torchvision"] = None
        try:
            import timm
            environment["timm"] = getattr(timm, "__version__", None)
        except Exception:
            environment["timm"] = None
        save_json_on_master(environment, os.path.join(args.output_dir, "environment.json"))
        protocol = {
            "fair_protocol": bool(args.fair_protocol),
            "optimizer": args.optimizer,
            "scheduler": "cosine",
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "adapter_weight_decay": float(args.weight_decay_adapter),
            "warmup_epochs": int(args.warmup_epochs),
            "warmup_steps": int(args.warmup_steps),
            "minimum_learning_rate": float(args.min_lr),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "update_frequency": int(args.update_freq),
            "augmentation": {
                "train_aug": args.train_aug, "mixup": args.mixup, "cutmix": args.cutmix,
                "color_jitter": args.color_jitter, "auto_augment": args.aa,
                "random_erasing": args.reprob, "label_smoothing": args.smoothing,
            },
            "dataset": args.dataset,
            "task_type": args.task_type,
            "backbone": args.backbone,
            "method": args.tuning_method,
            "seed": int(args.seed),
            "split_seed": int(args.split_seed),
        }
        # This fingerprint is identical for all PEFT rows in a controlled suite
        # except for method/backbone/seed metadata and method-specific structure.
        import hashlib
        fairness_fields = {k: v for k, v in protocol.items() if k not in {"method", "seed"}}
        protocol["protocol_fingerprint"] = hashlib.sha256(
            json.dumps(fairness_fields, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        save_json_on_master(protocol, os.path.join(args.output_dir, "resolved_protocol.json"))

    # Optional finetune checkpoint.
    if args.finetune:
        checkpoint = torch.hub.load_state_dict_from_url(args.finetune, map_location="cpu", check_hash=True) if args.finetune.startswith("https") else safe_torch_load(args.finetune, map_location="cpu")
        checkpoint_model = _extract_checkpoint_model(checkpoint, args.model_key)
        state_dict = model.state_dict()
        for k in list(checkpoint_model.keys()):
            if _is_head_key(k) and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint (shape mismatch)")
                del checkpoint_model[k]
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    # Shared task-aware head must be loaded before TRSO response calibration.
    # Resume checkpoints take precedence and are loaded immediately afterward.
    if args.head_from:
        load_matching_head(model, args.head_from, args.model_key, strict=True)
    apply_shared_head_freeze(model, args)

    # Load model state before TRSO calibration and before optimizer construction.
    # This is required because calibration determines which original tensors receive projected gradients.
    resume_checkpoint = utils.load_model_for_resume(args, model, strict=True)

    if (
        args.tuning_method == "prompt"
        and resume_checkpoint is None
        and not args.prompt_output_indices
        and args.prompt_mapping == "frequency"
    ):
        mapping_report = calibrate_prompt_frequency_mapping(
            model,
            data_loader_train,
            device,
            args.nb_classes,
            max_batches=args.prompt_mapping_batches,
        )
        print(f"[VisualPrompt] frequency mapping precision={mapping_report['mapping_precision']:.4f}")
        if args.output_dir and utils.is_main_process():
            save_json_on_master(mapping_report, os.path.join(args.output_dir, "prompt_mapping.json"))

    if args.tuning_method == "trso":
        if resume_checkpoint is not None:
            raise RuntimeError(
                "Direct resume of a calibrated parameter-free TRSO training run requires its transient data-derived projection bases. "
                "Start from --head_from or rerun the deterministic calibration pass."
            )
        calibration_started = time.perf_counter()
        calibrate_trso_model(model, data_loader_train, device, args)
        proposal_calibration_time_sec = time.perf_counter() - calibration_started
        print(f"[Timing] G-CREST-TRSO calibration: {proposal_calibration_time_sec:.3f} s")

    if args.tuning_method in {"spt_lora", "spt_adapter"}:
        if resume_checkpoint is not None:
            raise RuntimeError("SPT resume requires rerunning its data-derived sensitivity allocation.")
        from models.tuning_modules.spt import calibrate_spt
        calibration_started = time.perf_counter()
        report = calibrate_spt(
            model, data_loader_train, device,
            variant="lora" if args.tuning_method == "spt_lora" else "adapter",
            budget=args.spt_budget, sensitivity_samples=args.spt_sensitivity_samples,
            rank=args.spt_rank, adapter_dim=args.spt_adapter_dim, alpha=args.spt_alpha,
            output_path=(os.path.join(args.output_dir, "spt_sensitivity_report.json") if args.output_dir else None),
        )
        proposal_calibration_time_sec = time.perf_counter() - calibration_started
        print(
            f"[SPT] selected={report['selected_connections']:,} "
            f"actual={report['actual_backbone_trainable_parameters']:,} "
            f"structured={report['structured_matrices']} sparse={report['sparse_matrices']}"
        )
        print(f"[Timing] SPT sensitivity/allocation: {proposal_calibration_time_sec:.3f} s")

    # Old approximate memory profile retained for compatibility.
    try:
        memory_cost, detailed_info = profile_memory_cost(
            model,
            (1, 3, args.input_size, args.input_size),
            True,
            activation_bits=32,
            trainable_param_bits=32,
            frozen_param_bits=8,
            batch_size=min(8, args.batch_size),
            task_type=args.task_type,
        )
        print(f"memory_cost_MB: {memory_cost / 1e6:.3f}")
        print(f"param_size_MB: {detailed_info['param_size'] / 1e6:.3f}")
        print(f"act_size_MB: {detailed_info['act_size'] / 1e6:.3f}")
    except Exception as e:
        print(f"[Warn] memory_utils profile failed: {e}")

    if args.profile_efficiency:
        profiling_started = time.perf_counter()
        try:
            from tools.profile_efficiency import profile_model, save_profile
            profile = profile_model(
                model=model,
                device=device,
                input_size=args.input_size,
                batch_size=min(args.profile_batch_size, args.batch_size),
                use_amp=args.use_amp,
                task_type=args.task_type,
            )
            profile["deployment_state"] = (
                "unmerged_training_graph"
                if args.tuning_method == "trso" and args.trso_fast_inference
                else "runtime_graph"
            )
            print("[Efficiency Profile]", profile)
            if args.output_dir:
                filename = (
                    "efficiency_profile_unmerged.json"
                    if args.tuning_method == "trso" and args.trso_fast_inference
                    else "efficiency_profile.json"
                )
                save_profile(profile, os.path.join(args.output_dir, filename))
        except Exception as e:
            print(f"[Warn] efficiency profiling failed: {e}")
        finally:
            profiling_time_sec += time.perf_counter() - profiling_started

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(model, decay=args.model_ema_decay, device="cpu" if args.model_ema_force_cpu else "", resume="")
        print(f"Using EMA with decay = {args.model_ema_decay:.8f}")

    model_without_ddp = model
    if getattr(args, "distributed", False):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu] if device.type == "cuda" else None)
        model_without_ddp = model.module

    n_trainable = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model_without_ddp.parameters())
    print(f"Number of trainable params: {n_trainable:,}")
    print(f"Number of total params: {n_total:,}")
    head_trainable = sum(p.numel() for name, p in model_without_ddp.named_parameters() if p.requires_grad and _is_head_param(name))
    adapter_trainable = max(0, n_trainable - head_trainable)
    parameter_summary = {
        "trainable_params": int(n_trainable),
        "adapter_trainable_params": int(adapter_trainable),
        "head_trainable_params": int(head_trainable),
        "total_params": int(n_total),
        "trainable_ratio": float(n_trainable / max(1, n_total)),
        "adapter_trainable_ratio": float(adapter_trainable / max(1, n_total)),
        "method": args.tuning_method,
        "method_category": method_category(args.tuning_method),
        "strict_original_paper_baseline": method_category(args.tuning_method) == "paper_baseline",
        "original_paper": ORIGINAL_PAPER_REFERENCES.get(
            "conv" if args.tuning_method == "adapter" else args.tuning_method, ""
        ),
        "nonpaper_control_opt_in": bool(getattr(args, "allow_nonpaper_controls", False)),
        "backbone": args.backbone,
    }
    if args.tuning_method == "piggyback":
        from models.tuning_modules.piggyback import piggyback_storage
        storage = piggyback_storage(model_without_ddp)
        parameter_summary.update({
            "piggyback_masked_weights": storage.masked_weights,
            "piggyback_deployed_mask_bits": storage.deployed_mask_bits,
            "piggyback_deployed_mask_megabytes": storage.deployed_mask_megabytes,
            "piggyback_training_score_megabytes_fp32": storage.training_score_megabytes_fp32,
        })
    if args.tuning_method in {"spt_lora", "spt_adapter"}:
        spt_report = getattr(model_without_ddp, "_spt_report", {})
        if isinstance(spt_report, dict):
            parameter_summary.update({
                "spt_requested_budget": int(spt_report.get("requested_budget", 0) or 0),
                "spt_selected_connections": int(spt_report.get("selected_connections", 0) or 0),
                "spt_actual_backbone_trainable_parameters": int(spt_report.get("actual_backbone_trainable_parameters", 0) or 0),
                "spt_structured_matrices": int(spt_report.get("structured_matrices", 0) or 0),
                "spt_sparse_matrices": int(spt_report.get("sparse_matrices", 0) or 0),
                "spt_sensitivity_samples": int(spt_report.get("sensitivity_samples", 0) or 0),
            })
    if args.tuning_method == "trso":
        mdl_report = getattr(model_without_ddp, "_mdl_tangent_report", {})
        if isinstance(mdl_report, dict):
            mdl_adapter = int(mdl_report.get("adapter_parameters", 0) or 0)
            mdl_basis = int(mdl_report.get("frozen_basis_values", 0) or 0)
            parameter_summary.update({
                "adapter_trainable_params": mdl_adapter,
                "backbone_trainable_params": 0,
                "proposal_added_parameters": mdl_adapter,
                "zero_added_parameter": False,
                "trso_ablation": mdl_report.get("ablation", getattr(args, "trso_ablation", "full")),
                "mdl_adapter_parameters": mdl_adapter,
                "mdl_proposal_added_parameters": mdl_adapter,
                "mdl_effective_update_coordinates": mdl_adapter,
                "mdl_head_policy": mdl_report.get("head_policy"),
                "mdl_head_trainable_parameters": int(
                    mdl_report.get("head_trainable_parameters", 0) or 0
                ),
                "mdl_frozen_basis_values": mdl_basis,
                "mdl_frozen_basis_megabytes_fp32": float(mdl_basis * 4 / 1024**2),
                "mdl_deployed_extra_parameters": 0,
                "mdl_deployed_total_params": int(n_total - mdl_adapter),
                "mdl_global_selected_modes": int(mdl_report.get("global_selected_modes", 0) or 0),
                "mdl_global_shannon_effective_modes": float(mdl_report.get("global_shannon_effective_modes", 0.0) or 0.0),
                "mdl_global_numerical_support": int(mdl_report.get("global_numerical_support", 0) or 0),
                "mdl_candidate_modes": int(mdl_report.get("candidate_modes", 0) or 0),
                "mdl_rank_min": int(mdl_report.get("rank_min", 0) or 0),
                "mdl_rank_median": float(mdl_report.get("rank_median", 0.0) or 0.0),
                "mdl_rank_max": int(mdl_report.get("rank_max", 0) or 0),
                "mdl_mode_count_rule": mdl_report.get("mode_count_rule", "geometric"),
                "mdl_r_scale": float(mdl_report.get("r_scale", 1.0) or 1.0),
                "mdl_calibration_fraction": float(mdl_report.get("calibration_fraction", 1.0) or 1.0),
                "mdl_partition_mode": mdl_report.get("partition_mode", "alternating"),
                "mdl_partition_seed": int(mdl_report.get("partition_seed", 0) or 0),
            })
    if args.output_dir:
        save_json_on_master(parameter_summary, os.path.join(args.output_dir, "parameter_summary.json"))

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = max(
        1, math.ceil(len(data_loader_train) / max(1, args.update_freq))
    )
    print(f"LR = {args.lr:.8f}")
    print(f"Batch size = {total_batch_size}")
    print(f"Number of training examples = {len(dataset_train)}")
    print(f"Number of training steps per epoch = {num_training_steps_per_epoch}")

    parameter_groups = build_optimizer_parameter_groups(
        model_without_ddp, args, adapter_param_ids=adapter_param_ids
    )
    for group in parameter_groups:
        print(
            f"[ParamGroups] {group['group_name']}: "
            f"params={sum(p.numel() for p in group['params']):,}, "
            f"lr_scale={group['lr_scale']:g}, wd={group['weight_decay']:g}"
        )
    if not parameter_groups:
        raise RuntimeError("No trainable parameters were supplied to the optimizer")
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(parameter_groups, lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, betas=(0.9, 0.999), eps=args.opt_eps)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    print(f"Optimizer = {optimizer.__class__.__name__}")
    loss_scaler = NativeScaler()

    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)

    criterion = build_task_criterion(args, mixup_active=mixup_fn is not None)
    eval_criterion = build_task_criterion(args, mixup_active=False)
    print(f"criterion = {criterion} | task={args.task_type}")

    restored_training_state = utils.restore_optimizer_state(
        args, resume_checkpoint, optimizer, loss_scaler, model_ema
    )

    initial_stats = None
    if args.evaluate_before_training and data_loader_val is not None and resume_checkpoint is None:
        initial_stats = evaluate(
            data_loader_val, model, device, use_amp=args.use_amp,
            task_type=args.task_type, criterion=eval_criterion,
        )
        print(f"Initial validation before adaptation: {format_primary(initial_stats, args.task_type)}")
        if args.output_dir:
            save_json_on_master(initial_stats, os.path.join(args.output_dir, "initial_validation_summary.json"))

    metric_name, maximize_metric = primary_metric(args.task_type)

    if args.eval:
        print("Eval only mode")
        eval_loader = data_loader_test if data_loader_test is not None else data_loader_val
        eval_ds = dataset_test if data_loader_test is not None else dataset_val
        if eval_loader is None:
            raise RuntimeError("No evaluation loader available.")
        if args.tuning_method == "trso" and args.trso_fast_inference:
            from models.tuning_modules.mdl_tangent_core import merge_mdl_tangent_cores_
            merged = merge_mdl_tangent_cores_(model_without_ddp)
            print(f"[G-CREST-TRSO] exactly merged {merged} updates; runtime adapters=0")
            if args.profile_efficiency:
                from tools.profile_efficiency import profile_model, save_profile
                profile = profile_model(
                    model=model_without_ddp,
                    device=device,
                    input_size=args.input_size,
                    batch_size=min(args.profile_batch_size, args.batch_size),
                    use_amp=args.use_amp,
                    task_type=args.task_type,
                )
                profile["deployment_state"] = "exactly_merged"
                if args.output_dir:
                    save_profile(profile, os.path.join(args.output_dir, "efficiency_profile.json"))
        if args.tuning_method in {"fact_tt", "fact_tk"}:
            from models.tuning_modules.fact import merge_fact_
            merged = merge_fact_(model_without_ddp)
            print(f"[FacT] exactly merged {merged} factorized weight updates.")
        if args.tuning_method == "repadapter" and args.repadapter_merge:
            from models.tuning_modules.repadapter import merge_repadapter_
            merged = merge_repadapter_(model_without_ddp)
            print(f"[RepAdapter] exactly folded {merged} adapter branches into ViT projections; runtime adapters=0.")
        if args.tuning_method == "arc" and args.arc_merge:
            from models.tuning_modules.arc import merge_arc_
            model_without_ddp.eval()
            merged = merge_arc_(model_without_ddp)
            print(f"[ARC] exactly folded {merged} re-composed branches into ViT projections; runtime ARC branches=0.")
        if args.tuning_method == "spt_lora":
            from models.tuning_modules.spt import merge_spt_
            merged = merge_spt_(model_without_ddp)
            print(f"[SPT-LoRA] exactly merged {merged} sparse/low-rank updates.")
        stats = evaluate(
            eval_loader,
            model,
            device,
            use_amp=args.use_amp,
            measure_latency=args.measure_eval_latency,
            task_type=args.task_type,
            criterion=eval_criterion,
        )
        print(f"Evaluation on {len(eval_ds)} samples: {format_primary(stats, args.task_type)}")
        if args.output_dir:
            save_json_on_master(stats, os.path.join(args.output_dir, "eval_summary.json"))
        return

    default_best = float("-inf") if maximize_metric else float("inf")
    best_val_metric = float(restored_training_state.get("best_val_metric", default_best))
    best_epoch = int(restored_training_state.get("best_epoch", -1))
    history = list(restored_training_state.get("history", []))

    # Only genuinely trained epochs are eligible for TRSO model selection.
    # The zero-update Linear state is recorded for analysis but is never a fallback.

    training_perf_start = time.perf_counter()
    start_time = time.time()

    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):
        if getattr(args, "distributed", False) and hasattr(data_loader_train.sampler, "set_epoch"):
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            mixup_fn,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
            use_amp=args.use_amp,
            task_type=args.task_type,
        )

        val_stats = {}
        is_better = False
        if data_loader_val is not None:
            val_stats = evaluate(
                data_loader_val,
                model,
                device,
                use_amp=args.use_amp,
                task_type=args.task_type,
                criterion=eval_criterion,
            )
            current_metric = float(val_stats[metric_name])
            is_better = current_metric > best_val_metric if maximize_metric else current_metric < best_val_metric
            print(f"Validation on {len(dataset_val)} samples: {format_primary(val_stats, args.task_type)}")
            if is_better:
                best_val_metric = current_metric
                best_epoch = epoch
            best_stats = {metric_name: best_val_metric}
            print(f"Best validation {format_primary(best_stats, args.task_type)} at epoch {best_epoch}")

        if log_writer is not None and val_stats:
            for key, value in val_stats.items():
                if isinstance(value, (int, float)):
                    log_writer.update(**{f"val_{key}": value}, head="perf", step=epoch)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            "epoch": epoch,
            "n_trainable_parameters": n_trainable,
            "n_total_parameters": n_total,
            f"best_val_{metric_name}": best_val_metric,
            "best_epoch": best_epoch,
            "primary_metric": metric_name,
            "maximize_primary_metric": maximize_metric,
        }
        history.append(log_stats)

        checkpoint_training_state = {
            "best_val_metric": best_val_metric,
            "best_epoch": best_epoch,
            "history": history,
            "primary_metric": metric_name,
            "maximize_primary_metric": maximize_metric,
            "global_update_step": (epoch + 1) * num_training_steps_per_epoch,
        }
        if args.output_dir and args.save_ckpt and is_better:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch="best",
                model_ema=model_ema, training_state=checkpoint_training_state,
            )
        if args.output_dir and args.save_ckpt and (
            (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs
        ):
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                model_ema=model_ema, training_state=checkpoint_training_state,
            )

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats, default=str) + "\n")
            if args.save_history:
                save_json_on_master(history, os.path.join(args.output_dir, "history.json"))

        if args.model_ema and args.model_ema_eval and data_loader_val is not None:
            ema_stats = evaluate(
                data_loader_val,
                model_ema.ema,
                device,
                use_amp=args.use_amp,
                task_type=args.task_type,
                criterion=eval_criterion,
            )
            print(f"EMA validation: {format_primary(ema_stats, args.task_type)}")

    total_time = time.time() - start_time
    print(f"Training time {str(datetime.timedelta(seconds=int(total_time)))}")

    convergence_summary = {
        f"best_val_{metric_name}": best_val_metric,
        "primary_metric": metric_name,
        "maximize_primary_metric": maximize_metric,
        "best_epoch": best_epoch,
        "total_train_time_sec": total_time,
        "epochs": args.epochs,
        "n_trainable_parameters": n_trainable,
        "n_total_parameters": n_total,
        "proposal_calibration_time_sec": float(proposal_calibration_time_sec),
        "setup_and_calibration_time_sec": float(max(0.0, training_perf_start - run_wall_start)),
        "training_gpu_hours": float(total_time * (max(1, utils.get_world_size()) if device.type == "cuda" else 0) / 3600.0),
        "total_train_samples_seen": int(len(dataset_train) * max(0, args.epochs - args.start_epoch)),
        "effective_training_samples_per_second": float(
            len(dataset_train) * max(0, args.epochs - args.start_epoch) / max(total_time, 1e-12)
        ),
    }
    if history and best_epoch >= 0:
        if maximize_metric and best_val_metric > 0:
            target = 0.95 * best_val_metric
            epochs_to_target = next(
                (int(row["epoch"]) + 1 for row in history if row.get(f"val_{metric_name}", float("-inf")) >= target),
                None,
            )
            convergence_summary["epochs_to_95pct_best"] = epochs_to_target
        elif not maximize_metric and np.isfinite(best_val_metric):
            # For an error metric, reaching within 5% of the best error is the analogous threshold.
            target = 1.05 * best_val_metric
            epochs_to_target = next(
                (int(row["epoch"]) + 1 for row in history if row.get(f"val_{metric_name}", float("inf")) <= target),
                None,
            )
            convergence_summary["epochs_to_within_5pct_best"] = epochs_to_target

        convergence_summary["mean_epoch_time_sec"] = float(
            np.mean([row.get("train_epoch_time", 0.0) for row in history])
        )
        convergence_summary["median_epoch_time_sec"] = float(
            np.median([row.get("train_epoch_time", 0.0) for row in history])
        )
        throughput_values = [
            row.get("train_train_samples_per_second") for row in history
            if row.get("train_train_samples_per_second") is not None
        ]
        if throughput_values:
            convergence_summary["mean_train_samples_per_second"] = float(np.mean(throughput_values))
            convergence_summary["median_train_samples_per_second"] = float(np.median(throughput_values))
        if best_epoch >= 0:
            convergence_summary["time_to_best_sec"] = float(sum(
                row.get("train_epoch_time", 0.0) for row in history if int(row.get("epoch", -1)) <= best_epoch
            ))
        mem_values = [
            row.get("train_peak_train_memory_mb")
            for row in history
            if row.get("train_peak_train_memory_mb") is not None
        ]
        if mem_values:
            convergence_summary["peak_train_memory_mb"] = float(max(mem_values))
    if args.output_dir:
        save_json_on_master(convergence_summary, os.path.join(args.output_dir, "convergence_summary.json"))

    # Final test: restore the best validation checkpoint and evaluate once.
    if data_loader_test is not None:
        best_ckpt = os.path.join(args.output_dir, "checkpoint-best.pth") if args.output_dir else ""
        if not best_ckpt or not os.path.exists(best_ckpt):
            raise FileNotFoundError(
                "Final test requires checkpoint-best.pth produced by validation selection."
            )
        ckpt = safe_torch_load(best_ckpt, map_location="cpu")
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise RuntimeError(f"Best checkpoint is malformed: {best_ckpt}")
        model_without_ddp.load_state_dict(ckpt["model"], strict=True)
        if args.tuning_method == "trso":
            from models.tuning_modules.mdl_tangent_core import (
                merge_mdl_tangent_cores_, set_mdl_tangent_trainability,
            )
            set_mdl_tangent_trainability(model_without_ddp)
            if args.trso_fast_inference:
                merged = merge_mdl_tangent_cores_(model_without_ddp)
                print(f"[G-CREST-TRSO] exactly merged {merged} updates; runtime adapters=0")
                if args.profile_efficiency:
                    from tools.profile_efficiency import profile_model, save_profile
                    profile = profile_model(
                        model=model_without_ddp,
                        device=device,
                        input_size=args.input_size,
                        batch_size=min(args.profile_batch_size, args.batch_size),
                        use_amp=args.use_amp,
                        task_type=args.task_type,
                    )
                    profile["deployment_state"] = "exactly_merged"
                    if args.output_dir:
                        save_profile(profile, os.path.join(args.output_dir, "efficiency_profile.json"))
        if args.tuning_method in {"fact_tt", "fact_tk"}:
            from models.tuning_modules.fact import merge_fact_
            merged = merge_fact_(model_without_ddp)
            print(f"[FacT] exactly merged {merged} factorized weight updates.")
        if args.tuning_method == "repadapter" and args.repadapter_merge:
            from models.tuning_modules.repadapter import merge_repadapter_
            merged = merge_repadapter_(model_without_ddp)
            print(f"[RepAdapter] exactly folded {merged} adapter branches into ViT projections; runtime adapters=0.")
        if args.tuning_method == "arc" and args.arc_merge:
            from models.tuning_modules.arc import merge_arc_
            model_without_ddp.eval()
            merged = merge_arc_(model_without_ddp)
            print(f"[ARC] exactly folded {merged} re-composed branches into ViT projections; runtime ARC branches=0.")
        if args.tuning_method == "spt_lora":
            from models.tuning_modules.spt import merge_spt_
            merged = merge_spt_(model_without_ddp)
            print(f"[SPT-LoRA] exactly merged {merged} sparse/low-rank updates.")
        final_evaluation_started = time.perf_counter()
        test_stats = evaluate(
            data_loader_test,
            model,
            device,
            use_amp=args.use_amp,
            measure_latency=args.measure_eval_latency,
            task_type=args.task_type,
            criterion=eval_criterion,
        )
        final_evaluation_time_sec = time.perf_counter() - final_evaluation_started
        print(f"Final test on {len(dataset_test)} samples: {format_primary(test_stats, args.task_type)}")
        if args.output_dir:
            save_json_on_master(test_stats, os.path.join(args.output_dir, "test_summary.json"))
    total_wall_time_sec = time.perf_counter() - run_wall_start
    if args.output_dir and bool(getattr(args, "report_wall_clock", True)):
        accelerator_count = max(1, utils.get_world_size()) if device.type == "cuda" else 0
        timing_report = {
            "task_type": args.task_type,
            "dataset": args.dataset,
            "backbone": args.backbone,
            "method": args.tuning_method,
            "proposal_calibration_time_sec": float(proposal_calibration_time_sec),
            "profiling_time_sec": float(profiling_time_sec),
            "setup_and_calibration_time_sec": float(max(0.0, training_perf_start - run_wall_start)),
            "training_time_sec": float(total_time),
            "mean_epoch_time_sec": float(convergence_summary.get("mean_epoch_time_sec", 0.0)),
            "median_epoch_time_sec": float(convergence_summary.get("median_epoch_time_sec", 0.0)),
            "mean_train_samples_per_second": float(convergence_summary.get("mean_train_samples_per_second", 0.0)),
            "median_train_samples_per_second": float(convergence_summary.get("median_train_samples_per_second", 0.0)),
            "effective_training_samples_per_second": float(convergence_summary.get("effective_training_samples_per_second", 0.0)),
            "time_to_best_sec": float(convergence_summary.get("time_to_best_sec", 0.0)),
            "peak_train_memory_mb": float(convergence_summary.get("peak_train_memory_mb", 0.0)),
            "final_evaluation_time_sec": float(final_evaluation_time_sec),
            "total_wall_time_sec": float(total_wall_time_sec),
            "accelerator_type": str(device.type),
            "accelerator_count": int(accelerator_count),
            "training_gpu_hours": float(total_time * accelerator_count / 3600.0),
            "gpu_hours": float(total_wall_time_sec * accelerator_count / 3600.0),
            "train_samples": int(len(dataset_train)),
            "epochs_completed": int(max(0, args.epochs - args.start_epoch)),
            "total_train_samples_seen": int(len(dataset_train) * max(0, args.epochs - args.start_epoch)),
            "wall_clock_samples_per_second": float(
                len(dataset_train) * max(0, args.epochs - args.start_epoch) / max(total_wall_time_sec, 1e-12)
            ),
        }
        save_json_on_master(timing_report, os.path.join(args.output_dir, "timing_summary.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TRSO training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    args = canonicalize_args(args)

    if args.list_backbones or args.list_compatibility:
        main(args)
        raise SystemExit(0)

    # The original table can overwrite explicit CLI choices. It is now opt-in.
    if args.legacy_auto_hparams and not args.is_tuning and args.tuning_method not in ("full", "linear"):
        try:
            args = utils.auto_load_optim_param(args, args.model, args.tuning_method, args.dataset)
            args = canonicalize_args(args)
        except Exception as exc:
            print(f"[Warn] legacy auto_load_optim_param failed or unavailable: {exc}")
    elif args.is_tuning:
        args.save_ckpt = False

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.data = Path(args.data_path).name
    main(args)
