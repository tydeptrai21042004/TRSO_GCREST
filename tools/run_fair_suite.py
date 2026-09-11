"""Dataset/task/backbone-aware controlled comparison runner.

The runner is intentionally capability driven:
- all task-compatible methods use one controlled outer recipe;
- each literature baseline keeps source-audited method-internal defaults;
- the default comparison starts every downstream task head from the same fresh
  initialization policy (no hidden linear-probe warm start);
- optional linear-probe warm start is an explicit ablation only;
- full fine-tuning and linear probing may use separate learning rates;
- unsupported method/backbone/task combinations are written to an explicit
  compatibility report instead of disappearing from the result table.

For paper/official-recipe paired comparisons, use
``python -m tools.run_paper_fair_pairs``.

Examples
--------
Single-label classification on two backbones::

    python -m tools.run_fair_suite \
      --dataset dtd --data_path ./data --download True \
      --backbones resnet50@torchvision,vit_tiny_patch16_224@timm \
      --seeds 0,1,2 --execute

Multi-label CSV data::

    python -m tools.run_fair_suite \
      --dataset csv --task multilabel --data_path ./dataset \
      --dataset_args_json '{"csv_train":"train.csv","csv_val":"val.csv",\
                            "csv_test":"test.csv","csv_target_columns":"a,b,c"}' \
      --backbones resnet50@torchvision,swin_t@torchvision
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_recipes import paper_method_defaults, recipe_for
from datasets.build import available_datasets
from datasets.download import parse_download_mode
from datasets.registry import get_dataset_spec
from task_registry import (
    TASK_DEPTH_ESTIMATION, TASK_MULTILABEL, TASK_OBJECT_DETECTION,
    TASK_REGRESSION, TASK_SEMANTIC_SEGMENTATION, TASK_SINGLE_LABEL,
    normalize_task, task_choices,
)
from models.model_support import (
    METHOD_SUPPORT, STRICT_AUTO_METHODS, REFERENCE_CONTROL_METHODS, PROPOSAL_METHODS,
    canonical_method, infer_backbone_family_name, method_category,
    static_method_compatibility,
)
from tools.experiment_grid import (
    RunSpec,
    build_specs,
    execute_specs,
    execute_specs_parallel,
    parse_csv_values,
    write_manifest,
)

DEFAULT_METHODS = tuple(STRICT_AUTO_METHODS)

SINGLE_LABEL_DATASETS = {
    "cifar10", "cifar100", "mnist", "fashion_mnist", "emnist", "kmnist",
    "qmnist", "usps", "svhn", "stl10", "food101", "oxfordiiitpet",
    "flowers102", "stanford_cars", "caltech101", "dtd", "eurosat",
    "fgvc_aircraft", "sun397", "gtsrb", "fer2013", "pcam", "country211",
    "rendered_sst2", "places365", "inaturalist", "imagefolder", "cub200",
    "nabirds", "stanford_dogs", "vtab", "fewshot",
    "pathmnist", "dermamnist", "bloodmnist", "pneumoniamnist", "organamnist", "tissuemnist",
}

SEGMENTATION_DATASETS = {
    "voc2012_segmentation", "oxford_pet_segmentation", "sbd_segmentation",
    "cityscapes_segmentation", "segmentation_folder", "fake_segmentation",
}
DEPTH_DATASETS = {"depth_folder", "fake_depth"}
DETECTION_DATASETS = {"voc_detection", "coco_detection", "fake_detection"}

TASK_BACKBONE_PRESETS = {
    TASK_SINGLE_LABEL: "resnet50@torchvision,convnext_tiny@torchvision,vit_tiny_patch16_224@timm,swin_t@torchvision",
    TASK_MULTILABEL: "resnet50@torchvision,convnext_tiny@torchvision,vit_tiny_patch16_224@timm,swin_t@torchvision",
    TASK_REGRESSION: "resnet50@torchvision,convnext_tiny@torchvision,vit_tiny_patch16_224@timm",
    TASK_SEMANTIC_SEGMENTATION: "fcn_resnet50@torchvision,deeplabv3_resnet50@torchvision,lraspp_mobilenet_v3_large@torchvision",
    TASK_DEPTH_ESTIMATION: "fcn_resnet50@torchvision,deeplabv3_resnet50@torchvision,lraspp_mobilenet_v3_large@torchvision",
    TASK_OBJECT_DETECTION: "fasterrcnn_resnet50_fpn_v2@torchvision,fasterrcnn_mobilenet_v3_large_fpn@torchvision,retinanet_resnet50_fpn_v2@torchvision",
}


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="dtd", choices=available_datasets())
    p.add_argument("--task", default="auto", choices=task_choices(include_auto=True))
    p.add_argument("--data_path", default="./data")
    p.add_argument("--download", type=parse_download_mode, default="no")
    p.add_argument("--weights", default="DEFAULT", help="Pretrained weight identifier; use 'none' for offline smoke tests.")
    p.add_argument("--pretrained", type=str2bool, default=None)
    p.add_argument("--dataset_args_json", default="{}", help="Extra dataset CLI arguments as a JSON object.")
    p.add_argument("--output_root", default="outputs_fair")
    p.add_argument("--manifest", default="experiments/fair_manifest.json")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--split_seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--input_size", type=int, default=0, help="0 resolves native pretrained size before dataset transforms.")
    p.add_argument(
        "--backbones",
        default="auto",
        help="Comma-separated backbone@source entries, or 'auto' for task-appropriate presets.",
    )
    p.add_argument(
        "--methods", default="auto",
        help=("Comma-separated methods, or one of: auto/paper_baselines (strict literature baselines only), "
              "reference_controls (full,linear), proposal (TRSO full only).")
    )
    p.add_argument(
        "--external_head_manifests", default="",
        help=("Comma-separated manifests containing completed linear-probe rows. "
              "Used only when --head_init_policy=linear_probe."),
    )
    p.add_argument(
        "--head_init_policy", default="random", choices=["random", "linear_probe"],
        help=("Main-table default is random: every method starts with its normal fresh task head. "
              "linear_probe is an explicit warm-start ablation and may reuse --external_head_manifests."),
    )
    p.add_argument("--trso_ablation", default="full", choices=["full", "diagonal_only", "no_sampling_variance", "no_crossfit", "head_only"], help="TRSO structural ablation; full is used for benchmark comparisons.")
    p.add_argument(
        "--allow_nonpaper_controls", type=str2bool, default=False,
        help="Allow separately labeled engineering/transferred controls; never enabled by auto.",
    )
    p.add_argument(
        "--allow_paper_ablations", type=str2bool, default=False,
        help="Allow paper-internal ablations in a separately labeled study; never enabled by auto.",
    )
    p.add_argument(
        "--allow_unverified_paper_reimplementations", type=str2bool, default=False,
        help="Allow qualified FacT/VQT/SPT reproduction candidates; never enabled by auto.",
    )
    p.add_argument("--peft_lr", type=float, default=1e-3)
    p.add_argument("--full_lr", type=float, default=1e-4)
    p.add_argument("--linear_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--augmentation", default="strong", choices=["basic", "strong"])
    p.add_argument("--peft_head_lr_scale", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu_ids", default="0")
    p.add_argument("--parallel_runs", type=int, default=1)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--max_runs", type=int, default=0)
    p.add_argument("--profile_efficiency", type=str2bool, default=True)
    p.add_argument("--measure_eval_latency", type=str2bool, default=True)
    p.add_argument("--peft_freeze_head", type=str2bool, default=False, help="Initialize from the same best linear head, then jointly adapt it for every compatible PEFT method.")
    p.add_argument("--allow_val_as_test", type=str2bool, default=False)
    return p


def resolve_task(args: argparse.Namespace, dataset_args: dict[str, Any]) -> str:
    requested = normalize_task(args.task)
    if requested != "auto":
        return requested
    dataset = args.dataset
    if dataset in SINGLE_LABEL_DATASETS:
        return "single_label"
    if dataset == "coco":
        return "single_label" if str(dataset_args.get("coco_task", "multilabel")).lower() == "majority" else "multilabel"
    if dataset == "voc2007":
        return "multilabel"
    if dataset == "celeba":
        return TASK_REGRESSION if str(dataset_args.get("celeba_task", "attributes")).lower() == "landmarks" else TASK_MULTILABEL
    if dataset in SEGMENTATION_DATASETS:
        return TASK_SEMANTIC_SEGMENTATION
    if dataset in DEPTH_DATASETS:
        return TASK_DEPTH_ESTIMATION
    if dataset in DETECTION_DATASETS:
        return TASK_OBJECT_DETECTION
    if dataset == "fake":
        return TASK_SINGLE_LABEL
    if dataset == "csv":
        raise ValueError("--task must be explicit for --dataset csv.")
    raise ValueError(f"Could not infer task for dataset {dataset!r}; pass --task explicitly.")


def parse_backbones(text: str, task: str | None = None) -> list[tuple[str, str]]:
    if str(text).strip().lower() == "auto":
        resolved_task = normalize_task(task or TASK_SINGLE_LABEL)
        text = TASK_BACKBONE_PRESETS[resolved_task]
    rows: list[tuple[str, str]] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "@" in token:
            backbone, source = token.rsplit("@", 1)
        else:
            backbone, source = token, "auto"
        rows.append((backbone.strip(), source.strip()))
    if not rows:
        raise ValueError("At least one backbone is required.")
    return rows


def parse_methods(text: str) -> list[str]:
    selector = str(text).strip().lower()
    if selector in {"auto", "paper_baselines", "literature_baselines"}:
        # ``adapter`` is a legacy alias of Conv-Adapter, not a distinct paper row.
        return [method for method in DEFAULT_METHODS if method != "adapter"]
    if selector in {"reference_controls", "references"}:
        return ["full", "linear"]
    if selector in {"proposal", "trso"}:
        return ["trso"]
    methods = [canonical_method(item) for item in str(text).split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHOD_SUPPORT]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {sorted(METHOD_SUPPORT)}")
    # Conv-Adapter's legacy alias must not appear as a duplicate paper row.
    unique: list[str] = []
    for method in methods:
        method = "conv" if method == "adapter" else method
        if method not in unique:
            unique.append(method)
    return unique


def base_common(args: argparse.Namespace, task: str, dataset_args: dict[str, Any]) -> dict[str, Any]:
    strong_single_label = getattr(args, "augmentation", "strong") == "strong" and task == "single_label"
    common = {
        "dataset": args.dataset,
        "task": task,
        "data_path": args.data_path,
        "download": args.download,
        "weights": getattr(args, "weights", "DEFAULT"),
        "pretrained": getattr(args, "pretrained", None),
        "input_size": args.input_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": args.device,
        "use_amp": True,
        "profile_efficiency": args.profile_efficiency,
        "measure_eval_latency": args.measure_eval_latency,
        "save_ckpt": True,
        "save_ckpt_freq": 1,
        "save_history": True,
        "evaluate_before_training": True,
        "final_test": True,
        "auto_resume": False,
        "fair_protocol": True,
        "fair_optimizer": args.optimizer,
        "fair_peft_lr": args.peft_lr,
        "fair_full_lr": args.full_lr,
        "fair_linear_lr": args.linear_lr,
        "fair_weight_decay": args.weight_decay,
        "fair_warmup_epochs": min(int(args.warmup_epochs), max(0, int(args.epochs) - 1)),
        "fair_min_lr": args.min_lr,
        "paper_hparams": False,
        "legacy_auto_hparams": False,
        "train_aug": "standard",
        "aa": "rand-m9-mstd0.5-inc1" if strong_single_label else "none",
        "color_jitter": 0.2 if strong_single_label else 0.0,
        "mixup": 0.2 if strong_single_label else 0.0,
        "cutmix": 0.0,
        "smoothing": 0.1 if strong_single_label else 0.0,
        "reprob": 0.1 if strong_single_label else 0.0,
        "keep_pretrained_head": False,
        "peft_freeze_head": getattr(args, "peft_freeze_head", False),
        "allow_nonpaper_controls": bool(getattr(args, "allow_nonpaper_controls", False)),
        "allow_paper_ablations": bool(getattr(args, "allow_paper_ablations", False)),
        "allow_unverified_paper_reimplementations": bool(
            getattr(args, "allow_unverified_paper_reimplementations", False)
        ),
        "peft_head_lr_scale": float(getattr(args, "peft_head_lr_scale", 1.0)),
        "protocol_name": "controlled_paper_structure_v2",
        "recipe_source": "baseline_recipes.py",
        "recipe_fidelity": "controlled_outer_paper_structure",
        "head_init_policy": str(getattr(args, "head_init_policy", "random")),
        "clip_grad": 1.0,
        "no_decay_bias_norm": True,
        "split_seed": args.split_seed,
        "allow_val_as_test": args.allow_val_as_test,
        "segmentation_ignore_index": int(dataset_args.get("segmentation_ignore_index", 255)),
        "segmentation_num_classes": int(dataset_args.get("segmentation_num_classes", 2)),
        "dense_train_aug": dataset_args.get("dense_train_aug", "scale_crop_flip"),
        "depth_loss": dataset_args.get("depth_loss", "silog_l1"),
        "detection_num_classes": int(dataset_args.get("detection_num_classes", 2)),
    }
    common.update(dataset_args)
    return common


def method_variant(
    method: str,
    family: str,
    seed: int,
    head_path: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build one method row from the canonical source-audited defaults.

    Method-internal parameters come from :mod:`baseline_recipes`; the outer
    optimization recipe remains controlled by :func:`base_common`.  A shared
    linear-probe head is attached only when the user explicitly requests the
    ``linear_probe`` head-initialization ablation.
    """
    row: dict[str, Any] = {
        "tuning_method": method,
        "seed": seed,
        "keep_pretrained_head": method == "prompt",
    }
    row.update(paper_method_defaults(method))

    head_policy = str(getattr(args, "head_init_policy", "random"))
    if (
        head_policy == "linear_probe"
        and method not in {"full", "linear", "prompt", "vqt", "ml_decoder", "segadapter"}
        and head_path
    ):
        row["head_from"] = head_path

    if method == "trso":
        row.update({"trso_fast_inference": True, "trso_ablation": getattr(args, "trso_ablation", "full")})
    elif method == "sidetune":
        row.update({"sidetune_arch": "lightweight", "sidetune_width": 64, "sidetune_depth": 4})
    elif method == "lora":
        row.update({"lora_r": 8, "lora_alpha": 16.0, "lora_dropout": 0.0})
    elif method == "bitfit":
        row.update({"bitfit_bias_scope": "all", "bitfit_train_head": True})
    elif method == "piggyback":
        row["piggyback_mask_linear"] = "vgg16" in str(getattr(args, "backbones", "")).lower()

    recipe = recipe_for(method)
    if recipe is not None:
        row.update({
            "recipe_fidelity": recipe.training_fidelity,
            "recipe_source": recipe.official_source,
        })
    return row


def load_external_linear_heads(paths_text: str) -> dict[tuple[str, str, str, str, int], str]:
    """Index completed linear-probe checkpoints from earlier reference stages."""
    indexed: dict[tuple[str, str, str, str, int], str] = {}
    for token in str(paths_text or "").split(","):
        token = token.strip()
        if not token:
            continue
        path = Path(token)
        if not path.is_file():
            raise FileNotFoundError(f"External head manifest not found: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"External head manifest must contain a list: {path}")
        for row in rows:
            params = row.get("parameters", {})
            if params.get("tuning_method") != "linear":
                continue
            key = (
                str(params.get("dataset", "")),
                str(params.get("task", "")),
                str(params.get("backbone", "")),
                str(params.get("model_source", "auto")),
                int(params.get("seed", 0)),
            )
            checkpoint = str(Path(row["output_dir"]) / "checkpoint-best.pth")
            indexed[key] = checkpoint
    return indexed


def build_suite(args: argparse.Namespace):
    dataset_args = json.loads(args.dataset_args_json)
    if not isinstance(dataset_args, dict):
        raise ValueError("--dataset_args_json must decode to a JSON object.")
    task = resolve_task(args, dataset_args)
    seeds = parse_csv_values(args.seeds, int)
    backbones = parse_backbones(args.backbones, task)
    methods = parse_methods(args.methods)
    common_base = base_common(args, task, dataset_args)
    head_policy = str(getattr(args, "head_init_policy", "random"))
    external_heads = (
        load_external_linear_heads(getattr(args, "external_head_manifests", ""))
        if head_policy == "linear_probe" else {}
    )

    head_specs: list[RunSpec] = []
    comparison_specs: list[RunSpec] = []
    compatibility: list[dict[str, Any]] = []

    for backbone, source in backbones:
        family = infer_backbone_family_name(backbone, source)
        supported_methods: list[str] = []
        for method in methods:
            ok, reason, resolved_family = static_method_compatibility(
                method,
                backbone,
                task,
                source=source,
                allow_nonpaper_controls=bool(getattr(args, "allow_nonpaper_controls", False)),
                allow_paper_ablations=bool(getattr(args, "allow_paper_ablations", False)),
                allow_unverified_paper_reimplementations=bool(
                    getattr(args, "allow_unverified_paper_reimplementations", False)
                ),
            )
            compatibility.append({
                "dataset": args.dataset,
                "task": task,
                "backbone": backbone,
                "model_source": source,
                "family": resolved_family,
                "method": method,
                "method_category": method_category(method),
                "status": "scheduled" if ok else "skipped",
                "reason": reason,
            })
            if ok:
                supported_methods.append(method)

        # Linear probing is always an independently reported reference when
        # requested.  It becomes a warm-start source only in the explicit
        # --head_init_policy=linear_probe ablation.
        linear_supported = "linear" in supported_methods
        head_path_by_seed: dict[int, str] = {}
        for seed in seeds:
            key = (args.dataset, task, backbone, source, int(seed))
            fallback_key = (args.dataset, task, backbone, "auto", int(seed))
            if key in external_heads:
                head_path_by_seed[int(seed)] = external_heads[key]
            elif fallback_key in external_heads:
                head_path_by_seed[int(seed)] = external_heads[fallback_key]
        if linear_supported:
            head_common = {
                **common_base,
                "backbone": backbone,
                "model_source": source,
                "epochs": args.epochs,
            }
            variants = [("linear", {"tuning_method": "linear", "seed": seed}) for seed in seeds]
            built = build_specs(
                suite=f"fair_{args.dataset}_{task}_{family}_{backbone}",
                variants=variants,
                common=head_common,
                output_root=args.output_root,
            )
            head_specs.extend(built)
            if head_policy == "linear_probe":
                for spec in built:
                    head_path_by_seed[int(spec.parameters["seed"])] = str(Path(spec.output_dir) / "checkpoint-best.pth")

        compare_common = {
            **common_base,
            "backbone": backbone,
            "model_source": source,
            "epochs": args.epochs,
        }
        variants: list[tuple[str, dict[str, Any]]] = []
        for seed in seeds:
            for method in supported_methods:
                if method == "linear":
                    continue
                variants.append((method, method_variant(method, family, seed, head_path_by_seed.get(seed), args)))
        comparison_specs.extend(build_specs(
            suite=f"fair_{args.dataset}_{task}_{family}_{backbone}",
            variants=variants,
            common=compare_common,
            output_root=args.output_root,
        ))

    return task, seeds, head_specs, comparison_specs, compatibility, dataset_args


def write_compatibility(rows: list[dict[str, Any]], manifest_path: str | Path) -> tuple[Path, Path]:
    manifest_path = Path(manifest_path)
    json_path = manifest_path.with_name(manifest_path.stem + "_compatibility.json")
    csv_path = json_path.with_suffix(".csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main() -> None:
    args = parser().parse_args()
    task, seeds, head_specs, comparison_specs, compatibility, dataset_args = build_suite(args)
    all_specs = [*head_specs, *comparison_specs]
    manifest, manifest_csv = write_manifest(all_specs, args.manifest)
    compatibility_json, compatibility_csv = write_compatibility(compatibility, args.manifest)
    scheduled = sum(row["status"] == "scheduled" for row in compatibility)
    skipped = len(compatibility) - scheduled
    protocol = {
        "dataset": args.dataset,
        "task": task,
        "dataset_args": dataset_args,
        "seeds": seeds,
        "backbones": parse_backbones(args.backbones, task),
        "same_peft_optimizer": args.optimizer,
        "same_peft_learning_rate": args.peft_lr,
        "same_scheduler": "cosine",
        "same_warmup_epochs": args.warmup_epochs,
        "same_weight_decay": args.weight_decay,
        "peft_head_lr_scale": args.peft_head_lr_scale,
        "augmentation": args.augmentation,
        "full_learning_rate": args.full_lr,
        "linear_learning_rate": args.linear_lr,
        "epochs": args.epochs,
        "input_size": args.input_size,
        "input_size_note": "0 means native pretrained size resolved before dataset transforms.",
        "scheduled_method_backbone_pairs": scheduled,
        "skipped_method_backbone_pairs": skipped,
        "head_init_policy": args.head_init_policy,
        "shared_head_policy": (
            "No linear-probe warm start. Every compatible baseline and TRSO starts from its normal fresh task head; Visual Prompting retains the frozen source classifier by method definition."
            if args.head_init_policy == "random"
            else "Explicit LP-warm-start ablation: compatible PEFT methods/TRSO reuse the matching best linear-probe head. This mode must not be mixed with the main random-head table."
        ),
        "external_head_manifests": ([x for x in str(args.external_head_manifests).split(",") if x] if args.head_init_policy == "linear_probe" else []),
        "method_internal_defaults": "Source-audited paper defaults from baseline_recipes.py; outer optimizer recipe is controlled.",
        "unsupported_policy": "Explicit skip report; no silent architectural approximation.",
    }
    protocol_path = Path(args.manifest).with_name(Path(args.manifest).stem + "_protocol.json")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_specs)} executable runs to {manifest} and {manifest_csv}")
    print(f"Compatibility: {compatibility_json} and {compatibility_csv}")
    print(f"Protocol: {protocol_path}")
    print(f"Scheduled pairs={scheduled}; explicit skips={skipped}")

    if args.execute:
        gpu_ids = parse_csv_values(args.gpu_ids, int)[: max(1, int(args.parallel_runs))]
        if len(gpu_ids) > 1:
            execute_specs_parallel(head_specs, execute=True, gpu_ids=gpu_ids, max_runs=0)
            execute_specs_parallel(comparison_specs, execute=True, gpu_ids=gpu_ids, max_runs=args.max_runs)
        else:
            execute_specs(head_specs, execute=True, max_runs=0)
            execute_specs(comparison_specs, execute=True, max_runs=args.max_runs)
    else:
        execute_specs(all_specs, execute=False, max_runs=args.max_runs)


if __name__ == "__main__":
    main()
