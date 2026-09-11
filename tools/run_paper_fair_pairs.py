"""Generate paper-recipe paired baseline-vs-TRSO experiments.

The core rule is simple: for every baseline recipe/trial, the baseline and TRSO
receive the same dataset split, backbone, input resolution, batch size, epoch
budget, optimizer, learning rate, scheduler, warm-up, weight decay,
augmentation, seed, and fresh-head policy.  Only the adaptation mechanism is
allowed to differ.

This avoids two common but opposite mistakes:
1. forcing every published baseline into the proposal's optimizer recipe; and
2. giving a baseline its paper-tuned optimizer while leaving the proposal on a
   different recipe.

When an official source uses hyperparameter search (for example VPT on VTAB),
the *same search trials* are generated for both the baseline and TRSO.  Each
method can then select its own best validation trial under equal search budget.

For methods whose exact outer recipe is not encoded from a public source, this
runner never invents one.  By default it creates a clearly-labelled matched
controlled fallback.  Pass ``--require_verified_outer_recipe True`` to skip
such baselines instead.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_recipes import paper_method_defaults, paper_outer_trials, recipe_for, recipe_manifest
from datasets.build import available_datasets
from datasets.download import parse_download_mode
from models.model_support import STRICT_AUTO_METHODS, canonical_method, static_method_compatibility
from tools.experiment_grid import (
    build_specs,
    execute_specs,
    execute_specs_parallel,
    parse_csv_values,
    write_manifest,
)
from tools.run_fair_suite import parse_backbones, resolve_task, str2bool


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="dtd", choices=available_datasets())
    p.add_argument("--task", default="auto")
    p.add_argument("--data_path", default="./data")
    p.add_argument("--download", type=parse_download_mode, default="no")
    p.add_argument("--dataset_args_json", default="{}")
    p.add_argument("--backbone", default="vit_b_16@torchvision", help="Exactly one backbone@source.")
    p.add_argument("--baselines", default="auto", help="Comma-separated published baselines or auto.")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--split_seed", type=int, default=2026)
    p.add_argument("--weights", default="DEFAULT")
    p.add_argument("--pretrained", type=str2bool, default=None)
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32, help="Fallback batch size and optional paper-recipe override.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--search_mode", choices=["full", "compact"], default="full")
    p.add_argument("--use_paper_batch_size", type=str2bool, default=True, help="Use an encoded official batch size when available.")
    p.add_argument("--require_verified_outer_recipe", type=str2bool, default=False, help="Skip methods whose source-verified outer optimizer recipe is not encoded.")
    p.add_argument("--overrides_json", default="{}", help="Per-method outer-recipe overrides. Example: {\"ssf\":{\"optimizer\":\"adamw\",\"epochs\":100,\"lr\":0.001,\"weight_decay\":0.0,\"warmup_epochs\":5}}")
    p.add_argument("--method_overrides_json", default="{}", help="Per-baseline method-structure overrides from an official dataset config, e.g. {\"vpt_deep\":{\"vpt_num_tokens\":50}}.")

    # Transparent fallback used only when an exact/source-search outer recipe is
    # not encoded. It is never labelled paper exact.
    p.add_argument("--fallback_optimizer", default="adamw", choices=["adamw", "adam", "sgd"])
    p.add_argument("--fallback_epochs", type=int, default=30)
    p.add_argument("--fallback_lr", type=float, default=1e-3)
    p.add_argument("--fallback_weight_decay", type=float, default=1e-4)
    p.add_argument("--fallback_warmup_epochs", type=int, default=3)
    p.add_argument("--fallback_min_lr", type=float, default=1e-6)

    p.add_argument("--augmentation", choices=["basic", "strong"], default="strong")
    p.add_argument("--output_root", default="outputs_paper_pairs")
    p.add_argument("--manifest", default="experiments/paper_fair_pairs.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu_ids", default="0")
    p.add_argument("--parallel_runs", type=int, default=1)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--max_runs", type=int, default=0)
    p.add_argument("--profile_efficiency", type=str2bool, default=True)
    p.add_argument("--measure_eval_latency", type=str2bool, default=True)
    p.add_argument("--allow_val_as_test", type=str2bool, default=False)
    return p


def _parse_baselines(text: str) -> list[str]:
    if str(text).strip().lower() == "auto":
        return list(STRICT_AUTO_METHODS)
    out: list[str] = []
    for item in str(text).split(","):
        if not item.strip():
            continue
        method = canonical_method(item.strip())
        method = "conv" if method == "adapter" else method
        if method == "trso":
            raise ValueError("--baselines must contain literature baselines, not trso")
        if method not in out:
            out.append(method)
    return out


def _outer_fallback(args: argparse.Namespace, recipe=None) -> list[dict[str, Any]]:
    verified_optimizer = getattr(recipe, "optimizer", None) if recipe is not None else None
    return [{
        "optimizer": verified_optimizer or args.fallback_optimizer,
        "scheduler": "cosine",
        "epochs": args.fallback_epochs,
        "batch_size": args.batch_size,
        "lr": args.fallback_lr,
        "weight_decay": args.fallback_weight_decay,
        "weight_decay_adapter": args.fallback_weight_decay,
        "warmup_epochs": min(args.fallback_warmup_epochs, max(0, args.fallback_epochs - 1)),
        "min_lr": args.fallback_min_lr,
        "paper_trial_index": 0,
        "paper_search_mode": "matched_controlled_fallback",
        "paper_base_lr": args.fallback_lr,
    }]


def _normalize_override(method: str, raw: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    if raw is None:
        return []
    rows = raw if isinstance(raw, list) else [raw]
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Override for {method} must be an object or list of objects")
        required = {"optimizer", "epochs", "lr", "weight_decay"}
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Override for {method} is missing {missing}")
        epochs = int(row["epochs"])
        out.append({
            "optimizer": str(row["optimizer"]),
            "scheduler": str(row.get("scheduler", "cosine")),
            "epochs": epochs,
            "batch_size": int(row.get("batch_size", args.batch_size)),
            "lr": float(row["lr"]),
            "weight_decay": float(row["weight_decay"]),
            "weight_decay_adapter": float(row.get("weight_decay_adapter", row["weight_decay"])),
            "warmup_epochs": min(int(row.get("warmup_epochs", 0)), max(0, epochs - 1)),
            "min_lr": float(row.get("min_lr", 0.0)),
            "paper_trial_index": int(row.get("paper_trial_index", index)),
            "paper_search_mode": str(row.get("paper_search_mode", "user_source_override")),
            "paper_base_lr": float(row.get("paper_base_lr", row["lr"])),
        })
    return out


def _common(args: argparse.Namespace, task: str, dataset_args: dict[str, Any], *, backbone: str, source: str) -> dict[str, Any]:
    strong = args.augmentation == "strong" and task == "single_label"
    common: dict[str, Any] = {
        "dataset": args.dataset,
        "task": task,
        "data_path": args.data_path,
        "download": args.download,
        "backbone": backbone,
        "model_source": source,
        "weights": args.weights,
        "pretrained": args.pretrained,
        "input_size": args.input_size,
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
        "fair_protocol": False,
        "paper_hparams": False,
        "legacy_auto_hparams": False,
        "head_init_policy": "random",
        "peft_head_lr_scale": 1.0,
        "peft_freeze_head": False,
        "no_decay_bias_norm": True,
        "train_aug": "standard",
        "aa": "rand-m9-mstd0.5-inc1" if strong else "none",
        "color_jitter": 0.2 if strong else 0.0,
        "mixup": 0.2 if strong else 0.0,
        "cutmix": 0.0,
        "smoothing": 0.1 if strong else 0.0,
        "reprob": 0.1 if strong else 0.0,
        "keep_pretrained_head": False,
        "clip_grad": 1.0,
        "split_seed": args.split_seed,
        "allow_val_as_test": args.allow_val_as_test,
        "allow_nonpaper_controls": False,
        "allow_paper_ablations": False,
        "allow_unverified_paper_reimplementations": False,
        "segmentation_ignore_index": int(dataset_args.get("segmentation_ignore_index", 255)),
        "segmentation_num_classes": int(dataset_args.get("segmentation_num_classes", 2)),
        "dense_train_aug": dataset_args.get("dense_train_aug", "scale_crop_flip"),
        "depth_loss": dataset_args.get("depth_loss", "silog_l1"),
        "detection_num_classes": int(dataset_args.get("detection_num_classes", 2)),
    }
    common.update(dataset_args)
    return common


def build_pair_specs(args: argparse.Namespace):
    dataset_args = json.loads(args.dataset_args_json)
    if not isinstance(dataset_args, dict):
        raise ValueError("--dataset_args_json must decode to an object")
    overrides = json.loads(args.overrides_json)
    if not isinstance(overrides, dict):
        raise ValueError("--overrides_json must decode to an object")
    method_overrides = json.loads(args.method_overrides_json)
    if not isinstance(method_overrides, dict):
        raise ValueError("--method_overrides_json must decode to an object")
    task = resolve_task(args, dataset_args)
    parsed = parse_backbones(args.backbone, task)
    if len(parsed) != 1:
        raise ValueError("--backbone must resolve to exactly one backbone@source")
    backbone, source = parsed[0]
    seeds = parse_csv_values(args.seeds, int)
    baselines = _parse_baselines(args.baselines)
    common = _common(args, task, dataset_args, backbone=backbone, source=source)

    specs = []
    audit: list[dict[str, Any]] = []
    for baseline in baselines:
        ok, reason, family = static_method_compatibility(
            baseline, backbone, task, source=source,
            allow_nonpaper_controls=False,
            allow_paper_ablations=False,
            allow_unverified_paper_reimplementations=False,
        )
        recipe = recipe_for(baseline)
        if not ok:
            audit.append({
                "baseline": baseline, "status": "skipped_incompatible", "reason": reason,
                "family": family, "recipe": recipe.to_dict() if recipe else None,
            })
            continue
        if recipe is None:
            audit.append({"baseline": baseline, "status": "skipped_no_recipe", "reason": "No canonical recipe metadata"})
            continue

        override_trials = _normalize_override(baseline, overrides.get(baseline), args)
        batch_for_recipe = recipe.batch_size if (args.use_paper_batch_size and recipe.batch_size) else args.batch_size
        source_trials = paper_outer_trials(
            baseline, batch_size=batch_for_recipe, search_mode=args.search_mode
        )
        if override_trials:
            trials = override_trials
            fidelity = "user_source_override"
            origin = "override"
        elif source_trials:
            trials = source_trials
            if any(str(row.get("paper_search_mode")) == "compact" for row in source_trials):
                fidelity = recipe.training_fidelity + "_compact_search"
            else:
                fidelity = recipe.training_fidelity
            origin = "source_verified"
        elif args.require_verified_outer_recipe:
            audit.append({
                "baseline": baseline,
                "status": "skipped_outer_recipe_not_encoded",
                "reason": recipe.notes or recipe.training_fidelity,
                "recipe": recipe.to_dict(),
            })
            continue
        else:
            trials = _outer_fallback(args, recipe)
            fidelity = (
                "matched_source_constraints_fallback_not_paper_exact"
                if recipe.optimizer else "matched_controlled_fallback_not_paper_exact"
            )
            origin = "fallback"

        pair_count = 0
        for trial_index, outer in enumerate(trials):
            # Enforce a legal warm-up for small user overrides.
            outer = dict(outer)
            outer["warmup_epochs"] = min(int(outer.get("warmup_epochs", 0)), max(0, int(outer["epochs"]) - 1))
            for seed in seeds:
                shared = {
                    **common,
                    **outer,
                    "seed": seed,
                    "protocol_name": "paper_recipe_paired_v1",
                    "paired_baseline": baseline,
                    "recipe_source": recipe.official_source,
                    "recipe_fidelity": fidelity,
                    "head_init_policy": "random",
                    "peft_head_lr_scale": 1.0,
                }
                baseline_variant = {
                    "tuning_method": baseline,
                    **paper_method_defaults(baseline),
                    **dict(method_overrides.get(baseline, {})),
                }
                if baseline == "prompt":
                    baseline_variant["keep_pretrained_head"] = True
                if baseline == "piggyback":
                    baseline_variant["piggyback_mask_linear"] = "vgg16" in backbone.lower()
                trso_variant = {
                    "tuning_method": "trso",
                    "trso_fast_inference": True,
                    "trso_ablation": "full",
                    "keep_pretrained_head": False,
                }
                name_prefix = f"{baseline}_trial{trial_index:03d}"
                specs.extend(build_specs(
                    suite=f"paper_pair_{args.dataset}_{backbone}_{baseline}",
                    variants=[
                        (f"{name_prefix}_baseline", baseline_variant),
                        (f"{name_prefix}_trso", trso_variant),
                    ],
                    common=shared,
                    output_root=args.output_root,
                ))
                pair_count += 1
        audit.append({
            "baseline": baseline,
            "status": "scheduled",
            "family": family,
            "trial_origin": origin,
            "recipe_fidelity": fidelity,
            "outer_trials": len(trials),
            "seeds": seeds,
            "paired_seed_trial_blocks": pair_count,
            "training_runs": pair_count * 2,
            "fairness_rule": "baseline and TRSO share every outer optimization/data/head-init field per seed/trial",
            "method_override": method_overrides.get(baseline, {}),
            "recipe": recipe.to_dict(),
        })
    return task, backbone, source, seeds, specs, audit, dataset_args


def _write_audit(path: Path, *, args: argparse.Namespace, task: str, backbone: str, source: str, seeds: list[int], audit: list[dict[str, Any]], dataset_args: dict[str, Any]) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "paper_recipe_paired_v1",
        "principle": "For every baseline recipe/HPO trial, rerun TRSO with the identical outer recipe and fresh-head policy.",
        "selection_rule": "When a paper source uses HPO, select baseline and TRSO hyperparameters independently by validation metric from identical candidate grids; never select on test.",
        "head_initialization": "random/fresh for both sides; no linear-probe warm start",
        "dataset": args.dataset,
        "task": task,
        "dataset_args": dataset_args,
        "backbone": backbone,
        "model_source": source,
        "seeds": seeds,
        "search_mode": args.search_mode,
        "require_verified_outer_recipe": args.require_verified_outer_recipe,
        "runs": audit,
        "recipe_catalog": recipe_manifest(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["baseline", "status", "family", "trial_origin", "recipe_fidelity", "outer_trials", "training_runs", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in audit:
            writer.writerow({key: row.get(key, "") for key in fields})
    return path, csv_path


def main() -> int:
    args = parser().parse_args()
    task, backbone, source, seeds, specs, audit, dataset_args = build_pair_specs(args)
    manifest, manifest_csv = write_manifest(specs, args.manifest)
    audit_path = Path(args.manifest).with_name(Path(args.manifest).stem + "_audit.json")
    audit_json, audit_csv = _write_audit(
        audit_path, args=args, task=task, backbone=backbone, source=source,
        seeds=seeds, audit=audit, dataset_args=dataset_args,
    )
    print(f"Wrote {len(specs)} paired training runs: {manifest} / {manifest_csv}")
    print(f"Audit: {audit_json} / {audit_csv}")
    for row in audit:
        print(f"[{row['status']}] {row['baseline']}: {row.get('recipe_fidelity', row.get('reason', ''))}")

    if args.execute:
        gpu_ids = parse_csv_values(args.gpu_ids, int)[: max(1, int(args.parallel_runs))]
        if len(gpu_ids) > 1:
            execute_specs_parallel(specs, execute=True, gpu_ids=gpu_ids, max_runs=args.max_runs)
        else:
            execute_specs(specs, execute=True, max_runs=args.max_runs)
    else:
        execute_specs(specs, execute=False, max_runs=args.max_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
