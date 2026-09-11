"""Build and optionally execute the G-CREST-TRSO structural ablation suite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from models.tuning_modules.mdl_tangent_core import TRSO_ABLATIONS
from tools.experiment_grid import build_specs, execute_specs, parse_csv_values, write_manifest


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed structural ablations of Cross-Fitted Tangent-Core G-CREST-TRSO."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--download", type=str2bool, default=False)
    parser.add_argument("--dataset_args_json", default="{}")
    parser.add_argument("--task", default="auto")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--model_source", default="auto")
    parser.add_argument("--weights", default="DEFAULT")
    parser.add_argument("--pretrained", type=str2bool, default=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--ablations", default=",".join(TRSO_ABLATIONS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--augmentation", default="basic", choices=["basic", "strong"])
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--head_from", default="")
    parser.add_argument("--fair_protocol", type=str2bool, default=True)
    parser.add_argument("--peft_head_lr_scale", type=float, default=0.5)
    parser.add_argument("--peft_freeze_head", type=str2bool, default=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_root", default="outputs_trso_ablation")
    parser.add_argument("--manifest", default="experiments/trso_ablation_manifest.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max_runs", type=int, default=0)
    parser.add_argument("--profile_efficiency", type=str2bool, default=True)
    parser.add_argument("--final_test", type=str2bool, default=True)
    parser.add_argument("--allow_val_as_test", type=str2bool, default=False)
    parser.add_argument("--mode_count_rule", default="geometric", choices=["geometric", "shannon", "arithmetic", "harmonic"])
    parser.add_argument("--r_scale", type=float, default=1.0)
    parser.add_argument("--fixed_r", type=int, default=0)
    parser.add_argument("--calibration_fraction", type=float, default=1.0)
    parser.add_argument("--calibration_max_batches", type=int, default=0)
    parser.add_argument("--partition_mode", default="alternating", choices=["alternating", "seeded_random"])
    parser.add_argument("--partition_seed", type=int, default=0)
    parser.add_argument("--svd_oversampling", type=int, default=0)
    parser.add_argument("--svd_power_iterations", type=int, default=2)
    parser.add_argument("--svd_seed", type=int, default=0)
    return parser


def build_ablation_specs(args: argparse.Namespace):
    seeds = parse_csv_values(args.seeds, int)
    requested = parse_csv_values(args.ablations, str)
    unknown = sorted(set(requested) - set(TRSO_ABLATIONS))
    if unknown:
        raise ValueError(f"Unknown TRSO ablations: {unknown}; valid={TRSO_ABLATIONS}")
    dataset_args = json.loads(getattr(args, "dataset_args_json", "{}"))
    if not isinstance(dataset_args, dict):
        raise ValueError("--dataset_args_json must decode to a JSON object")
    variants: list[tuple[str, dict[str, Any]]] = []
    for ablation in requested:
        for seed in seeds:
            variants.append((ablation, {"trso_ablation": ablation, "seed": seed}))
    common = {
        "dataset": args.dataset,
        "data_path": args.data_path,
        "download": getattr(args, "download", False),
        "task": args.task,
        "backbone": args.backbone,
        "model_source": args.model_source,
        "weights": args.weights,
        "pretrained": args.pretrained,
        "tuning_method": "trso",
        "trso_fast_inference": True,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "input_size": args.input_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "min_lr": args.min_lr,
        "split_seed": getattr(args, "split_seed", 0),
        "head_from": getattr(args, "head_from", ""),
        "fair_protocol": getattr(args, "fair_protocol", True),
        "fair_optimizer": args.optimizer,
        "fair_peft_lr": args.lr,
        "fair_full_lr": args.lr,
        "fair_linear_lr": args.lr,
        "fair_weight_decay": args.weight_decay,
        "fair_warmup_epochs": min(int(args.warmup_epochs), max(0, int(args.epochs) - 1)),
        "fair_min_lr": args.min_lr,
        "train_aug": "standard",
        "aa": "rand-m9-mstd0.5-inc1" if getattr(args, "augmentation", "basic") == "strong" and args.task in {"auto", "single_label"} else "none",
        "color_jitter": 0.2 if getattr(args, "augmentation", "basic") == "strong" and args.task in {"auto", "single_label"} else 0.0,
        "mixup": 0.2 if getattr(args, "augmentation", "basic") == "strong" and args.task in {"auto", "single_label"} else 0.0,
        "cutmix": 0.0,
        "smoothing": 0.1 if getattr(args, "augmentation", "basic") == "strong" and args.task in {"auto", "single_label"} else 0.0,
        "reprob": 0.1 if getattr(args, "augmentation", "basic") == "strong" and args.task in {"auto", "single_label"} else 0.0,
        "keep_pretrained_head": False,
        "clip_grad": 1.0,
        "no_decay_bias_norm": True,
        "use_amp": args.device != "cpu",
        "save_ckpt": True,
        "save_ckpt_freq": 1,
        "save_history": True,
        "evaluate_before_training": True,
        "measure_eval_latency": True,
        "peft_head_lr_scale": getattr(args, "peft_head_lr_scale", 0.5),
        "peft_freeze_head": getattr(args, "peft_freeze_head", False),
        "paper_hparams": False,
        "legacy_auto_hparams": False,
        "device": args.device,
        "profile_efficiency": args.profile_efficiency,
        "final_test": args.final_test,
        "allow_val_as_test": args.allow_val_as_test,
        "trso_mode_count_rule": getattr(args, "mode_count_rule", "geometric"),
        "trso_r_scale": getattr(args, "r_scale", 1.0),
        "trso_fixed_r": getattr(args, "fixed_r", 0),
        "trso_calibration_fraction": getattr(args, "calibration_fraction", 1.0),
        "trso_calibration_max_batches": getattr(args, "calibration_max_batches", 0),
        "trso_partition_mode": getattr(args, "partition_mode", "alternating"),
        "trso_partition_seed": getattr(args, "partition_seed", 0),
        "trso_svd_oversampling": getattr(args, "svd_oversampling", 0),
        "trso_svd_power_iterations": getattr(args, "svd_power_iterations", 2),
        "trso_svd_seed": getattr(args, "svd_seed", 0),
    }
    common.update(dataset_args)
    return build_specs(
        suite="trso_tangent_core_ablation_v6",
        variants=variants,
        common=common,
        output_root=args.output_root,
    )


def main(argv=None) -> int:
    args = get_parser().parse_args(argv)
    specs = build_ablation_specs(args)
    manifest_json, manifest_csv = write_manifest(specs, args.manifest)
    protocol = {
        "suite": "trso_tangent_core_ablation_v6",
        "explicit_tangent_core_parameters": True,
        "zero_added_parameter_required": False,
        "task_head_reported_separately": True,
        "ablations": list(dict.fromkeys(spec.name for spec in specs)),
        "seeds": sorted({int(spec.parameters["seed"]) for spec in specs}),
        "runs": len(specs),
        "manifest_json": str(manifest_json),
        "manifest_csv": str(manifest_csv),
    }
    protocol_path = manifest_json.with_name(manifest_json.stem + "_protocol.json")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2))
    execute_specs(specs, execute=args.execute, max_runs=args.max_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
