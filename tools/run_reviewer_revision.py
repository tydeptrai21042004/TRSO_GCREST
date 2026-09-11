"""Reviewer-revision experiment planner/runner for G-CREST-TRSO.

This utility adds *evaluation-only* controls requested by the reviewers while
keeping the proposed default method unchanged:

- 3+ seed reproducibility runs;
- calibration-fraction / partition sensitivity;
- alternative D0/D1 mode-count rules;
- 0.5x/1x/2x R sensitivity;
- broader reliability-component ablations;
- LoRA rank sweeps for matched-budget and oracle comparisons.

Every planned run is written to a JSON/CSV manifest before execution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiment_grid import build_specs, execute_specs, parse_csv_values, write_manifest
from datasets.download import parse_download_mode


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
    p = argparse.ArgumentParser(description="Plan or execute reviewer-requested revision experiments.")
    p.add_argument("--study", required=True, choices=[
        "multiseed", "reliability", "mode_rules", "r_scale",
        "calibration", "batch_sensitivity", "lora_rank_sweep",
    ])
    p.add_argument("--dataset", required=True)
    p.add_argument("--data_path", default="./data")
    p.add_argument("--download", type=parse_download_mode, default="no")
    p.add_argument("--dataset_args_json", default="{}")
    p.add_argument("--task", default="auto")
    p.add_argument("--backbone", required=True)
    p.add_argument("--model_source", default="auto")
    p.add_argument("--weights", default="DEFAULT")
    p.add_argument("--pretrained", type=str2bool, default=True)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--partition_seeds", default="0,1,2")
    p.add_argument("--calibration_fractions", default="0.25,0.5,1.0")
    p.add_argument("--batch_sizes", default="8,16,32,64", help="Batch sizes for calibration/batching sensitivity; proposal definition is unchanged.")
    p.add_argument("--mode_rules", default="shannon,harmonic,geometric,arithmetic")
    p.add_argument("--r_scales", default="0.5,1.0,2.0")
    p.add_argument("--lora_ranks", default="1,2,4,8,16,32,64")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--augmentation", default="strong", choices=["basic", "strong"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_root", default="outputs_reviewer_revision")
    p.add_argument("--manifest", default="experiments/reviewer_revision_manifest.json")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--max_runs", type=int, default=0)
    p.add_argument("--profile_efficiency", type=str2bool, default=True)
    p.add_argument("--final_test", type=str2bool, default=True)
    p.add_argument("--allow_val_as_test", type=str2bool, default=False)
    p.add_argument("--svd_oversampling", type=int, default=0)
    p.add_argument("--svd_power_iterations", type=int, default=2)
    return p


def _common(args: argparse.Namespace) -> dict[str, Any]:
    dataset_args = json.loads(args.dataset_args_json)
    if not isinstance(dataset_args, dict):
        raise ValueError("--dataset_args_json must decode to an object")
    strong = args.augmentation == "strong" and args.task in {"auto", "single_label"}
    common: dict[str, Any] = {
        "dataset": args.dataset,
        "data_path": args.data_path,
        "download": args.download,
        "task": args.task,
        "backbone": args.backbone,
        "model_source": args.model_source,
        "weights": args.weights,
        "pretrained": args.pretrained,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "input_size": args.input_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "min_lr": args.min_lr,
        "split_seed": args.split_seed,
        "fair_protocol": True,
        "fair_optimizer": args.optimizer,
        "fair_peft_lr": args.lr,
        "fair_full_lr": 1e-4,
        "fair_linear_lr": args.lr,
        "fair_weight_decay": args.weight_decay,
        "fair_warmup_epochs": min(args.warmup_epochs, max(0, args.epochs - 1)),
        "fair_min_lr": args.min_lr,
        "train_aug": "standard",
        "aa": "rand-m9-mstd0.5-inc1" if strong else "none",
        "color_jitter": 0.2 if strong else 0.0,
        "mixup": 0.2 if strong else 0.0,
        "cutmix": 0.0,
        "smoothing": 0.1 if strong else 0.0,
        "reprob": 0.1 if strong else 0.0,
        "keep_pretrained_head": False,
        "clip_grad": 1.0,
        "no_decay_bias_norm": True,
        "use_amp": args.device != "cpu",
        "save_ckpt": True,
        "save_history": True,
        "evaluate_before_training": True,
        "measure_eval_latency": True,
        "device": args.device,
        "profile_efficiency": args.profile_efficiency,
        "final_test": args.final_test,
        "allow_val_as_test": args.allow_val_as_test,
        "paper_hparams": False,
        "legacy_auto_hparams": False,
    }
    common.update(dataset_args)
    return common


def build_revision_specs(args: argparse.Namespace):
    seeds = parse_csv_values(args.seeds, int)
    common = _common(args)
    variants: list[tuple[str, dict[str, Any]]] = []
    study = args.study

    if study == "lora_rank_sweep":
        common.update({"tuning_method": "lora", "allow_nonpaper_controls": True})
        for rank in parse_csv_values(args.lora_ranks, int):
            for seed in seeds:
                variants.append((f"lora_r{rank}", {"lora_r": rank, "lora_alpha": max(1.0, 2.0 * rank), "seed": seed}))
    else:
        common.update({
            "tuning_method": "trso",
            "trso_fast_inference": True,
            "trso_ablation": "full",
            "trso_mode_count_rule": "geometric",
            "trso_r_scale": 1.0,
            "trso_fixed_r": 0,
            "trso_calibration_fraction": 1.0,
            "trso_calibration_max_batches": 0,
            "trso_partition_mode": "alternating",
            "trso_partition_seed": 0,
            "trso_svd_oversampling": args.svd_oversampling,
            "trso_svd_power_iterations": args.svd_power_iterations,
            "trso_svd_seed": 0,
        })
        if study == "multiseed":
            variants = [("proposal", {"seed": seed}) for seed in seeds]
        elif study == "reliability":
            for ablation in ("full", "no_crossfit", "no_sampling_variance", "diagonal_only"):
                for seed in seeds:
                    variants.append((ablation, {"trso_ablation": ablation, "seed": seed}))
        elif study == "mode_rules":
            for rule in parse_csv_values(args.mode_rules, str):
                for seed in seeds:
                    variants.append((rule, {"trso_mode_count_rule": rule, "seed": seed}))
        elif study == "r_scale":
            for scale in parse_csv_values(args.r_scales, float):
                for seed in seeds:
                    variants.append((f"rscale_{scale:g}", {"trso_r_scale": scale, "seed": seed}))
        elif study == "calibration":
            # Keep optimization seed fixed per requested seed, and vary only the
            # calibration amount/partition construction so allocation stability
            # can be separated from training randomness.
            partition_seeds = parse_csv_values(args.partition_seeds, int)
            fractions = parse_csv_values(args.calibration_fractions, float)
            for fraction in fractions:
                for seed in seeds:
                    variants.append((f"frac_{fraction:g}_alternating", {
                        "trso_calibration_fraction": fraction,
                        "trso_partition_mode": "alternating",
                        "trso_partition_seed": 0,
                        "seed": seed,
                    }))
                for partition_seed in partition_seeds:
                    # Use the first optimization seed for isolated partition sensitivity.
                    variants.append((f"frac_{fraction:g}_random_p{partition_seed}", {
                        "trso_calibration_fraction": fraction,
                        "trso_partition_mode": "seeded_random",
                        "trso_partition_seed": partition_seed,
                        "seed": seeds[0],
                    }))
        elif study == "batch_sensitivity":
            # Evaluation-only diagnostic: the submitted proposal is unchanged.
            # We vary loader batching while holding the optimization seed and
            # dataset split fixed, exposing any batching dependence explicitly.
            for batch_size in parse_csv_values(args.batch_sizes, int):
                variants.append((f"batch_{batch_size}", {
                    "batch_size": batch_size,
                    "seed": seeds[0],
                    "trso_partition_mode": "alternating",
                    "trso_partition_seed": 0,
                }))
        else:
            raise ValueError(study)

    suite = f"reviewer_{study}_{args.dataset}_{args.backbone}".replace("/", "_")
    return build_specs(
        suite=suite,
        variants=variants,
        common=common,
        output_root=args.output_root,
    )


def main(argv=None) -> int:
    args = get_parser().parse_args(argv)
    specs = build_revision_specs(args)
    manifest_json, manifest_csv = write_manifest(specs, args.manifest)
    protocol = {
        "study": args.study,
        "dataset": args.dataset,
        "backbone": args.backbone,
        "runs": len(specs),
        "proposal_default_unchanged": True,
        "default_R_rule": "ceil(sqrt(D0*D1))",
        "manifest_json": str(manifest_json),
        "manifest_csv": str(manifest_csv),
    }
    protocol_path = Path(manifest_json).with_name(Path(manifest_json).stem + "_protocol.json")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2))
    execute_specs(specs, execute=args.execute, max_runs=args.max_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
