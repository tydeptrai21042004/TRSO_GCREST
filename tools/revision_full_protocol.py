"""Canonical main-table protocol for the reviewer revision.

This protocol does not overwrite the submitted 46-run reproduction.  It
reruns every manuscript setting with seeds 0/1/2 and only the paper baselines
approved for that architecture/task, plus Full/Linear reference controls and
G-CREST-TRSO.

Expected training runs: 204.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RevisionSetting:
    key: str
    dataset: str
    task: str
    backbone: str
    methods: tuple[str, ...]
    batch_size: int
    dataset_args: dict

    @property
    def expected_runs(self) -> int:
        return len(self.methods) * 3


VIT_METHODS = (
    "full", "linear", "ssf", "adaptformer", "repadapter", "arc",
    "vpt_shallow", "vpt_deep", "convpass", "fact_tt", "fact_tk", "vqt",
    "spt_lora", "spt_adapter", "trso",
)

SETTINGS: tuple[RevisionSetting, ...] = (
    RevisionSetting(
        "dtd_resnet50", "dtd", "single_label", "resnet50@torchvision",
        ("full", "linear", "prompt", "conv", "piggyback", "trso"), 16,
        {"dtd_partition": 1},
    ),
    RevisionSetting(
        "dtd_vit_b16", "dtd", "single_label", "vit_b_16@torchvision",
        VIT_METHODS, 8, {"dtd_partition": 1},
    ),
    RevisionSetting(
        "dtd_resnet18", "dtd", "single_label", "resnet18@torchvision",
        ("full", "linear", "prompt", "trso"), 32, {"dtd_partition": 1},
    ),
    RevisionSetting(
        "dtd_swin_t", "dtd", "single_label", "swin_t@torchvision",
        ("full", "linear", "ssf", "trso"), 32, {"dtd_partition": 1},
    ),
    RevisionSetting(
        "flowers_resnet18", "flowers102", "single_label", "resnet18@torchvision",
        ("full", "linear", "prompt", "trso"), 32, {},
    ),
    RevisionSetting(
        "flowers_vit_b16", "flowers102", "single_label", "vit_b_16@torchvision",
        VIT_METHODS, 16, {},
    ),
    RevisionSetting(
        "flowers_swin_t", "flowers102", "single_label", "swin_t@torchvision",
        ("full", "linear", "ssf", "trso"), 32, {},
    ),
    RevisionSetting(
        "pet_resnet18", "oxfordiiitpet", "single_label", "resnet18@torchvision",
        ("full", "linear", "prompt", "trso"), 32, {},
    ),
    RevisionSetting(
        "pet_swin_t", "oxfordiiitpet", "single_label", "swin_t@torchvision",
        ("full", "linear", "ssf", "trso"), 32, {},
    ),
    RevisionSetting(
        "voc2007_mobilenetv3_small", "voc2007", "multilabel", "mobilenet_v3_small@torchvision",
        ("full", "linear", "ml_decoder", "trso"), 32, {},
    ),
    RevisionSetting(
        "pet_segmentation_lraspp_mobilenetv3_large", "oxford_pet_segmentation",
        "semantic_segmentation", "lraspp_mobilenet_v3_large@torchvision",
        ("full", "linear", "segadapter", "trso"), 8,
        {"segmentation_num_classes": 3, "segmentation_ignore_index": 255},
    ),
)

EXPECTED_MAIN_RUNS = sum(item.expected_runs for item in SETTINGS)
assert EXPECTED_MAIN_RUNS == 204, EXPECTED_MAIN_RUNS


def command_for(setting: RevisionSetting, *, output_root: str, data_path: str, download: str) -> list[str]:
    backbone, source = setting.backbone.rsplit("@", 1)
    manifest = str(Path("experiments") / "revision_main" / f"{setting.key}.json")
    cmd = [
        sys.executable, "-m", "tools.run_fair_suite",
        "--dataset", setting.dataset,
        "--task", setting.task,
        "--data_path", data_path,
        "--download", download,
        "--backbones", f"{backbone}@{source}",
        "--methods", ",".join(setting.methods),
        "--seeds", "0,1,2",
        "--split_seed", "0",
        "--epochs", "30",
        "--batch_size", str(setting.batch_size),
        "--input_size", "224",
        "--optimizer", "adamw",
        "--peft_lr", "1e-3",
        "--full_lr", "1e-4",
        "--linear_lr", "1e-3",
        "--weight_decay", "1e-4",
        "--warmup_epochs", "3",
        "--min_lr", "1e-6",
        "--augmentation", "strong",
        "--head_init_policy", "random",
        "--peft_head_lr_scale", "1.0",
        "--dataset_args_json", json.dumps(setting.dataset_args, separators=(",", ":")),
        "--output_root", str(Path(output_root) / setting.key),
        "--manifest", manifest,
    ]
    return cmd


def write_audit_files(path: Path) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "reviewer_revision_main_v1",
        "seeds": [0, 1, 2],
        "split_seed": 0,
        "epochs": 30,
        "warmup_epochs": 3,
        "optimizer": "adamw",
        "full_lr": 1e-4,
        "linear_lr": 1e-3,
        "peft_lr": 1e-3,
        "weight_decay": 1e-4,
        "min_lr": 1e-6,
        "input_size": 224,
        "head_init_policy": "random",
        "peft_head_lr_scale": 1.0,
        "method_internal_defaults": "baseline_recipes.py",
        "expected_training_runs": EXPECTED_MAIN_RUNS,
        "settings": [asdict(item) | {"expected_runs": item.expected_runs} for item in SETTINGS],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "key", "dataset", "task", "backbone", "methods", "batch_size", "expected_runs", "dataset_args"
        ])
        writer.writeheader()
        for item in SETTINGS:
            writer.writerow({
                "key": item.key, "dataset": item.dataset, "task": item.task,
                "backbone": item.backbone, "methods": ",".join(item.methods),
                "batch_size": item.batch_size, "expected_runs": item.expected_runs,
                "dataset_args": json.dumps(item.dataset_args, sort_keys=True),
            })
    return path, csv_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", default="all", help="Setting key or 'all'.")
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--download", default="auto")
    parser.add_argument("--output_root", default="outputs_revision_main")
    parser.add_argument("--audit", default="experiments/revision_full_protocol.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    audit_json, audit_csv = write_audit_files(Path(args.audit))
    selected = SETTINGS if args.setting == "all" else tuple(x for x in SETTINGS if x.key == args.setting)
    if not selected:
        raise ValueError(f"Unknown --setting {args.setting!r}; choose one of {[x.key for x in SETTINGS]}")

    print(f"Revision main protocol: {EXPECTED_MAIN_RUNS} total training runs")
    print(f"Audit: {audit_json} / {audit_csv}")
    for setting in selected:
        cmd = command_for(setting, output_root=args.output_root, data_path=args.data_path, download=args.download)
        print(" ".join(cmd))
        if args.execute:
            subprocess.run([*cmd, "--execute"], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
