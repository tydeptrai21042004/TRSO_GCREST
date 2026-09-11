"""Active Kaggle execution plan for the reviewer-ready 204-run main matrix.

The publication matrix contains 68 dataset/backbone/method experiments and
three seeds (0, 1, 2), hence 204 independent training runs.  Experiments are
balanced across 18 Kaggle sessions by their estimated T4 runtime.  Inside every
session the experiment groups are ordered from fastest to slowest.

Runtime estimates are planning values, not measured claims.  The session runner
rescales them for the detected GPU and continuously reports actual elapsed time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Iterable

SEEDS: tuple[int, ...] = (0, 1, 2)
SESSION_COUNT = 18
REFERENCE_GPU = "NVIDIA T4"
EPOCHS = 30
WARMUP_EPOCHS = 3
INPUT_SIZE = 224
SPLIT_SEED = 0

# Hard safety policy requested for Kaggle 12-hour sessions.
SESSION_HARD_LIMIT_MINUTES = 11 * 60 + 50  # 11h50m
ZIP_RESERVE_MINUTES = 12
NO_NEW_RUN_SAFETY_MULTIPLIER = 1.35


DISPLAY_NAMES = {
    "full": "Full",
    "linear": "Linear",
    "prompt": "Visual Prompting",
    "conv": "Conv-Adapter",
    "piggyback": "Piggyback",
    "ssf": "SSF",
    "adaptformer": "AdaptFormer",
    "repadapter": "RepAdapter",
    "arc": "ARC",
    "vpt_shallow": "VPT-Shallow",
    "vpt_deep": "VPT-Deep",
    "convpass": "ConvPass",
    "fact_tt": "FacT-TT",
    "fact_tk": "FacT-TK",
    "vqt": "VQT",
    "spt_lora": "SPT-LoRA",
    "spt_adapter": "SPT-Adapter",
    "ml_decoder": "ML-Decoder",
    "segadapter": "SegAdapter",
    "trso": "Proposal",
}


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    dataset: str
    task: str
    backbone: str
    methods: tuple[str, ...]
    batch_size: int
    dataset_args: dict


@dataclass(frozen=True)
class Experiment:
    setting_key: str
    setting_label: str
    dataset: str
    task: str
    backbone: str
    method: str
    method_label: str
    batch_size: int
    dataset_args: dict
    estimated_minutes_t4_per_seed: int

    @property
    def experiment_id(self) -> str:
        return f"{self.setting_key}__{self.method}"

    @property
    def estimated_group_minutes_t4(self) -> int:
        return int(self.estimated_minutes_t4_per_seed) * len(SEEDS)


VIT_METHODS = (
    "full", "linear", "ssf", "adaptformer", "repadapter", "arc",
    "vpt_shallow", "vpt_deep", "convpass", "fact_tt", "fact_tk", "vqt",
    "spt_lora", "spt_adapter", "trso",
)

SETTINGS: tuple[Setting, ...] = (
    Setting("dtd_resnet50", "DTD / ResNet-50", "dtd", "single_label", "resnet50@torchvision",
            ("full", "linear", "prompt", "conv", "piggyback", "trso"), 16, {"dtd_partition": 1}),
    Setting("dtd_vit_b16", "DTD / ViT-B16", "dtd", "single_label", "vit_b_16@torchvision",
            VIT_METHODS, 8, {"dtd_partition": 1}),
    Setting("dtd_resnet18", "DTD / ResNet-18", "dtd", "single_label", "resnet18@torchvision",
            ("full", "linear", "prompt", "trso"), 32, {"dtd_partition": 1}),
    Setting("dtd_swin_t", "DTD / Swin-T", "dtd", "single_label", "swin_t@torchvision",
            ("full", "linear", "ssf", "trso"), 32, {"dtd_partition": 1}),
    Setting("flowers_resnet18", "Flowers / ResNet-18", "flowers102", "single_label", "resnet18@torchvision",
            ("full", "linear", "prompt", "trso"), 32, {}),
    Setting("flowers_vit_b16", "Flowers / ViT-B16", "flowers102", "single_label", "vit_b_16@torchvision",
            VIT_METHODS, 16, {}),
    Setting("flowers_swin_t", "Flowers / Swin-T", "flowers102", "single_label", "swin_t@torchvision",
            ("full", "linear", "ssf", "trso"), 32, {}),
    Setting("pet_resnet18", "Pet / ResNet-18", "oxfordiiitpet", "single_label", "resnet18@torchvision",
            ("full", "linear", "prompt", "trso"), 32, {}),
    Setting("pet_swin_t", "Pet / Swin-T", "oxfordiiitpet", "single_label", "swin_t@torchvision",
            ("full", "linear", "ssf", "trso"), 32, {}),
    Setting("voc2007_mobilenetv3_small", "VOC2007 / MobileNetV3-S", "voc2007", "multilabel",
            "mobilenet_v3_small@torchvision", ("full", "linear", "ml_decoder", "trso"), 32, {}),
    Setting("pet_segmentation_lraspp_mobilenetv3_large", "Pet segmentation / LR-ASPP MobileNetV3-L",
            "oxford_pet_segmentation", "semantic_segmentation", "lraspp_mobilenet_v3_large@torchvision",
            ("full", "linear", "segadapter", "trso"), 8,
            {"segmentation_num_classes": 3, "segmentation_ignore_index": 255}),
)

# Estimated minutes for one 30-epoch seed on a single T4 with AMP. These are
# intentionally conservative planning values derived from model/task complexity,
# not paper-reported wall-clock measurements. Actual time is recorded by the runner.
ESTIMATED_MINUTES_T4: dict[str, dict[str, int]] = {
    "dtd_resnet50": {"full": 52, "linear": 12, "prompt": 28, "conv": 34, "piggyback": 36, "trso": 33},
    "dtd_vit_b16": {"full": 95, "linear": 18, "ssf": 42, "adaptformer": 45, "repadapter": 46,
                    "arc": 48, "vpt_shallow": 37, "vpt_deep": 49, "convpass": 51, "fact_tt": 44,
                    "fact_tk": 45, "vqt": 43, "spt_lora": 68, "spt_adapter": 70, "trso": 52},
    "dtd_resnet18": {"full": 28, "linear": 8, "prompt": 18, "trso": 22},
    "dtd_swin_t": {"full": 85, "linear": 17, "ssf": 45, "trso": 50},
    "flowers_resnet18": {"full": 22, "linear": 7, "prompt": 14, "trso": 18},
    "flowers_vit_b16": {"full": 75, "linear": 15, "ssf": 34, "adaptformer": 36, "repadapter": 37,
                        "arc": 39, "vpt_shallow": 30, "vpt_deep": 40, "convpass": 42, "fact_tt": 36,
                        "fact_tk": 37, "vqt": 35, "spt_lora": 55, "spt_adapter": 57, "trso": 42},
    "flowers_swin_t": {"full": 65, "linear": 14, "ssf": 36, "trso": 40},
    "pet_resnet18": {"full": 30, "linear": 9, "prompt": 20, "trso": 24},
    "pet_swin_t": {"full": 85, "linear": 17, "ssf": 45, "trso": 50},
    "voc2007_mobilenetv3_small": {"full": 55, "linear": 12, "ml_decoder": 50, "trso": 40},
    "pet_segmentation_lraspp_mobilenetv3_large": {"full": 120, "linear": 25, "segadapter": 75, "trso": 80},
}


def experiments() -> tuple[Experiment, ...]:
    rows: list[Experiment] = []
    for setting in SETTINGS:
        estimates = ESTIMATED_MINUTES_T4[setting.key]
        missing = set(setting.methods) - set(estimates)
        if missing:
            raise RuntimeError(f"Missing runtime estimates for {setting.key}: {sorted(missing)}")
        for method in setting.methods:
            rows.append(Experiment(
                setting_key=setting.key,
                setting_label=setting.label,
                dataset=setting.dataset,
                task=setting.task,
                backbone=setting.backbone,
                method=method,
                method_label=DISPLAY_NAMES[method],
                batch_size=setting.batch_size,
                dataset_args=dict(setting.dataset_args),
                estimated_minutes_t4_per_seed=int(estimates[method]),
            ))
    return tuple(rows)


def build_sessions(session_count: int = SESSION_COUNT) -> dict[int, tuple[Experiment, ...]]:
    """Greedy balanced packing, then fastest -> slowest ordering per session."""
    rows = sorted(
        experiments(),
        key=lambda x: (-x.estimated_group_minutes_t4, x.experiment_id),
    )
    bins: list[list[Experiment]] = [[] for _ in range(session_count)]
    totals = [0 for _ in range(session_count)]
    for exp in rows:
        index = min(range(session_count), key=lambda i: (totals[i], i))
        bins[index].append(exp)
        totals[index] += exp.estimated_group_minutes_t4
    result: dict[int, tuple[Experiment, ...]] = {}
    for index, bucket in enumerate(bins, 1):
        bucket = sorted(bucket, key=lambda x: (x.estimated_minutes_t4_per_seed, x.setting_key, x.method))
        result[index] = tuple(bucket)
    return result


SESSIONS = build_sessions()


def session_estimated_minutes_t4(session_id: int) -> int:
    return sum(x.estimated_group_minutes_t4 for x in SESSIONS[int(session_id)])


def expanded_run_count(session_id: int) -> int:
    return len(SESSIONS[int(session_id)]) * len(SEEDS)


def total_group_count() -> int:
    return len(experiments())


def total_run_count() -> int:
    return total_group_count() * len(SEEDS)


def session_payload(session_id: int) -> dict:
    sid = int(session_id)
    groups = []
    for order, exp in enumerate(SESSIONS[sid], 1):
        groups.append({
            "order_fast_to_slow": order,
            **asdict(exp),
            "experiment_id": exp.experiment_id,
            "seeds": list(SEEDS),
            "estimated_minutes_t4_all_seeds": exp.estimated_group_minutes_t4,
        })
    return {
        "protocol": "reviewer_main_controlled_fair_v3_kaggle_sessions",
        "session": sid,
        "session_count": SESSION_COUNT,
        "reference_gpu": REFERENCE_GPU,
        "epochs": EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "input_size": INPUT_SIZE,
        "split_seed": SPLIT_SEED,
        "seeds": list(SEEDS),
        "head_init_policy": "random",
        "peft_head_lr_scale": 1.0,
        "method_internal_defaults": "baseline_recipes.py",
        "outer_training_protocol": {
            "optimizer": "adamw",
            "peft_lr": 1e-3,
            "full_lr": 1e-4,
            "linear_lr": 1e-3,
            "weight_decay": 1e-4,
            "min_lr": 1e-6,
            "augmentation": "strong",
        },
        "hard_limit_minutes": SESSION_HARD_LIMIT_MINUTES,
        "zip_reserve_minutes": ZIP_RESERVE_MINUTES,
        "estimated_session_minutes_t4": session_estimated_minutes_t4(sid),
        "estimated_session_hours_t4": round(session_estimated_minutes_t4(sid) / 60.0, 2),
        "experiment_groups": groups,
        "expanded_seed_runs": expanded_run_count(sid),
    }


def full_payload() -> dict:
    return {
        "protocol": "reviewer_main_controlled_fair_v3_kaggle_sessions",
        "reference_gpu": REFERENCE_GPU,
        "session_count": SESSION_COUNT,
        "experiment_groups": total_group_count(),
        "seeds": list(SEEDS),
        "total_training_runs": total_run_count(),
        "estimated_total_minutes_t4": sum(x.estimated_group_minutes_t4 for x in experiments()),
        "estimated_total_hours_t4": round(sum(x.estimated_group_minutes_t4 for x in experiments()) / 60.0, 2),
        "sessions": [session_payload(i) for i in range(1, SESSION_COUNT + 1)],
    }


def write_plan(directory: str | Path) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "reviewer_matrix_plan.json"
    csv_path = directory / "reviewer_matrix_plan.csv"
    json_path.write_text(json.dumps(full_payload(), indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "session", "order_fast_to_slow", "setting", "dataset", "backbone", "method", "method_label",
            "seeds", "batch_size", "estimated_minutes_t4_per_seed", "estimated_minutes_t4_all_seeds",
            "estimated_session_hours_t4",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sid in range(1, SESSION_COUNT + 1):
            session_hours = round(session_estimated_minutes_t4(sid) / 60.0, 2)
            for order, exp in enumerate(SESSIONS[sid], 1):
                writer.writerow({
                    "session": sid,
                    "order_fast_to_slow": order,
                    "setting": exp.setting_label,
                    "dataset": exp.dataset,
                    "backbone": exp.backbone,
                    "method": exp.method,
                    "method_label": exp.method_label,
                    "seeds": ",".join(map(str, SEEDS)),
                    "batch_size": exp.batch_size,
                    "estimated_minutes_t4_per_seed": exp.estimated_minutes_t4_per_seed,
                    "estimated_minutes_t4_all_seeds": exp.estimated_group_minutes_t4,
                    "estimated_session_hours_t4": session_hours,
                })
    return json_path, csv_path


# Import-time invariants protect the publication plan from silent drift.
assert total_group_count() == 68, total_group_count()
assert total_run_count() == 204, total_run_count()
assert len(SESSIONS) == SESSION_COUNT
assert set(range(1, SESSION_COUNT + 1)) == set(SESSIONS)
assert len({x.experiment_id for x in experiments()}) == 68
assert sorted(x.experiment_id for bucket in SESSIONS.values() for x in bucket) == sorted(x.experiment_id for x in experiments())
assert all(
    list(bucket) == sorted(bucket, key=lambda x: (x.estimated_minutes_t4_per_seed, x.setting_key, x.method))
    for bucket in SESSIONS.values()
)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    json_path, csv_path = write_plan(root)
    print(json_path)
    print(csv_path)
    for sid in range(1, SESSION_COUNT + 1):
        print(
            f"Session {sid:02d}: {len(SESSIONS[sid])} experiment groups / "
            f"{expanded_run_count(sid)} seed-runs / {session_estimated_minutes_t4(sid)} min "
            f"({session_estimated_minutes_t4(sid)/60:.2f} h) on {REFERENCE_GPU} estimate"
        )
