"""Runtime-safe Kaggle plan for reviewer-requested sensitivity experiments.

This protocol complements ``kaggle/reviewer_matrix``.  The main matrix covers
all principal paper tables with seeds 0/1/2; this module covers experiments
explicitly requested during peer review: broader reliability ablations,
calibration amount/partition/batch sensitivity, alternative D0/D1 rules,
0.5x/1x/2x R stress tests, and a validation-ranked LoRA budget sweep.

The 72 logical groups expand to 180 training runs and are greedily packed into
14 Kaggle sessions.  Runtime values are conservative NVIDIA T4 planning
estimates, not reported benchmark measurements.  Every logical group stays in a
single session so its requested seeds can be archived together.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any

REFERENCE_GPU = "NVIDIA T4"
SESSION_COUNT = 14
SESSION_HARD_LIMIT_MINUTES = 11 * 60 + 50
ZIP_RESERVE_MINUTES = 12
NO_NEW_RUN_SAFETY_MULTIPLIER = 1.35
MAX_PLANNED_SESSION_MINUTES_T4 = 480
EPOCHS = 30
WARMUP_EPOCHS = 3
INPUT_SIZE = 224
SPLIT_SEED = 0
DEFAULT_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class SensitivitySetting:
    key: str
    label: str
    dataset: str
    task: str
    backbone: str
    batch_size: int
    dataset_args: dict[str, Any]
    trso_minutes_t4: int
    lora_minutes_t4: int | None = None


@dataclass(frozen=True)
class SensitivityGroup:
    group_id: str
    study: str
    setting_key: str
    setting_label: str
    dataset: str
    task: str
    backbone: str
    batch_size: int
    dataset_args: dict[str, Any]
    variant: str
    controls: dict[str, Any]
    seeds: tuple[int, ...]
    reviewer_requirements: tuple[str, ...]
    estimated_minutes_t4_per_run: int

    @property
    def estimated_group_minutes_t4(self) -> int:
        return self.estimated_minutes_t4_per_run * len(self.seeds)

    def job_id(self, seed: int) -> str:
        return f"{self.group_id}__seed{int(seed)}"


SETTINGS = {
    "dtd_resnet18": SensitivitySetting(
        "dtd_resnet18", "DTD / ResNet-18", "dtd", "single_label",
        "resnet18@torchvision", 32, {"dtd_partition": 1}, 22,
    ),
    "dtd_resnet50": SensitivitySetting(
        "dtd_resnet50", "DTD / ResNet-50", "dtd", "single_label",
        "resnet50@torchvision", 16, {"dtd_partition": 1}, 33,
    ),
    "flowers_vit_b16": SensitivitySetting(
        "flowers_vit_b16", "Flowers-102 / ViT-B16", "flowers102", "single_label",
        "vit_b_16@torchvision", 16, {}, 42, 45,
    ),
    "dtd_vit_b16": SensitivitySetting(
        "dtd_vit_b16", "DTD / ViT-B16", "dtd", "single_label",
        "vit_b_16@torchvision", 8, {"dtd_partition": 1}, 52, 55,
    ),
}


def _slug(value: Any) -> str:
    text = str(value).strip().lower().replace(".", "p")
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _group(setting_key: str, study: str, variant: str, *, controls: dict[str, Any],
           seeds: tuple[int, ...] = DEFAULT_SEEDS, reviewers: tuple[str, ...],
           method: str = "trso") -> SensitivityGroup:
    setting = SETTINGS[setting_key]
    estimate = setting.trso_minutes_t4 if method == "trso" else int(setting.lora_minutes_t4 or setting.trso_minutes_t4)
    gid = f"{study}__{setting_key}__{_slug(variant)}"
    return SensitivityGroup(
        group_id=gid,
        study=study,
        setting_key=setting.key,
        setting_label=setting.label,
        dataset=setting.dataset,
        task=setting.task,
        backbone=setting.backbone,
        batch_size=setting.batch_size,
        dataset_args=dict(setting.dataset_args),
        variant=variant,
        controls=dict(controls),
        seeds=tuple(int(x) for x in seeds),
        reviewer_requirements=tuple(reviewers),
        estimated_minutes_t4_per_run=estimate,
    )


def groups() -> tuple[SensitivityGroup, ...]:
    rows: list[SensitivityGroup] = []

    # Reviewer 2 C2: repeat reliability ablation on 2-3 additional settings,
    # including a Transformer.  "no_crossfit" is the implementation name for
    # removing partition-consistency weighting; it is not statistical cross-fitting.
    for setting_key in ("dtd_resnet18", "dtd_resnet50", "flowers_vit_b16"):
        for ablation in ("full", "no_crossfit", "no_sampling_variance", "diagonal_only"):
            rows.append(_group(
                setting_key, "reliability", ablation,
                controls={"trso_ablation": ablation}, reviewers=("R2-C2",),
            ))

    # Reviewer 1 C3 / Reviewer 2 C1,C5: calibration amount and partition stability.
    # Alternating-partition rows use all optimization seeds.  Seeded-random rows
    # hold optimization seed at 0 and vary only the partition seed, isolating the
    # allocation effect requested by the reviewers.
    for setting_key in ("dtd_resnet18", "flowers_vit_b16"):
        for fraction in (0.25, 0.5, 1.0):
            variant = f"frac_{fraction:g}_alternating"
            rows.append(_group(
                setting_key, "calibration", variant,
                controls={"calibration_fraction": fraction, "partition_mode": "alternating", "partition_seed": 0},
                reviewers=("R1-C3", "R2-C1", "R2-C5"),
            ))
            for partition_seed in (0, 1, 2):
                variant = f"frac_{fraction:g}_random_p{partition_seed}"
                rows.append(_group(
                    setting_key, "calibration", variant,
                    controls={"calibration_fraction": fraction, "partition_mode": "seeded_random", "partition_seed": partition_seed},
                    seeds=(0,), reviewers=("R1-C3", "R2-C1", "R2-C5"),
                ))

    # Reviewer 1 C3 / Reviewer 2 C5: calibration batching without changing the
    # optimization batch size.
    for setting_key, sizes in (
        ("dtd_resnet18", (8, 16, 32, 64)),
        ("flowers_vit_b16", (4, 8, 16, 32)),
    ):
        for calibration_batch_size in sizes:
            variant = f"calib_batch_{calibration_batch_size}"
            rows.append(_group(
                setting_key, "batch_sensitivity", variant,
                controls={"calibration_batch_size": calibration_batch_size},
                reviewers=("R1-C3", "R2-C5"),
            ))

    # Reviewer 3 C1-C3 / Reviewer 2 C3: explicit alternatives to geometric mean.
    for setting_key in ("dtd_resnet18", "flowers_vit_b16"):
        for rule in ("shannon", "harmonic", "geometric", "arithmetic"):
            rows.append(_group(
                setting_key, "mode_rules", rule,
                controls={"mode_rule": rule}, reviewers=("R2-C3", "R3-C1", "R3-C2", "R3-C3"),
            ))

    # Reviewer 2 C3: direct 0.5x/1x/2x retained-mode stress test.
    for setting_key in ("dtd_resnet18", "flowers_vit_b16"):
        for scale in (0.5, 1.0, 2.0):
            variant = f"rscale_{scale:g}"
            rows.append(_group(
                setting_key, "r_scale", variant,
                controls={"r_scale": scale}, reviewers=("R2-C3",),
            ))

    # Reviewer 1 C5 / Reviewer 2 C3 / Reviewer 3 C4: fidelity-audited vanilla
    # LoRA rank sweep.  The post-processing script chooses the closest parameter
    # budget and a validation-selected oracle rank; test metrics are never used
    # to choose the rank.
    for setting_key in ("dtd_vit_b16", "flowers_vit_b16"):
        for rank in (1, 2, 4, 8, 16, 32, 64):
            variant = f"lora_r{rank}"
            rows.append(_group(
                setting_key, "lora_rank_sweep", variant,
                controls={"lora_rank": rank}, reviewers=("R1-C5", "R2-C3", "R3-C4"), method="lora",
            ))

    return tuple(rows)


def total_group_count() -> int:
    return len(groups())


def total_run_count() -> int:
    return sum(len(group.seeds) for group in groups())


def build_sessions(session_count: int = SESSION_COUNT) -> dict[int, tuple[SensitivityGroup, ...]]:
    # Greedy bin packing by whole logical group keeps requested seeds together.
    ordered = sorted(groups(), key=lambda g: (-g.estimated_group_minutes_t4, g.group_id))
    bins: list[list[SensitivityGroup]] = [[] for _ in range(session_count)]
    totals = [0] * session_count
    for group in ordered:
        idx = min(range(session_count), key=lambda i: (totals[i], i))
        bins[idx].append(group)
        totals[idx] += group.estimated_group_minutes_t4
    result: dict[int, tuple[SensitivityGroup, ...]] = {}
    for sid, bucket in enumerate(bins, 1):
        result[sid] = tuple(sorted(bucket, key=lambda g: (g.estimated_group_minutes_t4, g.group_id)))
    return result


SESSIONS = build_sessions()


def session_estimated_minutes_t4(session_id: int) -> int:
    return sum(g.estimated_group_minutes_t4 for g in SESSIONS[int(session_id)])


def session_run_count(session_id: int) -> int:
    return sum(len(g.seeds) for g in SESSIONS[int(session_id)])


def reviewer_coverage() -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for group in groups():
        for requirement in group.reviewer_requirements:
            row = mapping.setdefault(requirement, {"logical_groups": 0, "training_runs": 0, "studies": set(), "settings": set()})
            row["logical_groups"] += 1
            row["training_runs"] += len(group.seeds)
            row["studies"].add(group.study)
            row["settings"].add(group.setting_label)
    return {
        key: {
            **value,
            "studies": sorted(value["studies"]),
            "settings": sorted(value["settings"]),
        }
        for key, value in sorted(mapping.items())
    }


def session_payload(session_id: int) -> dict[str, Any]:
    sid = int(session_id)
    payload_groups = []
    for order, group in enumerate(SESSIONS[sid], 1):
        payload_groups.append({
            "order_fast_to_slow": order,
            **asdict(group),
            "estimated_group_minutes_t4": group.estimated_group_minutes_t4,
            "job_ids": [group.job_id(seed) for seed in group.seeds],
        })
    return {
        "protocol": "reviewer_sensitivity_controlled_fair_v1_kaggle_sessions",
        "session": sid,
        "session_count": SESSION_COUNT,
        "reference_gpu": REFERENCE_GPU,
        "epochs": EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "input_size": INPUT_SIZE,
        "split_seed": SPLIT_SEED,
        "hard_limit_minutes": SESSION_HARD_LIMIT_MINUTES,
        "zip_reserve_minutes": ZIP_RESERVE_MINUTES,
        "estimated_session_minutes_t4": session_estimated_minutes_t4(sid),
        "estimated_session_hours_t4": round(session_estimated_minutes_t4(sid) / 60.0, 2),
        "logical_groups": payload_groups,
        "expanded_training_runs": session_run_count(sid),
    }


def full_payload() -> dict[str, Any]:
    total_minutes = sum(g.estimated_group_minutes_t4 for g in groups())
    return {
        "protocol": "reviewer_sensitivity_controlled_fair_v1_kaggle_sessions",
        "reference_gpu": REFERENCE_GPU,
        "session_count": SESSION_COUNT,
        "logical_groups": total_group_count(),
        "total_training_runs": total_run_count(),
        "estimated_total_minutes_t4": total_minutes,
        "estimated_total_hours_t4": round(total_minutes / 60.0, 2),
        "reviewer_coverage": reviewer_coverage(),
        "sessions": [session_payload(i) for i in range(1, SESSION_COUNT + 1)],
    }


def write_plan(directory: str | Path) -> tuple[Path, Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "reviewer_sensitivity_plan.json"
    csv_path = directory / "reviewer_sensitivity_plan.csv"
    coverage_path = directory / "reviewer_coverage.json"
    json_path.write_text(json.dumps(full_payload(), indent=2), encoding="utf-8")
    coverage_path.write_text(json.dumps(reviewer_coverage(), indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "session", "order_fast_to_slow", "group_id", "study", "setting", "variant",
            "seeds", "reviewer_requirements", "estimated_minutes_t4_per_run",
            "estimated_group_minutes_t4", "estimated_session_hours_t4", "controls_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sid in range(1, SESSION_COUNT + 1):
            session_hours = round(session_estimated_minutes_t4(sid) / 60.0, 2)
            for order, group in enumerate(SESSIONS[sid], 1):
                writer.writerow({
                    "session": sid,
                    "order_fast_to_slow": order,
                    "group_id": group.group_id,
                    "study": group.study,
                    "setting": group.setting_label,
                    "variant": group.variant,
                    "seeds": ",".join(map(str, group.seeds)),
                    "reviewer_requirements": ",".join(group.reviewer_requirements),
                    "estimated_minutes_t4_per_run": group.estimated_minutes_t4_per_run,
                    "estimated_group_minutes_t4": group.estimated_group_minutes_t4,
                    "estimated_session_hours_t4": session_hours,
                    "controls_json": json.dumps(group.controls, sort_keys=True),
                })
    return json_path, csv_path, coverage_path


# Publication-plan invariants: fail loudly if a later edit silently drops work.
assert total_group_count() == 72, total_group_count()
assert total_run_count() == 180, total_run_count()
assert len(SESSIONS) == SESSION_COUNT
assert sorted(g.group_id for bucket in SESSIONS.values() for g in bucket) == sorted(g.group_id for g in groups())
assert len({g.group_id for g in groups()}) == total_group_count()
assert max(session_estimated_minutes_t4(sid) for sid in SESSIONS) <= MAX_PLANNED_SESSION_MINUTES_T4


if __name__ == "__main__":
    paths = write_plan(Path(__file__).resolve().parent)
    print("\n".join(map(str, paths)))
    for sid in range(1, SESSION_COUNT + 1):
        print(
            f"Session {sid:02d}: {len(SESSIONS[sid])} groups / {session_run_count(sid)} runs / "
            f"{session_estimated_minutes_t4(sid)} min ({session_estimated_minutes_t4(sid)/60:.2f} h) T4 estimate"
        )
