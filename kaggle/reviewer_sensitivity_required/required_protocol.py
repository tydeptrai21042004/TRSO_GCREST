"""Reviewer-required sensitivity scheduler (compact v2).

This folder is the reviewer-sensitivity companion to ``kaggle/reviewer_matrix``.
The 204-run main matrix already supplies three-seed default Proposal results and
all main-table baselines.  This compact scheduler therefore runs only NEW
sensitivity evidence that reviewers explicitly requested.

Reused from ``reviewer_matrix`` (not rerun here):
- default/full Proposal at seeds 0/1/2;
- 100% calibration with alternating partition;
- default calibration batch size;
- geometric D0/D1 rule;
- R scale = 1.0.

New work here:
- broader reliability ablations on 3 settings;
- calibration amount + partition stability on CNN and Transformer settings;
- calibration-batch sensitivity on the primary ablation setting;
- alternative D0/D1 rules on the primary ablation setting;
- 0.5x/2x R stress test on the primary ablation setting;
- LoRA validation-rank sweep on two ViT settings for matched-budget/oracle
  comparison (rank is selected on validation, never on test).

The 39 logical groups expand to 105 new training runs and are packed into
9 Kaggle sessions. Runtime values are conservative T4 planning estimates.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import csv, json
from pathlib import Path
from typing import Any

REFERENCE_GPU = "NVIDIA T4"
SESSION_COUNT = 9
SESSION_HARD_LIMIT_MINUTES = 11 * 60 + 50
ZIP_RESERVE_MINUTES = 12
NO_NEW_RUN_SAFETY_MULTIPLIER = 1.35
MAX_PLANNED_SESSION_MINUTES_T4 = 480
EPOCHS = 30
WARMUP_EPOCHS = 3
INPUT_SIZE = 224
SPLIT_SEED = 0
DEFAULT_SEEDS = (0, 1, 2)
PROTOCOL_NAME = "reviewer_sensitivity_required_v2_kaggle_sessions"

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
        group_id=gid, study=study, setting_key=setting.key, setting_label=setting.label,
        dataset=setting.dataset, task=setting.task, backbone=setting.backbone,
        batch_size=setting.batch_size, dataset_args=dict(setting.dataset_args), variant=variant,
        controls=dict(controls), seeds=tuple(int(x) for x in seeds),
        reviewer_requirements=tuple(reviewers), estimated_minutes_t4_per_run=estimate,
    )

def groups() -> tuple[SensitivityGroup, ...]:
    rows: list[SensitivityGroup] = []

    # R2-C2. The full/default Proposal rows are reused from reviewer_matrix.
    # This folder runs only the three component-removal variants.
    for setting_key in ("dtd_resnet18", "dtd_resnet50", "flowers_vit_b16"):
        for ablation in ("no_crossfit", "no_sampling_variance", "diagonal_only"):
            rows.append(_group(
                setting_key, "reliability", ablation,
                controls={"trso_ablation": ablation}, reviewers=("R2-C2",),
            ))

    # R1-C3 / R2-C1,C5. 100% alternating results are reused from reviewer_matrix.
    # 25/50% use all optimization seeds; partition stability is isolated at 100%
    # with optimization seed fixed to 0 and partition seed varied.
    for setting_key in ("dtd_resnet18", "flowers_vit_b16"):
        for fraction in (0.25, 0.5):
            rows.append(_group(
                setting_key, "calibration", f"frac_{fraction:g}_alternating",
                controls={"calibration_fraction": fraction, "partition_mode": "alternating", "partition_seed": 0},
                reviewers=("R1-C3", "R2-C1", "R2-C5"),
            ))
        for partition_seed in (0, 1, 2):
            rows.append(_group(
                setting_key, "calibration", f"frac_1_random_p{partition_seed}",
                controls={"calibration_fraction": 1.0, "partition_mode": "seeded_random", "partition_seed": partition_seed},
                seeds=(0,), reviewers=("R1-C3", "R2-C1", "R2-C5"),
            ))

    # R1-C3 / R2-C5. Primary ablation setting only. Default calibration batch=32
    # is reused from reviewer_matrix, so only non-default calibration batch sizes run.
    for calibration_batch_size in (8, 16, 64):
        rows.append(_group(
            "dtd_resnet18", "batch_sensitivity", f"calib_batch_{calibration_batch_size}",
            controls={"calibration_batch_size": calibration_batch_size},
            reviewers=("R1-C3", "R2-C5"),
        ))

    # R3-C1/C2 and R2-C3. Geometric default is reused from reviewer_matrix.
    for rule in ("shannon", "harmonic", "arithmetic"):
        rows.append(_group(
            "dtd_resnet18", "mode_rules", rule,
            controls={"mode_rule": rule}, reviewers=("R2-C3", "R3-C1", "R3-C2", "R3-C3"),
        ))

    # R2-C3. R scale 1.0 is reused from reviewer_matrix.
    for scale in (0.5, 2.0):
        rows.append(_group(
            "dtd_resnet18", "r_scale", f"rscale_{scale:g}",
            controls={"r_scale": scale}, reviewers=("R2-C3",),
        ))

    # R1-C5 / R2-C3 / R3-C4. Two representative Transformer settings: one
    # challenging regime and one favorable regime.  Ranks span the practical
    # budget neighborhood and support validation-selected oracle reporting.
    for setting_key in ("dtd_vit_b16", "flowers_vit_b16"):
        for rank in (1, 2, 4, 8, 16, 32):
            rows.append(_group(
                setting_key, "lora_rank_sweep", f"lora_r{rank}",
                controls={"lora_rank": rank}, reviewers=("R1-C5", "R2-C3", "R3-C4"), method="lora",
            ))

    return tuple(rows)

def total_group_count() -> int:
    return len(groups())

def total_run_count() -> int:
    return sum(len(g.seeds) for g in groups())

def build_sessions(session_count: int = SESSION_COUNT) -> dict[int, tuple[SensitivityGroup, ...]]:
    ordered = sorted(groups(), key=lambda g: (-g.estimated_group_minutes_t4, g.group_id))
    bins: list[list[SensitivityGroup]] = [[] for _ in range(session_count)]
    totals = [0] * session_count
    for group in ordered:
        idx = min(range(session_count), key=lambda i: (totals[i], i))
        bins[idx].append(group)
        totals[idx] += group.estimated_group_minutes_t4
    return {
        sid: tuple(sorted(bucket, key=lambda g: (g.estimated_group_minutes_t4, g.group_id)))
        for sid, bucket in enumerate(bins, 1)
    }

SESSIONS = build_sessions()

def session_estimated_minutes_t4(session_id: int) -> int:
    return sum(g.estimated_group_minutes_t4 for g in SESSIONS[int(session_id)])

def session_run_count(session_id: int) -> int:
    return sum(len(g.seeds) for g in SESSIONS[int(session_id)])

def reviewer_coverage() -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for group in groups():
        for req in group.reviewer_requirements:
            row = mapping.setdefault(req, {"logical_groups": 0, "training_runs": 0, "studies": set(), "settings": set()})
            row["logical_groups"] += 1
            row["training_runs"] += len(group.seeds)
            row["studies"].add(group.study)
            row["settings"].add(group.setting_label)
    return {
        k: {**v, "studies": sorted(v["studies"]), "settings": sorted(v["settings"])}
        for k, v in sorted(mapping.items())
    }

def reused_defaults() -> dict[str, Any]:
    return {
        "source": "kaggle/reviewer_matrix (204-run, 3-seed main matrix)",
        "default_proposal_seeds": [0, 1, 2],
        "reused_conditions": [
            "full/default Proposal for all sensitivity settings",
            "calibration_fraction=1.0 with alternating partition",
            "DTD/ResNet-18 default calibration_batch_size=32",
            "mode_rule=geometric",
            "r_scale=1.0",
        ],
        "reason": "Avoid duplicate training while preserving reviewer-controlled comparisons.",
    }

def session_payload(session_id: int) -> dict[str, Any]:
    sid = int(session_id)
    payload_groups = []
    for order, group in enumerate(SESSIONS[sid], 1):
        payload_groups.append({
            "order_fast_to_slow": order, **asdict(group),
            "estimated_group_minutes_t4": group.estimated_group_minutes_t4,
            "job_ids": [group.job_id(seed) for seed in group.seeds],
        })
    return {
        "protocol": PROTOCOL_NAME, "session": sid, "session_count": SESSION_COUNT,
        "reference_gpu": REFERENCE_GPU, "epochs": EPOCHS, "warmup_epochs": WARMUP_EPOCHS,
        "input_size": INPUT_SIZE, "split_seed": SPLIT_SEED,
        "hard_limit_minutes": SESSION_HARD_LIMIT_MINUTES, "zip_reserve_minutes": ZIP_RESERVE_MINUTES,
        "estimated_session_minutes_t4": session_estimated_minutes_t4(sid),
        "estimated_session_hours_t4": round(session_estimated_minutes_t4(sid) / 60.0, 2),
        "logical_groups": payload_groups, "expanded_training_runs": session_run_count(sid),
        "reused_defaults": reused_defaults(),
    }

def full_payload() -> dict[str, Any]:
    total_minutes = sum(g.estimated_group_minutes_t4 for g in groups())
    return {
        "protocol": PROTOCOL_NAME, "reference_gpu": REFERENCE_GPU, "session_count": SESSION_COUNT,
        "logical_groups": total_group_count(), "total_training_runs": total_run_count(),
        "estimated_total_minutes_t4": total_minutes,
        "estimated_total_hours_t4": round(total_minutes / 60.0, 2),
        "reviewer_coverage": reviewer_coverage(), "reused_defaults": reused_defaults(),
        "sessions": [session_payload(i) for i in range(1, SESSION_COUNT + 1)],
    }

def write_plan(directory: str | Path) -> tuple[Path, Path, Path]:
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "reviewer_required_plan.json"
    csv_path = directory / "reviewer_required_plan.csv"
    coverage_path = directory / "reviewer_required_coverage.json"
    json_path.write_text(json.dumps(full_payload(), indent=2), encoding="utf-8")
    coverage_path.write_text(json.dumps(reviewer_coverage(), indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["session","order_fast_to_slow","group_id","study","setting","variant","seeds",
                  "reviewer_requirements","estimated_minutes_t4_per_run","estimated_group_minutes_t4",
                  "estimated_session_hours_t4","controls_json"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for sid in range(1, SESSION_COUNT + 1):
            hours = round(session_estimated_minutes_t4(sid)/60.0, 2)
            for order, group in enumerate(SESSIONS[sid], 1):
                writer.writerow({
                    "session":sid,"order_fast_to_slow":order,"group_id":group.group_id,"study":group.study,
                    "setting":group.setting_label,"variant":group.variant,"seeds":",".join(map(str,group.seeds)),
                    "reviewer_requirements":",".join(group.reviewer_requirements),
                    "estimated_minutes_t4_per_run":group.estimated_minutes_t4_per_run,
                    "estimated_group_minutes_t4":group.estimated_group_minutes_t4,
                    "estimated_session_hours_t4":hours,"controls_json":json.dumps(group.controls,sort_keys=True),
                })
    return json_path, csv_path, coverage_path

assert total_group_count() == 39, total_group_count()
assert total_run_count() == 105, total_run_count()
assert len(SESSIONS) == SESSION_COUNT
assert len({g.group_id for g in groups()}) == total_group_count()
assert sorted(g.group_id for b in SESSIONS.values() for g in b) == sorted(g.group_id for g in groups())
assert max(session_estimated_minutes_t4(sid) for sid in SESSIONS) <= MAX_PLANNED_SESSION_MINUTES_T4

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    print("\n".join(map(str, write_plan(here))))
    for sid in range(1, SESSION_COUNT + 1):
        print(f"Session {sid:02d}: {len(SESSIONS[sid])} groups / {session_run_count(sid)} runs / "
              f"{session_estimated_minutes_t4(sid)} min ({session_estimated_minutes_t4(sid)/60:.2f} h) T4")
