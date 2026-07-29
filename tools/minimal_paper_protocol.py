"""Canonical 46-run comparison protocol split across six Kaggle sessions.

The protocol has four non-overlapping result groups:
1. strict literature baselines from dedicated vision-method papers;
2. corrected reference controls (full fine-tuning and linear probing);
3. the full TRSO proposal;
4. TRSO-only structural ablations.

Full fine-tuning and linear probing are reported as reference controls, never as
paper baselines. Each Kaggle session is self-contained: it first trains the
matching reference controls, then reuses the exact linear-probe checkpoint to
initialize compatible literature baselines, TRSO, and (in Session 3) TRSO
ablations. No trained full-TRSO checkpoint is reused by an ablation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model_support import method_category


@dataclass(frozen=True)
class Stage:
    name: str
    session: int
    category: str
    dataset: str
    task: str
    backbones: str
    methods: str
    expected_runs: int
    batch_size: int
    dataset_args: dict


STAGES = (
    # Session 1: DTD / ResNet-50 (6 runs).
    Stage("dtd_resnet50_reference_controls", 1, "reference_control", "dtd", "auto", "resnet50@torchvision",
          "full,linear", 2, 16, {"dtd_partition": 1}),
    Stage("dtd_resnet50_literature_baselines", 1, "paper_baseline", "dtd", "auto", "resnet50@torchvision",
          "prompt,conv,piggyback", 3, 16, {"dtd_partition": 1}),
    Stage("dtd_resnet50_trso_proposal", 1, "proposal", "dtd", "auto", "resnet50@torchvision",
          "trso", 1, 16, {"dtd_partition": 1}),

    # Session 2: DTD / ViT-B/16 (10 runs).
    Stage("dtd_vit_b16_reference_controls", 2, "reference_control", "dtd", "auto", "vit_b_16@torchvision",
          "full,linear", 2, 8, {"dtd_partition": 1}),
    Stage("dtd_vit_b16_literature_baselines", 2, "paper_baseline", "dtd", "auto", "vit_b_16@torchvision",
          "ssf,adaptformer,repadapter,arc,vpt_shallow,vpt_deep,convpass", 7, 8, {"dtd_partition": 1}),
    Stage("dtd_vit_b16_trso_proposal", 2, "proposal", "dtd", "auto", "vit_b_16@torchvision",
          "trso", 1, 8, {"dtd_partition": 1}),

    # Session 3: DTD / small backbones (8 main + 4 ablation runs).
    Stage("dtd_small_reference_controls", 3, "reference_control", "dtd", "auto",
          "resnet18@torchvision,swin_t@torchvision", "full,linear", 4, 32, {"dtd_partition": 1}),
    Stage("dtd_resnet18_literature_baseline", 3, "paper_baseline", "dtd", "auto", "resnet18@torchvision",
          "prompt", 1, 32, {"dtd_partition": 1}),
    Stage("dtd_swin_t_literature_baseline", 3, "paper_baseline", "dtd", "auto", "swin_t@torchvision",
          "ssf", 1, 32, {"dtd_partition": 1}),
    Stage("dtd_small_trso_proposal", 3, "proposal", "dtd", "auto",
          "resnet18@torchvision,swin_t@torchvision", "trso", 2, 32, {"dtd_partition": 1}),

    # Session 4: Flowers-102 / small backbones (6 runs).
    Stage("flowers102_small_reference_controls", 4, "reference_control", "flowers102", "auto",
          "resnet18@torchvision,swin_t@torchvision", "full,linear", 4, 32, {}),
    Stage("flowers102_small_trso_proposal", 4, "proposal", "flowers102", "auto",
          "resnet18@torchvision,swin_t@torchvision", "trso", 2, 32, {}),

    # Session 5: Oxford-IIIT Pet classification / small backbones (6 runs).
    Stage("oxford_pets_small_reference_controls", 5, "reference_control", "oxfordiiitpet", "auto",
          "resnet18@torchvision,swin_t@torchvision", "full,linear", 4, 32, {}),
    Stage("oxford_pets_small_trso_proposal", 5, "proposal", "oxfordiiitpet", "auto",
          "resnet18@torchvision,swin_t@torchvision", "trso", 2, 32, {}),

    # Session 6: multilabel + segmentation tasks (6 runs).
    Stage("voc2007_reference_controls", 6, "reference_control", "voc2007", "multilabel",
          "mobilenet_v3_small@torchvision", "full,linear", 2, 32, {}),
    Stage("voc2007_trso_proposal", 6, "proposal", "voc2007", "multilabel",
          "mobilenet_v3_small@torchvision", "trso", 1, 32, {}),
    Stage("oxford_pet_segmentation_reference_controls", 6, "reference_control", "oxford_pet_segmentation", "semantic_segmentation",
          "lraspp_mobilenet_v3_large@torchvision", "full,linear", 2, 8,
          {"segmentation_num_classes": 3, "segmentation_ignore_index": 255}),
    Stage("oxford_pet_segmentation_trso_proposal", 6, "proposal", "oxford_pet_segmentation", "semantic_segmentation",
          "lraspp_mobilenet_v3_large@torchvision", "trso", 1, 8,
          {"segmentation_num_classes": 3, "segmentation_ignore_index": 255}),
)

ABLATIONS = ("diagonal_only", "no_sampling_variance", "no_crossfit", "head_only")
ABLATION_SESSION = 3
CATEGORY_RUNS = {
    category: sum(stage.expected_runs for stage in STAGES if stage.category == category)
    for category in ("paper_baseline", "reference_control", "proposal")
}
MAIN_RUNS = sum(stage.expected_runs for stage in STAGES)
TOTAL_RUNS = MAIN_RUNS + len(ABLATIONS)
SESSION_MAIN_RUNS = {
    session: sum(stage.expected_runs for stage in STAGES if stage.session == session)
    for session in range(1, 7)
}
SESSION_TOTAL_RUNS = {
    session: count + (len(ABLATIONS) if session == ABLATION_SESSION else 0)
    for session, count in SESSION_MAIN_RUNS.items()
}

assert CATEGORY_RUNS == {"paper_baseline": 12, "reference_control": 20, "proposal": 10}
assert MAIN_RUNS == 42
assert TOTAL_RUNS == 46
assert SESSION_MAIN_RUNS == {1: 6, 2: 10, 3: 8, 4: 6, 5: 6, 6: 6}
assert SESSION_TOTAL_RUNS == {1: 6, 2: 10, 3: 12, 4: 6, 5: 6, 6: 6}
assert sum(SESSION_TOTAL_RUNS.values()) == TOTAL_RUNS
assert all("residual" not in stage.methods for stage in STAGES)
for stage in STAGES:
    for method in stage.methods.split(","):
        assert method_category(method) == stage.category, (stage.name, method, method_category(method), stage.category)


def stages_for_session(session: int) -> tuple[Stage, ...]:
    session = int(session)
    if session not in SESSION_TOTAL_RUNS:
        raise ValueError(f"Session must be one of 1..6, got {session}.")
    return tuple(stage for stage in STAGES if stage.session == session)


def payload() -> dict:
    return {
        "protocol": "minimal_paper_46_runs_v5_arc_reference_controls_six_sessions",
        "seed": 0,
        "split_seed": 0,
        "epochs": 30,
        "warmup_epochs": 3,
        "input_size": 224,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "augmentation": "strong",
        "category_runs": dict(CATEGORY_RUNS),
        "main_runs": MAIN_RUNS,
        "additional_trso_ablations": len(ABLATIONS),
        "total_runs": TOTAL_RUNS,
        "session_main_runs": dict(SESSION_MAIN_RUNS),
        "session_total_runs": dict(SESSION_TOTAL_RUNS),
        "stages": [asdict(stage) for stage in STAGES],
        "reference_control_policy": {
            "methods": ["full", "linear"],
            "reporting_group": "reference_control",
            "full_fine_tuning": "All backbone and task-head parameters are trainable.",
            "linear_probing": "The pretrained backbone is frozen; only the downstream task head is trainable.",
            "learning_rates": {"full": 1e-4, "linear": 1e-3},
            "shared_head_rule": "Within each session, the exact best linear-probe checkpoint initializes compatible PEFT and TRSO runs.",
        },
        "proposal_ablation_suite": {
            "session": ABLATION_SESSION,
            "category": "proposal_ablation",
            "method": "trso",
            "linear_head_reused_from": "dtd_small_reference_controls/resnet18/linear",
            "trained_full_proposal_checkpoint_reused": False,
            "ablations": list(ABLATIONS),
            "separate_output_root": "trso_proposal_ablation",
        },
        "excluded_from_strict_baselines": {
            "reference_controls": ["full", "linear"],
            "paper_internal_ablation": ["convpass_attn"],
            "qualified_reimplementation_candidates": ["fact_tt", "fact_tk", "vqt", "spt_lora", "spt_adapter"],
            "transferred_controls": ["lora", "bitfit", "sidetune"],
            "engineering_controls": ["norm", "bias", "last_block"],
            "proposal": ["trso"],
        },
        "six_session_execution": [
            {"session": session, "training_runs": SESSION_TOTAL_RUNS[session],
             "stage_names": [stage.name for stage in stages_for_session(session)],
             "includes_trso_ablations": session == ABLATION_SESSION}
            for session in range(1, 7)
        ],
        "residual_adapter_removed": True,
        "single_seed_limitation": "Report fixed-seed results only; do not claim statistical significance.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/minimal_paper_46_protocol.json")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = payload()
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(asdict(STAGES[0]).keys()))
        writer.writeheader()
        for stage in STAGES:
            row = asdict(stage)
            row["dataset_args"] = json.dumps(row["dataset_args"], sort_keys=True)
            writer.writerow(row)
    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
