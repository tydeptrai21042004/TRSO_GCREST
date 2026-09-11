"""Reviewer-revision execution sessions.

Sessions 01-06 from :mod:`tools.minimal_paper_protocol` remain the submitted
46-run reproduction.  Sessions 07-17 mirror the canonical 204-run revised main
matrix in :mod:`tools.revision_full_protocol`.  Sessions 18-21 collect the
reviewer-requested ablation/sensitivity evidence without changing the proposal
default.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class RevisionSession:
    session: int
    title: str
    commands: tuple[tuple[str, ...], ...]


def fair(dataset, backbones, methods, seeds="0,1,2", batch_size="32", task="auto", dataset_args=None):
    command = (
        "tools/run_fair_suite.py", "--dataset", dataset, "--download", "auto",
        "--task", task, "--backbones", backbones, "--methods", methods,
        "--seeds", seeds, "--split_seed", "0", "--epochs", "30",
        "--batch_size", str(batch_size), "--input_size", "224",
        "--optimizer", "adamw", "--peft_lr", "1e-3", "--full_lr", "1e-4",
        "--linear_lr", "1e-3", "--weight_decay", "1e-4",
        "--warmup_epochs", "3", "--min_lr", "1e-6", "--augmentation", "strong",
    )
    if dataset_args:
        command += ("--dataset_args_json", json.dumps(dataset_args, separators=(",", ":")),)
    return command + ("--execute",)


def reviewer(study, dataset, backbone, extra=(), *, batch_size="32", task="auto"):
    return (
        "tools/run_reviewer_revision.py", "--study", study, "--dataset", dataset,
        "--download", "auto", "--task", task, "--backbone", backbone,
        "--seeds", "0,1,2", "--split_seed", "0", "--epochs", "30",
        "--batch_size", str(batch_size), "--input_size", "224", "--optimizer", "adamw",
        "--lr", "1e-3", "--weight_decay", "1e-4", "--warmup_epochs", "3",
        "--min_lr", "1e-6", "--execute", *extra,
    )


VIT_METHODS = "full,linear,ssf,adaptformer,repadapter,arc,vpt_shallow,vpt_deep,convpass,fact_tt,fact_tk,vqt,spt_lora,spt_adapter,trso"

SESSIONS = {
    7: RevisionSession(7, "DTD / ResNet-50 revised main table", (
        fair("dtd", "resnet50@torchvision", "full,linear,prompt,conv,piggyback,trso", batch_size="16", dataset_args={"dtd_partition": 1}),
    )),
    8: RevisionSession(8, "DTD / ViT-B/16 revised main table", (
        fair("dtd", "vit_b_16@torchvision", VIT_METHODS, batch_size="8", dataset_args={"dtd_partition": 1}),
    )),
    9: RevisionSession(9, "DTD small backbones revised main tables", (
        fair("dtd", "resnet18@torchvision", "full,linear,prompt,trso", batch_size="32", dataset_args={"dtd_partition": 1}),
        fair("dtd", "swin_t@torchvision", "full,linear,ssf,trso", batch_size="32", dataset_args={"dtd_partition": 1}),
    )),
    10: RevisionSession(10, "Flowers-102 / ResNet-18 revised main table", (
        fair("flowers102", "resnet18@torchvision", "full,linear,prompt,trso", batch_size="32"),
    )),
    11: RevisionSession(11, "Flowers-102 / ViT-B/16 revised main table", (
        fair("flowers102", "vit_b_16@torchvision", VIT_METHODS, batch_size="16"),
    )),
    12: RevisionSession(12, "Flowers-102 / Swin-T revised main table", (
        fair("flowers102", "swin_t@torchvision", "full,linear,ssf,trso", batch_size="32"),
    )),
    13: RevisionSession(13, "Oxford-IIIT Pet / ResNet-18 revised main table", (
        fair("oxfordiiitpet", "resnet18@torchvision", "full,linear,prompt,trso", batch_size="32"),
    )),
    14: RevisionSession(14, "Oxford-IIIT Pet / Swin-T revised main table", (
        fair("oxfordiiitpet", "swin_t@torchvision", "full,linear,ssf,trso", batch_size="32"),
    )),
    15: RevisionSession(15, "VOC2007 / MobileNetV3-S ML-Decoder main table", (
        fair("voc2007", "mobilenet_v3_small@torchvision", "full,linear,ml_decoder,trso", batch_size="32", task="multilabel"),
    )),
    16: RevisionSession(16, "Pet trimap / LR-ASPP MobileNetV3-L SegAdapter main table", (
        fair(
            "oxford_pet_segmentation", "lraspp_mobilenet_v3_large@torchvision",
            "full,linear,segadapter,trso", batch_size="8", task="semantic_segmentation",
            dataset_args={"segmentation_num_classes": 3, "segmentation_ignore_index": 255},
        ),
    )),
    17: RevisionSession(17, "Main-table protocol audit / no extra training", ()),
    18: RevisionSession(18, "Reliability-weighting ablations on CNNs and Transformer", (
        reviewer("reliability", "dtd", "resnet18", batch_size="32"),
        reviewer("reliability", "dtd", "resnet50", batch_size="16"),
        reviewer("reliability", "flowers102", "vit_b_16", batch_size="16"),
    )),
    19: RevisionSession(19, "Calibration amount / partition / calibration-batch sensitivity", (
        reviewer("calibration", "dtd", "resnet18", batch_size="32", extra=("--calibration_fractions", "0.25,0.5,1", "--partition_seeds", "0,1,2")),
        reviewer("calibration", "flowers102", "vit_b_16", batch_size="16", extra=("--calibration_fractions", "0.25,0.5,1", "--partition_seeds", "0,1,2")),
        reviewer("batch_sensitivity", "dtd", "resnet18", batch_size="32", extra=("--batch_sizes", "8,16,32,64")),
        reviewer("batch_sensitivity", "flowers102", "vit_b_16", batch_size="16", extra=("--batch_sizes", "4,8,16,32")),
    )),
    20: RevisionSession(20, "D0/D1 rule and retained-mode-count stress tests", (
        reviewer("mode_rules", "dtd", "resnet18", batch_size="32", extra=("--mode_rules", "shannon,harmonic,geometric,arithmetic")),
        reviewer("mode_rules", "flowers102", "vit_b_16", batch_size="16", extra=("--mode_rules", "shannon,harmonic,geometric,arithmetic")),
        reviewer("r_scale", "dtd", "resnet18", batch_size="32", extra=("--r_scales", "0.5,1,2")),
        reviewer("r_scale", "flowers102", "vit_b_16", batch_size="16", extra=("--r_scales", "0.5,1,2")),
    )),
    21: RevisionSession(21, "Matched-budget / oracle LoRA rank sweep", (
        reviewer("lora_rank_sweep", "dtd", "vit_b_16", batch_size="8", extra=("--lora_ranks", "1,2,4,8,16,32,64")),
        reviewer("lora_rank_sweep", "flowers102", "vit_b_16", batch_size="16", extra=("--lora_ranks", "1,2,4,8,16,32,64")),
    )),
}


def session_payload(session: int) -> dict:
    item = SESSIONS[int(session)]
    return {
        **asdict(item),
        "canonical_46_run_protocol": False,
        "paper_role": "revision",
        "proposal_default_unchanged": True,
        "main_revision_expected_runs": 204,
    }
