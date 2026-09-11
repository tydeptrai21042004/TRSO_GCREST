"""Revision experiment protocol (Sessions 07-14).

Sessions 01-06 in ``minimal_paper_protocol.py`` remain the canonical submitted
46-run reproduction.  This file only defines additional revision evidence.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class RevisionSession:
    session: int
    title: str
    commands: tuple[tuple[str, ...], ...]


def fair(dataset, backbones, methods, seeds="0,1,2", batch_size="32"):
    return ("tools/run_fair_suite.py", "--dataset", dataset, "--download", "auto", "--backbones", backbones,
            "--methods", methods, "--seeds", seeds, "--epochs", "30", "--batch_size", batch_size,
            "--input_size", "0", "--execute")

def reviewer(study, dataset, backbone, extra=()):
    return ("tools/run_reviewer_revision.py", "--study", study, "--dataset", dataset, "--download", "auto",
            "--backbone", backbone, "--seeds", "0,1,2", "--epochs", "30", "--execute", *extra)

SESSIONS = {
    7: RevisionSession(7, "DTD ResNet-50 multi-seed robustness", (fair("dtd", "resnet50@torchvision", "full,linear,trso", batch_size="16"),)),
    8: RevisionSession(8, "Flowers-102 ViT-B/16 multi-seed robustness", (fair("flowers102", "vit_b_16@torchvision", "full,linear,repadapter,trso", batch_size="8"),)),
    9: RevisionSession(9, "DTD ResNet-18 reliability/full-core ablation", (reviewer("reliability", "dtd", "resnet18", ("--batch_size", "32")),)),
    10: RevisionSession(10, "DTD ResNet-18 automatic allocation-rule ablation", (reviewer("mode_rules", "dtd", "resnet18", ("--mode_rules", "shannon,harmonic,geometric,arithmetic")),)),
    11: RevisionSession(11, "DTD ResNet-18 calibration and batch sensitivity", (
        reviewer("calibration", "dtd", "resnet18"), reviewer("batch_sensitivity", "dtd", "resnet18", ("--batch_sizes", "8,16,32,64")),
    )),
    12: RevisionSession(12, "EuroSAT cross-domain generalization", (fair("eurosat", "resnet18@torchvision", "full,linear,trso"),)),
    13: RevisionSession(13, "PCAM medical-domain generalization", (fair("pcam", "resnet18@torchvision", "full,linear,trso"),)),
    14: RevisionSession(14, "FGVC-Aircraft fine-grained generalization", (fair("fgvc_aircraft", "resnet18@torchvision,swin_t@torchvision", "full,linear,trso"),)),
}

def session_payload(session: int) -> dict:
    item = SESSIONS[int(session)]
    return {**asdict(item), "canonical_46_run_protocol": False, "paper_role": "revision", "proposal_default_unchanged": True}
