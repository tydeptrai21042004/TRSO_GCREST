"""Audit a paper-recipe paired baseline-vs-TRSO manifest.

A valid pair has exactly two runs for every baseline/seed/HPO-trial block and
all outer experimental fields are identical.  Only method-internal parameters
and the tuning_method itself may differ.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

OUTER_KEYS = (
    "dataset", "task", "data_path", "backbone", "model_source", "weights",
    "pretrained", "input_size", "batch_size", "epochs", "optimizer",
    "scheduler", "lr", "weight_decay", "weight_decay_adapter",
    "warmup_epochs", "min_lr", "split_seed", "seed", "train_aug", "aa",
    "color_jitter", "mixup", "cutmix", "smoothing", "reprob",
    "head_init_policy", "peft_head_lr_scale", "paper_trial_index",
    "paper_search_mode", "paper_base_lr", "paired_baseline", "recipe_source",
    "recipe_fidelity",
)


def _load(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_manifest(path: str | Path) -> dict[str, Any]:
    rows = _load(path)
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    errors: list[str] = []
    warnings: list[str] = []

    for row in rows:
        p = row.get("parameters", {})
        baseline = str(p.get("paired_baseline", ""))
        if not baseline:
            errors.append(f"{row.get('name')}: missing paired_baseline")
            continue
        key = (baseline, int(p.get("seed", 0)), int(p.get("paper_trial_index", -1)))
        groups[key].append(p)

    group_reports = []
    for key, items in sorted(groups.items()):
        baseline, seed, trial = key
        baseline_rows = [p for p in items if p.get("tuning_method") == baseline]
        trso_rows = [p for p in items if p.get("tuning_method") == "trso"]
        if len(baseline_rows) != 1 or len(trso_rows) != 1:
            errors.append(
                f"{key}: expected exactly one {baseline} and one trso run; "
                f"got baseline={len(baseline_rows)}, trso={len(trso_rows)}"
            )
            continue
        a, b = baseline_rows[0], trso_rows[0]
        mismatches = {}
        for field in OUTER_KEYS:
            av = a.get(field)
            bv = b.get(field)
            if av != bv:
                mismatches[field] = {"baseline": av, "trso": bv}
        if mismatches:
            errors.append(f"{key}: paired outer fields differ: {mismatches}")
        for label, p in ((baseline, a), ("trso", b)):
            if bool(p.get("fair_protocol", False)):
                errors.append(f"{key}/{label}: paper-paired run must not enable fair_protocol overrides")
            if bool(p.get("paper_hparams", False)):
                errors.append(f"{key}/{label}: paper_hparams legacy override must be disabled")
            if str(p.get("head_init_policy", "")) != "random":
                errors.append(f"{key}/{label}: paired protocol requires random fresh-head policy")
            if p.get("head_from"):
                errors.append(f"{key}/{label}: paired protocol must not load a linear-probe head")
            if float(p.get("peft_head_lr_scale", 1.0)) != 1.0:
                errors.append(f"{key}/{label}: paired protocol requires head LR scale 1.0")
        fidelity = str(a.get("recipe_fidelity", ""))
        if "not_paper_exact" in fidelity:
            warnings.append(f"{key}: {fidelity}; report as matched fallback, not paper reproduction")
        group_reports.append({
            "baseline": baseline,
            "seed": seed,
            "paper_trial_index": trial,
            "recipe_fidelity": fidelity,
            "mismatches": mismatches,
        })

    return {
        "manifest": str(path),
        "paired_blocks": len(groups),
        "groups": group_reports,
        "errors": errors,
        "warnings": warnings,
        "fair": not errors,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="")
    args = p.parse_args()
    report = verify_manifest(args.manifest)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return 0 if report["fair"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
