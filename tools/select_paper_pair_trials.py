"""Validation-only hyperparameter selection for paper-recipe paired runs.

The script selects hyperparameters separately for each baseline and TRSO using
only validation metrics, then reads test metrics for the already-selected
configuration.  This prevents test-set tuning while keeping the HPO budget
identical on both sides of each paired comparison.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from task_registry import task_spec


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def select(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_baseline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        p = row.get("parameters", {})
        baseline = str(p.get("paired_baseline", ""))
        if baseline:
            by_baseline[baseline].append(row)

    comparisons = []
    missing = []
    for baseline, group_rows in sorted(by_baseline.items()):
        if not group_rows:
            continue
        task = str(group_rows[0]["parameters"].get("task", "single_label"))
        spec = task_spec(task)
        selected: dict[str, dict[str, Any]] = {}
        for method in (baseline, "trso"):
            per_trial: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in group_rows:
                p = row["parameters"]
                if p.get("tuning_method") != method:
                    continue
                run_dir = Path(row["output_dir"])
                val = _read(run_dir / "eval_summary.json")
                if spec.primary_metric not in val:
                    missing.append(str(run_dir / "eval_summary.json"))
                    continue
                per_trial[int(p.get("paper_trial_index", -1))].append({
                    "row": row,
                    "val": float(val[spec.primary_metric]),
                })
            candidates = []
            for trial, samples in sorted(per_trial.items()):
                if not samples:
                    continue
                candidates.append({
                    "trial": trial,
                    "validation_mean": mean(item["val"] for item in samples),
                    "validation_std": pstdev(item["val"] for item in samples) if len(samples) > 1 else 0.0,
                    "samples": samples,
                })
            if not candidates:
                continue
            key = (lambda item: item["validation_mean"])
            best = max(candidates, key=key) if spec.maximize else min(candidates, key=key)
            tests = []
            seed_rows = []
            for sample in best["samples"]:
                row = sample["row"]
                test = _read(Path(row["output_dir"]) / "test_summary.json")
                test_value = test.get(spec.primary_metric)
                if test_value is not None:
                    tests.append(float(test_value))
                seed_rows.append({
                    "seed": int(row["parameters"].get("seed", 0)),
                    "run_id": row.get("run_id"),
                    "output_dir": row["output_dir"],
                    "validation": sample["val"],
                    "test": float(test_value) if test_value is not None else None,
                })
            selected[method] = {
                "trial": best["trial"],
                "validation_mean": best["validation_mean"],
                "validation_std": best["validation_std"],
                "test_mean": mean(tests) if tests else None,
                "test_std": pstdev(tests) if len(tests) > 1 else (0.0 if tests else None),
                "seeds": seed_rows,
                "selection": "best mean validation metric; test read only after selection",
            }
        comparisons.append({
            "paired_baseline": baseline,
            "task": task,
            "primary_metric": spec.primary_metric,
            "maximize": spec.maximize,
            "baseline": selected.get(baseline),
            "trso": selected.get("trso"),
        })
    return {
        "manifest": str(manifest_path),
        "selection_policy": "independent validation-only selection from identical paired search grids",
        "comparisons": comparisons,
        "missing_validation_files": sorted(set(missing)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="")
    args = p.parse_args()
    report = select(args.manifest)
    output = Path(args.output) if args.output else Path(args.manifest).with_name(Path(args.manifest).stem + "_selected.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "paired_baseline", "primary_metric", "method", "selected_trial",
            "validation_mean", "validation_std", "test_mean", "test_std",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for comparison in report["comparisons"]:
            for method_key, method_name in (("baseline", comparison["paired_baseline"]), ("trso", "trso")):
                row = comparison.get(method_key)
                if not row:
                    continue
                writer.writerow({
                    "paired_baseline": comparison["paired_baseline"],
                    "primary_metric": comparison["primary_metric"],
                    "method": method_name,
                    "selected_trial": row["trial"],
                    "validation_mean": row["validation_mean"],
                    "validation_std": row["validation_std"],
                    "test_mean": row["test_mean"],
                    "test_std": row["test_std"],
                })
    print(f"Saved validation-selected paired results: {output} / {csv_path}")
    if report["missing_validation_files"]:
        print(f"Incomplete runs: {len(report['missing_validation_files'])} validation summaries missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
