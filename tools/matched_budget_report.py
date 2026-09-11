"""Select matched-budget and oracle LoRA runs against a completed TRSO run.

Expected workflow:
1) run the proposal and a LoRA rank sweep on the same dataset/backbone/protocol;
2) point this script at the proposal directory and the LoRA sweep root;
3) it reports (a) the closest trainable-parameter match and (b) the best
   validation-selected LoRA run among the sweep, without touching test labels.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def _load(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric(payload: dict[str, Any]) -> tuple[str | None, float | None]:
    # Prefer common validation primary metrics. Never choose by test score.
    for key in ("acc1", "map", "miou", "mean_dice", "r2"):
        value = payload.get(key)
        if value is not None:
            return key, float(value)
    # Regression losses: lower is better; handled by name.
    for key in ("rmse", "mae", "loss"):
        value = payload.get(key)
        if value is not None:
            return key, float(value)
    return None, None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--proposal_dir", required=True)
    p.add_argument("--baseline_root", required=True)
    p.add_argument("--out_csv", default="matched_budget_lora.csv")
    p.add_argument("--budget_field", default="adapter_trainable_params",
                   choices=["adapter_trainable_params", "trainable_params"])
    args = p.parse_args()

    proposal_params = _load(os.path.join(args.proposal_dir, "parameter_summary.json"))
    target = proposal_params.get(args.budget_field)
    if target is None:
        raise RuntimeError(f"Proposal summary lacks {args.budget_field}")
    target = int(target)

    rows = []
    for params_path in glob.glob(os.path.join(args.baseline_root, "**", "parameter_summary.json"), recursive=True):
        run_dir = os.path.dirname(params_path)
        params = _load(params_path)
        run_args = _load(os.path.join(run_dir, "args.json"))
        if str(run_args.get("tuning_method", "")).lower() != "lora":
            continue
        budget = params.get(args.budget_field)
        if budget is None:
            continue
        val = _load(os.path.join(run_dir, "eval_summary.json"))
        test = _load(os.path.join(run_dir, "test_summary.json"))
        metric_name, val_metric = _metric(val)
        rows.append({
            "run_dir": run_dir,
            "seed": run_args.get("seed"),
            "lora_r": run_args.get("lora_r"),
            "budget": int(budget),
            "target_budget": target,
            "budget_abs_error": abs(int(budget) - target),
            "budget_relative_error": abs(int(budget) - target) / max(target, 1),
            "validation_metric": metric_name,
            "validation_value": val_metric,
            "test_value": test.get(metric_name) if metric_name else None,
        })
    if not rows:
        raise RuntimeError("No completed LoRA runs found")
    df = pd.DataFrame(rows)
    df["matched_budget"] = False
    for _, group in df.groupby("seed", dropna=False):
        idx = group["budget_abs_error"].idxmin()
        df.loc[idx, "matched_budget"] = True

    # Oracle = best validation configuration per seed. Test is only read after
    # validation-based selection, preventing test-set hyperparameter tuning.
    df["oracle_validation"] = False
    for _, group in df.groupby("seed", dropna=False):
        named = group.dropna(subset=["validation_value"])
        if named.empty:
            continue
        metric = str(named.iloc[0]["validation_metric"])
        idx = named["validation_value"].idxmin() if metric in {"rmse", "mae", "loss"} else named["validation_value"].idxmax()
        df.loc[idx, "oracle_validation"] = True

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["seed", "budget_abs_error", "lora_r"]).to_csv(out, index=False)
    print(f"Target proposal budget: {target}")
    print(f"Saved: {out}")
    print(df[df["matched_budget"] | df["oracle_validation"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
