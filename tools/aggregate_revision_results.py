"""Aggregate experiment outputs into raw and paper-ready mean/std CSV files."""

from __future__ import annotations

import argparse
import glob
import json
import os
import math
from itertools import combinations
from typing import Any, Dict

import pandas as pd
try:
    from scipy.stats import t as student_t
except Exception:
    student_t = None


def read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_dict(prefix: str, values: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def infer_metadata(run_dir: str, root: str) -> Dict[str, Any]:
    rel = os.path.relpath(run_dir, root)
    parts = rel.split(os.sep)
    meta: Dict[str, Any] = {"run_dir": run_dir}
    # Grid layout: root/suite/variant/seed_<seed>_<digest>.
    if len(parts) >= 3:
        meta.update(
            {
                "experiment_suite": parts[-3],
                "experiment_name": parts[-2],
                "run_leaf": parts[-1],
            }
        )
    return meta


def summarize_mdl_tangent_calibration(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict) or not payload:
        return {}
    records = [row for row in payload.get("records", []) if isinstance(row, dict)]
    selected = [row for row in records if int(row.get("rank", 0)) > 0]

    def mean_of(key: str):
        values = [float(row[key]) for row in selected if row.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "calibration_batches": payload.get("calibration_batches"),
        "calibration_examples": payload.get("calibration_examples"),
        "selected_tensors": payload.get("selected_tensors", len(selected)),
        "candidate_tensors": payload.get("candidate_tensors", len(records)),
        "skipped_tensors": payload.get("skipped_tensors"),
        "ablation": payload.get("ablation"),
        "adapter_parameters": payload.get("adapter_parameters"),
        "proposal_added_parameters": payload.get("proposal_added_parameters"),
        "zero_added_parameter": payload.get("zero_added_parameter"),
        "effective_update_coordinates": payload.get("effective_update_coordinates"),
        "selected_original_parameter_values": payload.get("selected_original_parameter_values"),
        "head_policy": payload.get("head_policy"),
        "head_trainable_parameters": payload.get("head_trainable_parameters"),
        "frozen_basis_values": payload.get("frozen_basis_values"),
        "transient_reference_values": payload.get("transient_reference_values"),
        "transient_projection_state_values": payload.get("transient_projection_state_values"),
        "diagonal_tensors": sum(row.get("core_mode") == "diagonal" for row in selected),
        "sparse_cross_tensors": sum(row.get("core_mode") == "sparse_cross_evidence" for row in selected),
        "dense_tensors": sum(row.get("core_mode") == "dense_chance_rescue" for row in selected),
        "calibration_mean_loss": payload.get("calibration_mean_loss"),
        "chance_reference_loss": payload.get("chance_reference_loss"),
        "dense_rescue_activated": payload.get("dense_rescue_activated"),
        "selection_rule": payload.get("selection_rule"),
        "mean_selected_rank": mean_of("rank"),
        "mean_core_parameters": mean_of("core_parameters"),
        "mean_captured_fraction": (
            sum(
                float(row.get("captured_mean_energy", 0.0))
                / max(float(row.get("total_gradient_energy", 0.0)), 1e-30)
                for row in selected
            ) / len(selected)
            if selected else None
        ),
        # Reviewer-facing allocation diagnostics.
        "D0": payload.get("global_numerical_support"),
        "D1": payload.get("global_shannon_effective_modes"),
        "R": payload.get("global_selected_modes"),
        "candidate_modes": payload.get("candidate_modes"),
        "mode_count_rule": payload.get("mode_count_rule"),
        "mode_count_rule_value": payload.get("global_mode_count_rule_value"),
        "r_scale": payload.get("r_scale"),
        "fixed_r": payload.get("fixed_r"),
        "rank_min": payload.get("rank_min"),
        "rank_median": payload.get("rank_median"),
        "rank_max": payload.get("rank_max"),
        "calibration_fraction": payload.get("calibration_fraction"),
        "calibration_max_batches": payload.get("calibration_max_batches"),
        "partition_mode": payload.get("partition_mode"),
        "partition_seed": payload.get("partition_seed"),
        "svd_oversampling": payload.get("svd_oversampling"),
        "svd_power_iterations": payload.get("svd_power_iterations"),
        "layer_ranks_json": json.dumps(payload.get("layer_ranks", {}), sort_keys=True),
        "participating_tensor_names_json": json.dumps(payload.get("participating_tensor_names", [])),
        "candidate_modes_by_tensor_json": json.dumps(payload.get("candidate_modes_by_tensor", {}), sort_keys=True),
        "calibration_fold_counts_json": json.dumps(payload.get("calibration_fold_counts", [])),
    }


def _numeric_columns(df: pd.DataFrame, excluded: set[str]) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        if column in excluded:
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().any():
            df[column] = converted
            columns.append(column)
    return columns


def _mean_std_ci95(values: pd.Series) -> tuple[float | None, float | None, float | None, int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    n = int(numeric.shape[0])
    if n == 0:
        return None, None, None, 0
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        # Small-n seed studies require Student-t rather than the large-sample
        # normal 1.96 multiplier (e.g. t_0.975,2 ~= 4.303 for three seeds).
        critical = float(student_t.ppf(0.975, df=n - 1)) if student_t is not None else (4.302652729911275 if n == 3 else 1.96)
        ci95 = float(critical * std / math.sqrt(n))
    else:
        ci95 = 0.0
    return mean, std, ci95, n


def _allocation_stability(group: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in ("mdl_R", "mdl_D0", "mdl_D1", "mdl_selected_tensors", "mdl_adapter_parameters",
                   "mdl_rank_min", "mdl_rank_median", "mdl_rank_max"):
        if column in group.columns:
            mean, std, ci95, n = _mean_std_ci95(group[column])
            result[f"{column}_mean"] = mean
            result[f"{column}_std"] = std
            result[f"{column}_ci95"] = ci95
            result[f"{column}_n"] = n

    sets = []
    if "mdl_participating_tensor_names_json" in group.columns:
        for raw in group["mdl_participating_tensor_names_json"].dropna():
            try:
                sets.append(set(json.loads(raw)))
            except Exception:
                pass
    jaccards = []
    for first, second in combinations(sets, 2):
        union = first | second
        jaccards.append(len(first & second) / len(union) if union else 1.0)
    if jaccards:
        result["selected_tensor_jaccard_mean"] = float(sum(jaccards) / len(jaccards))
        result["selected_tensor_jaccard_min"] = float(min(jaccards))
        result["selected_tensor_jaccard_pairs"] = len(jaccards)

    rank_maps = []
    if "mdl_layer_ranks_json" in group.columns:
        for raw in group["mdl_layer_ranks_json"].dropna():
            try:
                rank_maps.append({str(k): int(v) for k, v in json.loads(raw).items()})
            except Exception:
                pass
    if rank_maps:
        names = sorted(set().union(*(set(m) for m in rank_maps)))
        per_tensor_std = []
        for name in names:
            vals = [m.get(name, 0) for m in rank_maps]
            if len(vals) > 1:
                series = pd.Series(vals, dtype=float)
                per_tensor_std.append(float(series.std(ddof=1)))
        if per_tensor_std:
            result["layer_rank_std_mean"] = float(sum(per_tensor_std) / len(per_tensor_std))
            result["layer_rank_std_max"] = float(max(per_tensor_std))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs_ablation")
    parser.add_argument("--out_csv", type=str, default="experiment_summary.csv")
    args = parser.parse_args()

    run_dirs = sorted(
        set(os.path.dirname(path) for path in glob.glob(os.path.join(args.root, "**", "args.json"), recursive=True))
    )
    if not run_dirs:
        files = glob.glob(os.path.join(args.root, "**", "test_summary.json"), recursive=True)
        files += glob.glob(os.path.join(args.root, "**", "eval_summary.json"), recursive=True)
        run_dirs = sorted(set(os.path.dirname(path) for path in files))

    rows = []
    for run_dir in run_dirs:
        row = infer_metadata(run_dir, args.root)
        run_args = read_json(os.path.join(run_dir, "args.json"))
        test = read_json(os.path.join(run_dir, "test_summary.json"))
        val = read_json(os.path.join(run_dir, "eval_summary.json"))
        efficiency = read_json(os.path.join(run_dir, "efficiency_profile.json"))
        convergence = read_json(os.path.join(run_dir, "convergence_summary.json"))
        parameters = read_json(os.path.join(run_dir, "parameter_summary.json"))
        timing = read_json(os.path.join(run_dir, "timing_summary.json"))
        mdl_calibration = read_json(os.path.join(run_dir, "mdl_tangent_calibration.json"))

        for key in (
            "experiment_suite",
            "experiment_name",
            "experiment_run_id",
            "dataset",
            "task",
            "task_type",
            "backbone",
            "tuning_method",
            "seed",
        ):
            if key in run_args:
                row["method" if key == "tuning_method" else key] = run_args[key]
        row.update(flatten_dict("test", test))
        row.update(flatten_dict("eval", val))
        row.update(flatten_dict("eff", efficiency))
        row.update(flatten_dict("conv", convergence))
        row.update(flatten_dict("param", parameters))
        row.update(flatten_dict("time", timing))
        row.update(flatten_dict("mdl", summarize_mdl_tangent_calibration(mdl_calibration)))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No completed runs found under {args.root}")

    output_path = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved raw summary: {output_path}")

    identifiers = {
        "run_dir",
        "run_leaf",
        "experiment_suite",
        "experiment_name",
        "experiment_run_id",
        "dataset",
        "task",
        "task_type",
        "backbone",
        "method",
        "seed",
    }
    metric_columns = _numeric_columns(df, identifiers)
    group_columns = [
        column
        for column in ("experiment_suite", "experiment_name", "dataset", "task_type", "backbone", "method")
        if column in df.columns
    ]
    if not group_columns or not metric_columns:
        print("Raw results were saved, but no numeric grouped summary was available.")
        return

    summary = df.groupby(group_columns, dropna=False)[metric_columns].agg(["mean", "std", "count"]).reset_index()
    mean_std_path = os.path.splitext(output_path)[0] + "_mean_std.csv"
    summary.to_csv(mean_std_path, index=False)
    print(f"Saved mean/std summary: {mean_std_path}")

    # Reviewer-facing confidence intervals and allocation stability.
    ci_rows = []
    stability_rows = []
    for keys, group in df.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_columns, keys))
        for metric in metric_columns:
            mean, std, ci95, n = _mean_std_ci95(group[metric])
            if n:
                ci_rows.append({**base, "metric": metric, "mean": mean, "std": std, "ci95_half_width": ci95, "n": n})
        if any(column.startswith("mdl_") for column in group.columns):
            stability_rows.append({**base, **_allocation_stability(group)})
    ci_path = os.path.splitext(output_path)[0] + "_ci95.csv"
    pd.DataFrame(ci_rows).to_csv(ci_path, index=False)
    print(f"Saved 95% CI table: {ci_path}")
    if stability_rows:
        stability_path = os.path.splitext(output_path)[0] + "_allocation_stability.csv"
        pd.DataFrame(stability_rows).to_csv(stability_path, index=False)
        print(f"Saved allocation stability: {stability_path}")

    # Compact paper-facing table. Non-scalar diagnostics remain in the raw CSV
    # and per-run JSON, while this table emphasizes accuracy, robustness,
    # calibration, efficiency, and convergence.
    preferred_metrics = [
        "test_acc1", "test_acc5", "test_loss", "test_macro_precision",
        "test_macro_recall", "test_macro_f1", "test_weighted_f1",
        "test_balanced_accuracy", "test_macro_auroc_ovr", "test_macro_average_precision",
        "test_macro_auroc", "test_micro_auroc", "test_mcc", "test_cohen_kappa",
        "test_ece", "test_brier_score", "test_nll", "test_predictive_entropy",
        "test_map", "test_micro_precision", "test_micro_recall",
        "test_micro_f1", "test_subset_accuracy", "test_hamming_accuracy",
        "test_label_cardinality_error",
        "test_mae", "test_median_absolute_error", "test_rmse",
        "test_r2", "test_pearson", "test_spearman",
        "test_miou", "test_mean_dice", "test_pixel_accuracy",
        "test_mean_class_accuracy", "test_frequency_weighted_iou",
        "test_abs_rel", "test_sq_rel", "test_rmse_log", "test_silog",
        "test_delta1", "test_delta2", "test_delta3", "test_log10",
        "test_map_50", "test_map_75", "test_mar_1", "test_mar_10", "test_mar_100",
        "param_trainable_params", "param_total_params", "param_trainable_ratio",
        "param_proposal_added_parameters", "param_zero_added_parameter",
        "param_backbone_trainable_params", "param_mdl_adapter_parameters",
        "param_mdl_effective_update_coordinates", "param_mdl_selected_original_parameter_values",
        "param_mdl_head_trainable_parameters", "param_mdl_transient_basis_values",
        "param_mdl_transient_reference_values", "param_mdl_transient_projection_state_values",
        "param_mdl_transient_projection_state_megabytes_fp32",
        "param_mdl_deployed_extra_parameters", "param_mdl_deployed_total_params",
        "param_piggyback_deployed_mask_megabytes",
        "mdl_calibration_batches", "mdl_calibration_examples",
        "mdl_selected_tensors", "mdl_candidate_tensors", "mdl_skipped_tensors",
        "mdl_adapter_parameters", "mdl_proposal_added_parameters",
        "mdl_effective_update_coordinates", "mdl_selected_original_parameter_values",
        "mdl_head_trainable_parameters", "mdl_frozen_basis_values",
        "mdl_transient_reference_values", "mdl_transient_projection_state_values",
        "mdl_diagonal_tensors", "mdl_dense_tensors",
        "mdl_mean_selected_rank", "mdl_mean_core_parameters",
        "mdl_mean_captured_fraction", "mdl_D0", "mdl_D1", "mdl_R",
        "mdl_candidate_modes", "mdl_rank_min", "mdl_rank_median", "mdl_rank_max",
        "mdl_calibration_fraction", "mdl_partition_seed", "mdl_r_scale",
        "eff_flops_g", "eff_latency_ms_per_image", "eff_fps",
        "eff_peak_inference_memory_mb", "conv_best_val_acc1",
        "conv_best_val_map", "conv_best_val_mae", "conv_best_val_rmse",
        "conv_best_epoch", "conv_total_train_time_sec",
        "conv_proposal_calibration_time_sec", "conv_setup_and_calibration_time_sec",
        "conv_mean_epoch_time_sec", "conv_median_epoch_time_sec",
        "conv_mean_train_samples_per_second", "conv_median_train_samples_per_second",
        "conv_effective_training_samples_per_second", "conv_training_gpu_hours",
        "conv_time_to_best_sec", "conv_epochs_to_95pct_best",
        "time_proposal_calibration_time_sec", "time_profiling_time_sec",
        "time_setup_and_calibration_time_sec", "time_training_time_sec",
        "time_mean_epoch_time_sec", "time_median_epoch_time_sec",
        "time_mean_train_samples_per_second", "time_median_train_samples_per_second",
        "time_effective_training_samples_per_second", "time_time_to_best_sec",
        "time_peak_train_memory_mb", "time_final_evaluation_time_sec",
        "time_total_wall_time_sec", "time_training_gpu_hours", "time_gpu_hours",
        "time_wall_clock_samples_per_second",
    ]
    available = [metric for metric in preferred_metrics if metric in metric_columns]
    if available:
        compact = df.groupby(group_columns, dropna=False)[available].agg(["mean", "std", "count"]).reset_index()
        paper_path = os.path.splitext(output_path)[0] + "_paper_metrics.csv"
        compact.to_csv(paper_path, index=False)
        print(f"Saved paper metrics: {paper_path}")
        print(compact.head(30).to_string(index=False))
    else:
        print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
