from __future__ import annotations

from argparse import Namespace
import math
import pandas as pd

from tools.run_reviewer_revision import build_revision_specs
from tools.aggregate_revision_results import _allocation_stability, _mean_std_ci95, _spearman_rank_correlation


def _args(tmp_path):
    return Namespace(
        study="batch_sensitivity", dataset="fake", data_path="./data", download=False,
        dataset_args_json="{}", task="auto", backbone="vit_b_16",
        model_source="torchvision", weights="none", pretrained=False,
        seeds="0,1,2", partition_seeds="0,1", calibration_fractions="0.25,1.0",
        batch_sizes="8,16,32", mode_rules="shannon,harmonic,geometric,arithmetic",
        r_scales="0.5,1,2", lora_ranks="1,2,4", epochs=1, batch_size=4,
        num_workers=0, input_size=32, optimizer="adamw", lr=1e-3,
        weight_decay=1e-4, warmup_epochs=0, min_lr=0.0, split_seed=0,
        augmentation="basic", device="cpu", output_root=str(tmp_path),
        manifest=str(tmp_path / "manifest.json"), execute=False, max_runs=0,
        profile_efficiency=False, final_test=False, allow_val_as_test=True,
        svd_oversampling=0, svd_power_iterations=2,
    )


def test_batch_sensitivity_changes_calibration_batch_only(tmp_path):
    specs = build_revision_specs(_args(tmp_path))
    assert {s.parameters["trso_calibration_batch_size"] for s in specs} == {8, 16, 32}
    assert {s.parameters["batch_size"] for s in specs} == {4}
    assert {s.parameters["seed"] for s in specs} == {0, 1, 2}
    assert len(specs) == 9


def test_three_seed_ci_uses_student_t_not_large_sample_196():
    values = pd.Series([1.0, 2.0, 3.0])
    mean, std, ci, n = _mean_std_ci95(values)
    assert n == 3 and mean == 2.0
    assert math.isclose(std, 1.0, rel_tol=1e-12)
    normal_ci = 1.96 / math.sqrt(3)
    assert ci > normal_ci * 2.0
    assert math.isclose(ci, 4.302652729911275 / math.sqrt(3), rel_tol=1e-3)


def test_tensor_rank_spearman_handles_order_and_constant_cases():
    assert math.isclose(_spearman_rank_correlation([1, 2, 3], [3, 2, 1]), -1.0, rel_tol=1e-12)
    assert _spearman_rank_correlation([2, 2, 2], [2, 2, 2]) == 1.0
    assert _spearman_rank_correlation([2, 2, 2], [1, 2, 3]) is None


def test_allocation_stability_reports_tensor_level_diagnostics():
    group = pd.DataFrame({
        "mdl_R": [6, 6],
        "mdl_D0": [12, 12],
        "mdl_D1": [3.0, 3.0],
        "mdl_selected_tensors": [3, 3],
        "mdl_adapter_parameters": [20, 20],
        "mdl_rank_min": [1, 1],
        "mdl_rank_median": [2, 2],
        "mdl_rank_max": [3, 3],
        "mdl_participating_tensor_names_json": [
            '["a", "b", "c"]', '["a", "b", "c"]'
        ],
        "mdl_layer_ranks_json": [
            '{"a": 1, "b": 2, "c": 3}',
            '{"a": 3, "b": 2, "c": 1}',
        ],
    })
    stats = _allocation_stability(group)
    assert stats["selected_tensor_jaccard_mean"] == 1.0
    assert math.isclose(stats["tensor_rank_spearman_mean"], -1.0, rel_tol=1e-12)
    assert math.isclose(stats["tensor_rank_pair_mae_mean"], 4.0 / 3.0, rel_tol=1e-12)
