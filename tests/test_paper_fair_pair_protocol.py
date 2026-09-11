from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

from baseline_recipes import paper_method_defaults, paper_outer_trials, recipe_for
from main import get_args_parser
from tools.experiment_grid import write_manifest
from tools.run_fair_suite import build_suite
from tools.run_paper_fair_pairs import build_pair_specs
from tools.verify_paper_pairs import OUTER_KEYS, verify_manifest


def _pair_args(tmp_path: Path, *, baselines: str = "adaptformer") -> Namespace:
    return Namespace(
        dataset="fake", task="single_label", data_path=str(tmp_path / "data"), download="no",
        dataset_args_json="{}", backbone="vit_b_16@torchvision", baselines=baselines,
        seeds="0,1", split_seed=2026, weights="none", pretrained=False,
        input_size=32, batch_size=4, num_workers=0, search_mode="compact",
        use_paper_batch_size=True, require_verified_outer_recipe=False,
        overrides_json="{}", method_overrides_json="{}",
        fallback_optimizer="adamw", fallback_epochs=2, fallback_lr=1e-3,
        fallback_weight_decay=1e-4, fallback_warmup_epochs=1, fallback_min_lr=1e-6,
        augmentation="basic", output_root=str(tmp_path / "outputs"),
        manifest=str(tmp_path / "pairs.json"), device="cpu", gpu_ids="0",
        parallel_runs=1, execute=False, max_runs=0, profile_efficiency=False,
        measure_eval_latency=False, allow_val_as_test=False,
    )


def _controlled_args(tmp_path: Path) -> Namespace:
    return Namespace(
        dataset="fake", task="single_label", data_path=str(tmp_path / "data"), download=False,
        dataset_args_json="{}", output_root=str(tmp_path / "controlled"),
        manifest=str(tmp_path / "controlled.json"), seeds="0", split_seed=2026,
        epochs=2, batch_size=4, num_workers=0, input_size=32,
        backbones="vit_b_16@torchvision", methods="adaptformer,trso", external_head_manifests="",
        head_init_policy="random", trso_ablation="full", allow_nonpaper_controls=False,
        allow_paper_ablations=False, allow_unverified_paper_reimplementations=False,
        peft_lr=1e-3, full_lr=1e-4, linear_lr=1e-3, weight_decay=1e-4,
        warmup_epochs=1, min_lr=1e-6, optimizer="adamw", augmentation="basic",
        peft_head_lr_scale=1.0, device="cpu", gpu_ids="0", parallel_runs=1,
        execute=False, max_runs=0, profile_efficiency=False, measure_eval_latency=False,
        peft_freeze_head=False, allow_val_as_test=False, weights="none", pretrained=False,
    )


def test_adaptformer_recipe_uses_official_image_defaults():
    recipe = recipe_for("adaptformer")
    assert recipe is not None
    defaults = paper_method_defaults("adaptformer")
    assert defaults["adaptformer_dim"] == 64
    trials = paper_outer_trials("adaptformer", batch_size=512)
    assert len(trials) == 1
    row = trials[0]
    assert row["optimizer"] == "sgd"
    assert row["epochs"] == 100
    assert row["lr"] == 0.2  # 0.1 * 512 / 256
    assert row["weight_decay"] == 0.0
    assert row["warmup_epochs"] == 20


def test_vpt_recipe_encodes_official_lr_wd_search_and_batch_scaling():
    full = paper_outer_trials("vpt_deep", batch_size=32, search_mode="full")
    assert len(full) == 40
    assert {row["optimizer"] for row in full} == {"sgd"}
    assert {row["epochs"] for row in full} == {30}
    assert max(row["paper_base_lr"] for row in full) == 50.0
    assert min(row["paper_base_lr"] for row in full) == 0.05
    assert max(row["lr"] for row in full) == 50.0 * 32 / 256


def test_controlled_main_table_has_fresh_head_and_paper_structure(tmp_path):
    args = _controlled_args(tmp_path)
    _, _, heads, comparisons, _, _ = build_suite(args)
    assert heads == []
    assert {row.parameters["tuning_method"] for row in comparisons} == {"adaptformer", "trso"}
    adapt = next(row.parameters for row in comparisons if row.parameters["tuning_method"] == "adaptformer")
    trso = next(row.parameters for row in comparisons if row.parameters["tuning_method"] == "trso")
    assert adapt["adaptformer_dim"] == 64
    assert "head_from" not in adapt and "head_from" not in trso
    assert adapt["head_init_policy"] == trso["head_init_policy"] == "random"
    assert adapt["peft_head_lr_scale"] == trso["peft_head_lr_scale"] == 1.0


def test_paper_pair_runner_gives_baseline_and_trso_identical_outer_recipe(tmp_path):
    args = _pair_args(tmp_path, baselines="adaptformer")
    _, _, _, _, specs, audit, _ = build_pair_specs(args)
    assert audit[0]["status"] == "scheduled"
    assert audit[0]["recipe_fidelity"] == "exact_official_default"
    # 1 fixed paper trial * 2 seeds * (baseline + TRSO)
    assert len(specs) == 4
    for seed in (0, 1):
        rows = [s.parameters for s in specs if s.parameters["seed"] == seed]
        assert len(rows) == 2
        baseline = next(row for row in rows if row["tuning_method"] == "adaptformer")
        trso = next(row for row in rows if row["tuning_method"] == "trso")
        for key in OUTER_KEYS:
            assert baseline.get(key) == trso.get(key), key
        assert "head_from" not in baseline and "head_from" not in trso
        assert baseline["adaptformer_dim"] == 64


def test_partial_source_constraints_are_preserved_in_fallback(tmp_path):
    args = _pair_args(tmp_path, baselines="vqt")
    _, _, _, _, specs, audit, _ = build_pair_specs(args)
    assert audit[0]["recipe_fidelity"] == "matched_source_constraints_fallback_not_paper_exact"
    assert {s.parameters["optimizer"] for s in specs} == {"adam"}
    assert all(s.parameters["head_init_policy"] == "random" for s in specs)


def test_pair_verifier_accepts_generated_manifest(tmp_path):
    args = _pair_args(tmp_path, baselines="adaptformer")
    _, _, _, _, specs, _, _ = build_pair_specs(args)
    manifest, _ = write_manifest(specs, tmp_path / "pairs.json")
    report = verify_manifest(manifest)
    assert report["fair"], report["errors"]
    assert not report["errors"]


def test_method_override_supports_dataset_specific_official_vpt_config(tmp_path):
    args = _pair_args(tmp_path, baselines="vpt_deep")
    args.method_overrides_json = json.dumps({"vpt_deep": {"vpt_num_tokens": 50}})
    _, _, _, _, specs, audit, _ = build_pair_specs(args)
    baseline_rows = [s.parameters for s in specs if s.parameters["tuning_method"] == "vpt_deep"]
    assert baseline_rows and all(row["vpt_num_tokens"] == 50 for row in baseline_rows)
    assert audit[0]["method_override"] == {"vpt_num_tokens": 50}


def test_main_cli_accepts_source_optimizer_and_provenance_fields():
    args = get_args_parser().parse_args([
        "--optimizer", "adam", "--scheduler", "constant",
        "--protocol_name", "paper_recipe_paired_v1",
        "--paired_baseline", "vqt",
    ])
    assert args.optimizer == "adam"
    assert args.scheduler == "constant"
    assert args.adaptformer_dim == 64
    assert args.peft_head_lr_scale == 1.0
    assert args.paired_baseline == "vqt"
