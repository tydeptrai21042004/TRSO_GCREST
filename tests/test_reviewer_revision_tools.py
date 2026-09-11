from argparse import Namespace
from tools.run_reviewer_revision import build_revision_specs


def _args(tmp_path, study):
    return Namespace(
        study=study, dataset="fake", data_path="./data", download=False,
        dataset_args_json="{}", task="auto", backbone="vit_b_16",
        model_source="torchvision", weights="none", pretrained=False,
        seeds="0,1,2", partition_seeds="0,1", calibration_fractions="0.25,1.0",
        mode_rules="shannon,harmonic,geometric,arithmetic", r_scales="0.5,1,2",
        lora_ranks="1,2,4", epochs=1, batch_size=4, num_workers=0, input_size=32,
        optimizer="adamw", lr=1e-3, weight_decay=1e-4, warmup_epochs=0,
        min_lr=0.0, split_seed=0, augmentation="basic", device="cpu",
        output_root=str(tmp_path), manifest=str(tmp_path / "manifest.json"),
        execute=False, max_runs=0, profile_efficiency=False, final_test=False,
        allow_val_as_test=True, svd_oversampling=0, svd_power_iterations=2,
    )


def test_mode_rule_revision_grid(tmp_path):
    specs = build_revision_specs(_args(tmp_path, "mode_rules"))
    assert len(specs) == 4 * 3
    assert {s.parameters["trso_mode_count_rule"] for s in specs} == {"shannon", "harmonic", "geometric", "arithmetic"}
    assert all(s.parameters["tuning_method"] == "trso" for s in specs)


def test_calibration_revision_grid_contains_fraction_and_partition_variants(tmp_path):
    specs = build_revision_specs(_args(tmp_path, "calibration"))
    assert any(s.parameters["trso_calibration_fraction"] == 0.25 for s in specs)
    assert any(s.parameters["trso_partition_mode"] == "seeded_random" for s in specs)
    assert any(s.parameters["trso_partition_seed"] == 1 for s in specs)


def test_lora_rank_sweep_is_validation_grid_not_test_tuning(tmp_path):
    specs = build_revision_specs(_args(tmp_path, "lora_rank_sweep"))
    assert len(specs) == 3 * 3
    assert {s.parameters["lora_r"] for s in specs} == {1, 2, 4}
    assert all(s.parameters["tuning_method"] == "lora" for s in specs)
