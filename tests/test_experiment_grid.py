from __future__ import annotations

import json
from pathlib import Path

from tools.experiment_grid import build_specs, expand_grid, one_factor_grid, write_manifest


def test_grid_and_one_factor_generation_are_deterministic():
    assert expand_grid({"fixed": 1}, {"b": [2, 3], "a": [4]}) == [
        {"fixed": 1, "a": 4, "b": 2},
        {"fixed": 1, "a": 4, "b": 3},
    ]
    rows = one_factor_grid({"x": 1, "y": 2}, {"x": [1, 3], "y": [2, 4]})
    assert rows == [{"x": 1, "y": 2}, {"x": 3, "y": 2}, {"x": 1, "y": 4}]


def test_manifest_has_unique_output_paths_and_commands(tmp_path):
    specs = build_specs(
        suite="unit",
        variants=[("a", {"seed": 0, "lr": 1e-3}), ("a", {"seed": 1, "lr": 1e-3})],
        common={"dataset": "fake", "tuning_method": "linear"},
        output_root=tmp_path,
    )
    assert len({spec.output_dir for spec in specs}) == 2
    assert all("--output_dir" in spec.command for spec in specs)
    assert all("--experiment_suite" in spec.command for spec in specs)
    assert all("--experiment_name" in spec.command for spec in specs)
    assert all(spec.parameters["experiment_run_id"] == spec.run_id for spec in specs)
    json_path, csv_path = write_manifest(specs, tmp_path / "manifest.json")
    assert json_path.exists() and csv_path.exists()
    payload = json.loads(json_path.read_text())
    assert len(payload) == 2 and all("command_text" in row for row in payload)


