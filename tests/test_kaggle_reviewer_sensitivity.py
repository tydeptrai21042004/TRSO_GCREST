from __future__ import annotations

import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SENS_DIR = ROOT / "kaggle" / "reviewer_sensitivity"
if str(SENS_DIR) not in sys.path:
    sys.path.insert(0, str(SENS_DIR))
sp = importlib.import_module("sensitivity_protocol")
sr = importlib.import_module("sensitivity_session_runner")


def test_sensitivity_matrix_is_complete_and_runtime_safe():
    assert sp.total_group_count() == 72
    assert sp.total_run_count() == 180
    assert sp.SESSION_COUNT == 14
    seen = []
    for sid in range(1, sp.SESSION_COUNT + 1):
        assert sp.session_estimated_minutes_t4(sid) <= 480
        for group in sp.SESSIONS[sid]:
            seen.append(group.group_id)
    assert len(seen) == len(set(seen)) == 72


def test_reviewer_demands_are_present_in_public_sensitivity_plan():
    studies = {group.study for group in sp.groups()}
    assert {"reliability", "calibration", "batch_sensitivity", "mode_rules", "r_scale", "lora_rank_sweep"} <= studies
    coverage = sp.reviewer_coverage()
    for key in ("R1-C3", "R1-C5", "R2-C1", "R2-C2", "R2-C3", "R2-C5", "R3-C1", "R3-C2", "R3-C3", "R3-C4"):
        assert key in coverage


def test_exact_run_spec_selects_one_requested_variant(tmp_path):
    group = next(g for g in sp.groups() if g.study == "mode_rules" and g.variant == "harmonic" and g.setting_key == "dtd_resnet18")
    spec = sr.exact_run_spec(group, 1, data_dir=tmp_path / "data", run_root=tmp_path / "runs", manifest=tmp_path / "m.json")
    assert spec.name == "harmonic"
    assert spec.parameters["seed"] == 1
    assert spec.parameters["trso_mode_count_rule"] == "harmonic"


def test_public_kaggle_cells_clone_only_correct_default_repo():
    kaggle = ROOT / "kaggle"
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in kaggle.rglob("*.py"))
    assert "tydeptrai21042004/trso_adapter" not in text
    main_wrappers = sorted((kaggle / "reviewer_matrix").glob("TRSO_Reviewer_Matrix_Session_*_OneCell.py"))
    sensitivity_wrappers = sorted((kaggle / "reviewer_sensitivity").glob("TRSO_Reviewer_Sensitivity_Session_*_OneCell.py"))
    assert len(main_wrappers) == 18
    assert len(sensitivity_wrappers) == 14
    for path in main_wrappers + sensitivity_wrappers:
        content = path.read_text(encoding="utf-8")
        assert "https://github.com/tydeptrai21042004/TRSO_GCREST.git" in content
        assert "git\", \"clone" in content or "git\", \"clone" in content
