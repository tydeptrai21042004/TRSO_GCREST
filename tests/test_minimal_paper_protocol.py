from pathlib import Path

from tools.minimal_paper_protocol import (
    ABLATIONS, ABLATION_SESSION, CATEGORY_RUNS, MAIN_RUNS,
    SESSION_MAIN_RUNS, SESSION_TOTAL_RUNS, STAGES, TOTAL_RUNS,
    payload, stages_for_session,
)
from models.model_support import method_category


ROOT = Path(__file__).resolve().parents[1]


def test_minimal_protocol_has_exactly_46_runs_and_no_residual_adapter():
    assert CATEGORY_RUNS == {"paper_baseline": 12, "reference_control": 20, "proposal": 10}
    assert MAIN_RUNS == 42
    assert len(ABLATIONS) == 4
    assert TOTAL_RUNS == 46
    assert all("residual" not in stage.methods.lower() for stage in STAGES)
    assert payload()["residual_adapter_removed"] is True


def test_protocol_categories_never_mix_methods():
    for stage in STAGES:
        assert stage.category in {"paper_baseline", "reference_control", "proposal"}
        assert all(method_category(method) == stage.category for method in stage.methods.split(","))
    assert payload()["proposal_ablation_suite"]["category"] == "proposal_ablation"
    assert payload()["excluded_from_strict_baselines"]["reference_controls"] == ["full", "linear"]


def test_minimal_protocol_covers_multiple_architectures_and_tasks():
    text = " ".join(stage.backbones for stage in STAGES)
    tasks = {stage.task for stage in STAGES}
    assert "resnet18" in text and "resnet50" in text
    assert "vit_b_16" in text and "swin_t" in text
    assert "mobilenet_v3_small" in text and "lraspp_mobilenet_v3_large" in text
    assert {"auto", "multilabel", "semantic_segmentation"}.issubset(tasks)


def test_protocol_is_split_into_exactly_six_self_contained_sessions():
    assert SESSION_MAIN_RUNS == {1: 6, 2: 10, 3: 8, 4: 6, 5: 6, 6: 6}
    assert SESSION_TOTAL_RUNS == {1: 6, 2: 10, 3: 12, 4: 6, 5: 6, 6: 6}
    assert ABLATION_SESSION == 3
    assert sum(SESSION_TOTAL_RUNS.values()) == TOTAL_RUNS
    for session in range(1, 7):
        stages = stages_for_session(session)
        assert stages
        assert any(stage.category == "reference_control" for stage in stages)
        for stage in stages:
            assert stage.session == session


def test_each_reference_stage_contains_corrected_full_and_linear_controls():
    references = [stage for stage in STAGES if stage.category == "reference_control"]
    assert sum(stage.expected_runs for stage in references) == 20
    assert all(stage.methods == "full,linear" for stage in references)
    assert payload()["reference_control_policy"]["learning_rates"] == {"full": 1e-4, "linear": 1e-3}


def test_kaggle_has_six_session_runners_and_merge_utility():
    kaggle = ROOT / "kaggle"
    scripts = sorted(kaggle.glob("TRSO_Paper_46_Session_*_OneCell.py"))
    assert len(scripts) == 6
    for session, script in enumerate(scripts, start=1):
        text = script.read_text(encoding="utf-8")
        assert f"SESSION_ID = {session}" in text
        assert "assert TOTAL_RUNS == 46" in text
        assert '"--full_lr", 1e-4' in text
        assert '"--linear_lr", 1e-3' in text
        assert '"--external_head_manifests"' in text
        assert "git", "clone" in text
        assert "https://github.com/tydeptrai21042004/trso_adapter.git" in text
        assert 'GITHUB_REF = os.environ.get("TRSO_GITHUB_REF", "main")' in text
        assert "CLONED_GITHUB_COMMIT = clone_release()" in text
        assert '"proposal_version": "global_cross_fitted_reproducibility_entropy_spectral_tangent_core"' in text
        assert "Upload the corrected TRSO" not in text
    assert (kaggle / "TRSO_Merge_6_Session_Results.py").is_file()
