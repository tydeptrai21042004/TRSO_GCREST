from __future__ import annotations

import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MATRIX_DIR = ROOT / "kaggle" / "reviewer_matrix"
if str(MATRIX_DIR) not in sys.path:
    sys.path.insert(0, str(MATRIX_DIR))

mp = importlib.import_module("matrix_protocol")
sr = importlib.import_module("session_runner")


def test_requested_matrix_is_exact_and_complete():
    expected = {
        "dtd_resnet50": {"full", "linear", "prompt", "conv", "piggyback", "trso"},
        "dtd_vit_b16": {"full", "linear", "ssf", "adaptformer", "repadapter", "arc", "vpt_shallow", "vpt_deep", "convpass", "fact_tt", "fact_tk", "vqt", "spt_lora", "spt_adapter", "trso"},
        "dtd_resnet18": {"full", "linear", "prompt", "trso"},
        "dtd_swin_t": {"full", "linear", "ssf", "trso"},
        "flowers_resnet18": {"full", "linear", "prompt", "trso"},
        "flowers_vit_b16": {"full", "linear", "ssf", "adaptformer", "repadapter", "arc", "vpt_shallow", "vpt_deep", "convpass", "fact_tt", "fact_tk", "vqt", "spt_lora", "spt_adapter", "trso"},
        "flowers_swin_t": {"full", "linear", "ssf", "trso"},
        "pet_resnet18": {"full", "linear", "prompt", "trso"},
        "pet_swin_t": {"full", "linear", "ssf", "trso"},
        "voc2007_mobilenetv3_small": {"full", "linear", "ml_decoder", "trso"},
        "pet_segmentation_lraspp_mobilenetv3_large": {"full", "linear", "segadapter", "trso"},
    }
    actual = {setting.key: set(setting.methods) for setting in mp.SETTINGS}
    assert actual == expected
    assert mp.total_group_count() == 68
    assert mp.SEEDS == (0, 1, 2)
    assert mp.total_run_count() == 204


def test_sessions_cover_every_experiment_once_and_are_fast_to_slow():
    assert mp.SESSION_COUNT == 18
    all_ids = []
    for sid in range(1, mp.SESSION_COUNT + 1):
        bucket = mp.SESSIONS[sid]
        estimates = [x.estimated_minutes_t4_per_seed for x in bucket]
        assert estimates == sorted(estimates)
        assert mp.session_estimated_minutes_t4(sid) <= 480
        all_ids.extend(x.experiment_id for x in bucket)
    assert len(all_ids) == 68
    assert len(set(all_ids)) == 68


def test_time_guard_and_archive_budget_are_explicit():
    assert mp.SESSION_HARD_LIMIT_MINUTES == 710
    assert mp.ZIP_RESERVE_MINUTES >= 10
    assert mp.NO_NEW_RUN_SAFETY_MULTIPLIER >= 1.25


def test_onecell_wrappers_exist_for_every_session():
    wrappers = sorted(MATRIX_DIR.glob("TRSO_Reviewer_Matrix_Session_*_OneCell.py"))
    assert len(wrappers) == mp.SESSION_COUNT
    for sid, path in enumerate(wrappers, 1):
        text = path.read_text(encoding="utf-8")
        assert f"SESSION_ID = {sid}" in text
        assert "https://github.com/tydeptrai21042004/TRSO_GCREST.git" in text
        assert "TRSO_GITHUB_COMMIT" in text
        assert "session_runner.py" in text


def test_controlled_fair_command_uses_fresh_heads_and_single_seed(tmp_path):
    exp = mp.SESSIONS[1][0]
    command = sr.build_command(
        exp,
        2,
        data_dir=tmp_path / "data",
        run_root=tmp_path / "runs",
        manifest=tmp_path / "manifest.json",
    )
    joined = " ".join(map(str, command))
    assert "--head_init_policy random" in joined
    assert "--peft_head_lr_scale 1.0" in joined
    assert "--seeds 2" in joined
    assert f"--methods {exp.method}" in joined
    assert "--epochs 30" in joined
    assert "--execute" in command


def test_every_session_payload_expands_every_experiment_to_exactly_three_seeds():
    for sid in range(1, mp.SESSION_COUNT + 1):
        payload = mp.session_payload(sid)
        assert payload["seeds"] == [0, 1, 2]
        assert payload["expanded_seed_runs"] == 3 * len(payload["experiment_groups"])
        for group in payload["experiment_groups"]:
            assert group["seeds"] == [0, 1, 2]


def test_metric_export_contains_all_json_reports_and_every_epoch(tmp_path):
    exp = mp.SESSIONS[1][0]
    output = tmp_path / "session"
    result_dir = output / "runs" / "fake"
    result_dir.mkdir(parents=True)
    (result_dir / "test_summary.json").write_text('{"acc1": 91.2, "ece": 0.03}', encoding="utf-8")
    (result_dir / "parameter_summary.json").write_text('{"trainable_params": 123, "total_params": 456}', encoding="utf-8")
    (result_dir / "timing_summary.json").write_text('{"gpu_hours": 0.25, "peak_train_memory_mb": 777}', encoding="utf-8")
    (result_dir / "history.json").write_text(
        '[{"epoch": 0, "train_loss": 2.0, "val_acc1": 80.0}, {"epoch": 1, "train_loss": 1.0, "val_acc1": 90.0}]',
        encoding="utf-8",
    )
    rid = sr.run_id(exp, 0)
    state = {
        "session": 1,
        "completed": {
            rid: {
                "setting": exp.setting_label,
                "method": exp.method_label,
                "seed": 0,
                "actual_minutes": 1.2,
                "estimated_minutes_t4": exp.estimated_minutes_t4_per_seed,
                "result_dir": str(result_dir),
            }
        },
        "failed": {},
        "timed_out": {},
    }
    sr.collect_reported_metrics(1, state, output)
    bundle = (output / "all_reported_metrics.json").read_text(encoding="utf-8")
    assert "test_summary.json" in bundle
    assert "parameter_summary.json" in bundle
    assert "timing_summary.json" in bundle
    assert "history.json" in bundle
    final_csv = (output / "all_final_metrics.csv").read_text(encoding="utf-8")
    assert "test_summary.acc1" in final_csv
    assert "parameter_summary.trainable_params" in final_csv
    assert "timing_summary.gpu_hours" in final_csv
    epoch_csv = (output / "all_epoch_metrics.csv").read_text(encoding="utf-8")
    assert "train_loss" in epoch_csv
    assert "val_acc1" in epoch_csv
    assert epoch_csv.count("\n") == 3  # header + two epochs
    coverage = __import__("json").loads((output / "metrics_coverage.json").read_text(encoding="utf-8"))
    assert coverage["required_seeds_per_experiment"] == [0, 1, 2]
    assert coverage["completed_seed_runs"] == 1
    assert not coverage["all_groups_have_exactly_three_completed_seeds"]
