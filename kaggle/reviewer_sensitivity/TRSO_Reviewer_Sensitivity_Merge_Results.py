"""Merge all reviewer-sensitivity session ZIPs and validate complete 180-run coverage."""
from __future__ import annotations
import argparse, csv, json, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from sensitivity_protocol import SESSION_COUNT, SESSIONS, total_run_count, reviewer_coverage


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/kaggle/input")
    p.add_argument("--output", default="/kaggle/working/trso_reviewer_sensitivity_merged")
    args = p.parse_args()
    source = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    states = []
    completed = {}
    commits = set()
    for sid in range(1, SESSION_COUNT + 1):
        pattern = f"*reviewer*sensitivity*session*{sid:02d}*results*.zip"
        matches = sorted(source.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            raise FileNotFoundError(f"Missing sensitivity session {sid:02d} ZIP under {source}")
        with zipfile.ZipFile(matches[0]) as zf:
            state_names = [n for n in zf.namelist() if n.endswith("/session_state.json")]
            if len(state_names) != 1:
                raise RuntimeError(f"Expected one session_state.json in {matches[0]}")
            state = json.loads(zf.read(state_names[0]))
            states.append(state)
            commits.add(str(state.get("git_commit", "unknown")))
            for job_id, row in state.get("completed", {}).items():
                completed[job_id] = row
            zf.extractall(out / f"session_{sid:02d}")
    expected = {g.job_id(seed) for sid in SESSIONS for g in SESSIONS[sid] for seed in g.seeds}
    missing = sorted(expected - set(completed))
    extra = sorted(set(completed) - expected)
    report = {
        "expected_training_runs": len(expected),
        "protocol_total_training_runs": total_run_count(),
        "completed_training_runs": len(set(completed) & expected),
        "missing_jobs": missing,
        "unexpected_jobs": extra,
        "git_commits": sorted(commits),
        "single_commit": len(commits) == 1,
        "reviewer_coverage": reviewer_coverage(),
        "complete": not missing and not extra and len(expected) == total_run_count(),
    }
    (out / "merge_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["complete"]:
        raise RuntimeError(f"Sensitivity merge incomplete: {len(missing)} missing, {len(extra)} unexpected")
    if len(commits) != 1:
        raise RuntimeError(f"Sensitivity sessions used multiple Git commits: {sorted(commits)}")
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
