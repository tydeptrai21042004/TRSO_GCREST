"""Notebook-safe merger for reviewer-required sensitivity session ZIPs.

This file can be either executed as a .py script OR pasted directly into a
Kaggle/Jupyter cell. It does not use __file__ and does not import the protocol.
The expected job IDs are read from each archived session_protocol.json.
"""
from __future__ import annotations
import argparse, csv, json, zipfile
from pathlib import Path
from typing import Any

EXPECTED_PROTOCOL = "reviewer_sensitivity_required_v2_kaggle_sessions"
EXPECTED_SESSION_COUNT = 9
EXPECTED_TOTAL_RUNS = 105

def _zip_state(path: Path) -> tuple[int, float]:
    try:
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.endswith("/session_state.json")]
            if len(names) != 1: return (-1, path.stat().st_mtime)
            state = json.loads(zf.read(names[0]))
            return (len(state.get("completed", {})), path.stat().st_mtime)
    except Exception:
        return (-1, path.stat().st_mtime)

def _find_best_session_zip(source: Path, sid: int) -> Path:
    patterns = [
        f"*reviewer*required*session*{sid:02d}*results*.zip",
        f"*required*session*{sid:02d}*.zip",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(source.rglob(pattern))
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(f"Missing required-sensitivity session {sid:02d} ZIP under {source}")
    return max(matches, key=_zip_state)

def _read_unique_json(zf: zipfile.ZipFile, suffix: str) -> dict[str, Any]:
    names = [n for n in zf.namelist() if n.endswith(suffix)]
    if len(names) != 1:
        raise RuntimeError(f"Expected exactly one {suffix!r}, found {len(names)}")
    return json.loads(zf.read(names[0]))

def _safe_extract(zf: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for info in zf.infolist():
        target = (destination / info.filename).resolve()
        if target != destination and destination not in target.parents:
            raise RuntimeError(f"Unsafe ZIP member: {info.filename}")
    zf.extractall(destination)

def _write_union_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    preferred = ["session","job_id","status","group_id","study","setting","variant","seed"]
    fields = [x for x in preferred if x in fields] + [x for x in fields if x not in preferred]
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def merge(input_dir="/kaggle/input", output_dir="/kaggle/working/trso_reviewer_required_merged") -> dict[str, Any]:
    source = Path(input_dir); out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    completed: dict[str, Any] = {}; expected: set[str] = set(); commits: set[str] = set()
    protocols: set[str] = set(); session_rows: list[dict[str, Any]] = []; all_status_rows: list[dict[str, Any]] = []
    chosen: dict[int, str] = {}
    seen_expected: set[str] = set()

    for sid in range(1, EXPECTED_SESSION_COUNT + 1):
        path = _find_best_session_zip(source, sid); chosen[sid] = str(path)
        with zipfile.ZipFile(path) as zf:
            state = _read_unique_json(zf, "/session_state.json")
            protocol = _read_unique_json(zf, "/session_protocol.json")
            if int(protocol.get("session", -1)) != sid:
                raise RuntimeError(f"Session ZIP mismatch: expected {sid}, got {protocol.get('session')} in {path}")
            protocols.add(str(protocol.get("protocol", "unknown")))
            commits.add(str(state.get("git_commit", "unknown")))
            ids = {str(j) for g in protocol.get("logical_groups", []) for j in g.get("job_ids", [])}
            overlap = seen_expected & ids
            if overlap:
                raise RuntimeError(f"Duplicate expected job IDs across sessions: {sorted(overlap)[:5]}")
            seen_expected |= ids; expected |= ids
            for job_id, row in state.get("completed", {}).items():
                completed[job_id] = {"session": sid, **row}
            for status_name in ("completed","failed","timed_out"):
                for job_id, row in state.get(status_name, {}).items():
                    all_status_rows.append({"session":sid,"job_id":job_id,"status":status_name,**row})
            session_rows.append({
                "session": sid, "zip": str(path), "expected": len(ids),
                "completed": len(set(state.get("completed", {})) & ids),
                "failed": len(set(state.get("failed", {})) & ids),
                "timed_out": len(set(state.get("timed_out", {})) & ids),
                "git_commit": str(state.get("git_commit", "unknown")),
            })
            _safe_extract(zf, out / f"session_{sid:02d}")

    missing = sorted(expected - set(completed))
    extra = sorted(set(completed) - expected)
    protocol_ok = protocols == {EXPECTED_PROTOCOL}
    total_ok = len(expected) == EXPECTED_TOTAL_RUNS
    single_commit = len(commits) == 1
    report = {
        "expected_protocol": EXPECTED_PROTOCOL, "protocols_found": sorted(protocols), "protocol_ok": protocol_ok,
        "expected_sessions": EXPECTED_SESSION_COUNT, "expected_training_runs": len(expected),
        "declared_expected_training_runs": EXPECTED_TOTAL_RUNS, "total_count_ok": total_ok,
        "completed_training_runs": len(set(completed) & expected), "missing_jobs": missing,
        "unexpected_jobs": extra, "git_commits": sorted(commits), "single_commit": single_commit,
        "chosen_session_zips": chosen,
        "complete": (not missing and not extra and protocol_ok and total_ok and single_commit),
    }
    (out / "merge_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_union_csv(out / "session_summary.csv", session_rows)
    _write_union_csv(out / "all_run_status.csv", all_status_rows)
    if not report["complete"]:
        raise RuntimeError(
            f"Merge incomplete: {len(missing)} missing, {len(extra)} unexpected, "
            f"protocol_ok={protocol_ok}, total_ok={total_ok}, single_commit={single_commit}. "
            f"See {out/'merge_validation.json'}"
        )
    print(json.dumps(report, indent=2)); return report

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/kaggle/input")
    p.add_argument("--output", default="/kaggle/working/trso_reviewer_required_merged")
    a, _unknown = p.parse_known_args()  # safe inside notebooks with ipykernel argv
    merge(a.input, a.output); return 0

if __name__ == "__main__":
    raise SystemExit(main())
