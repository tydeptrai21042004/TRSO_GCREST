"""Robust Kaggle runner for one reviewer-matrix session.

Run through one of the ``TRSO_Reviewer_Matrix_Session_XX_OneCell.py`` wrappers.
The runner executes one seed at a time so completed work can always be archived,
records actual runtime, stops launching work before the 11h50m hard limit, and
creates a ZIP containing every completed result plus logs/manifests/state.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import zipfile
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from matrix_protocol import (  # noqa: E402
    EPOCHS,
    INPUT_SIZE,
    NO_NEW_RUN_SAFETY_MULTIPLIER,
    SEEDS,
    SESSION_COUNT,
    SESSION_HARD_LIMIT_MINUTES,
    SESSIONS,
    SPLIT_SEED,
    WARMUP_EPOCHS,
    ZIP_RESERVE_MINUTES,
    session_estimated_minutes_t4,
    session_payload,
)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except Exception:
        return "unknown"


def detect_gpu() -> tuple[str, float]:
    """Return GPU name and a conservative multiplier relative to a T4 estimate."""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Enable a Kaggle GPU accelerator before running this session.")
        name = torch.cuda.get_device_name(0)
    except ImportError as exc:
        raise RuntimeError("PyTorch is required before starting the Kaggle session.") from exc

    text = name.lower()
    if "a100" in text or "h100" in text:
        factor = 0.40
    elif "l4" in text:
        factor = 0.65
    elif "v100" in text or "a10" in text or "3090" in text or "4090" in text:
        factor = 0.60
    elif "p100" in text:
        factor = 0.90
    elif "t4" in text:
        factor = 1.00
    else:
        factor = 1.00
    return name, factor


def safe_extract(zip_path: Path, destination: Path, expected_top: str) -> bool:
    """Extract a previous session archive without allowing path traversal."""
    if not zip_path.is_file():
        return False
    extracted = False
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if not name.startswith(expected_top + "/"):
                continue
            target = (destination / name).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe archive member: {name}")
            zf.extract(info, destination)
            extracted = True
    return extracted


def find_resume_zip(session_id: int) -> Path | None:
    explicit = os.environ.get("TRSO_RESUME_ZIP", "").strip()
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        raise FileNotFoundError(f"TRSO_RESUME_ZIP does not exist: {path}")
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return None
    pattern = f"*reviewer*matrix*session*{session_id:02d}*results*.zip"
    matches = sorted(kaggle_input.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def load_state(path: Path, *, session_id: int, commit: str, gpu_name: str) -> dict[str, Any]:
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        previous_commit = str(state.get("git_commit", "unknown"))
        if previous_commit not in {"unknown", commit} and not env_bool("TRSO_ALLOW_COMMIT_MISMATCH", False):
            raise RuntimeError(
                "Resume archive was created from a different Git commit. Pin TRSO_GITHUB_COMMIT "
                f"to {previous_commit}, or set TRSO_ALLOW_COMMIT_MISMATCH=1 only if intentional."
            )
        return state
    return {
        "protocol": "reviewer_main_controlled_fair_v3_kaggle_sessions",
        "session": session_id,
        "git_commit": commit,
        "gpu_name": gpu_name,
        "started_at_utc": utc_now(),
        "last_updated_utc": utc_now(),
        "completed": {},
        "failed": {},
        "timed_out": {},
        "stopped_for_time_limit": False,
        "session_complete": False,
    }


def save_state(state: dict[str, Any], path: Path) -> None:
    state["last_updated_utc"] = utc_now()
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def rolling_runtime_scale(state: dict[str, Any], gpu_factor: float) -> float:
    ratios = []
    for row in state.get("completed", {}).values():
        base = float(row.get("estimated_minutes_t4", 0) or 0)
        actual = float(row.get("actual_minutes", 0) or 0)
        if base > 0 and actual > 0:
            ratios.append(actual / base)
    if len(ratios) >= 2:
        return max(0.45, min(2.25, statistics.median(ratios[-8:])))
    return gpu_factor


def run_id(exp, seed: int) -> str:
    return f"{exp.experiment_id}__seed{seed}"


def write_status_csv(state: dict[str, Any], path: Path) -> None:
    rows = []
    for status_name in ("completed", "failed", "timed_out"):
        for rid, row in state.get(status_name, {}).items():
            rows.append({"run_id": rid, "status": status_name, **row})
    fields = [
        "run_id", "status", "setting", "method", "seed", "estimated_minutes_t4", "actual_minutes",
        "started_at_utc", "finished_at_utc", "manifest", "result_dir", "log", "returncode", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["run_id"]):
            writer.writerow(row)


def build_command(exp, seed: int, *, data_dir: Path, run_root: Path, manifest: Path) -> list[str]:
    return [
        sys.executable, "-m", "tools.run_fair_suite",
        "--dataset", exp.dataset,
        "--task", exp.task,
        "--data_path", str(data_dir),
        "--download", "auto",
        "--dataset_args_json", json.dumps(exp.dataset_args, separators=(",", ":")),
        "--backbones", exp.backbone,
        "--methods", exp.method,
        "--seeds", str(seed),
        "--split_seed", str(SPLIT_SEED),
        "--epochs", str(EPOCHS),
        "--warmup_epochs", str(WARMUP_EPOCHS),
        "--batch_size", str(exp.batch_size),
        "--num_workers", os.environ.get("TRSO_NUM_WORKERS", "4"),
        "--input_size", str(INPUT_SIZE),
        "--optimizer", "adamw",
        "--augmentation", "strong",
        "--peft_lr", "1e-3",
        "--full_lr", "1e-4",
        "--linear_lr", "1e-3",
        "--weight_decay", "1e-4",
        "--min_lr", "1e-6",
        "--head_init_policy", "random",
        "--peft_head_lr_scale", "1.0",
        "--peft_freeze_head", "False",
        "--output_root", str(run_root),
        "--manifest", str(manifest),
        "--device", "cuda",
        "--gpu_ids", "0",
        "--parallel_runs", "1",
        "--profile_efficiency", "True",
        "--measure_eval_latency", "True",
        "--execute",
    ]




def run_logged_with_timeout(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, timeout_seconds: int) -> tuple[int, bool]:
    """Run a training command in its own process group and kill descendants on timeout."""
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            return int(proc.wait(timeout=timeout_seconds)), False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                proc.wait()
            return 124, True


def resolve_result_dir(manifest: Path) -> Path | None:
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(payload, list) and len(payload) == 1:
            path = Path(payload[0]["output_dir"])
            return path
    except Exception:
        return None
    return None


def is_result_complete(result_dir: Path | None) -> bool:
    if result_dir is None:
        return False
    return any((result_dir / name).is_file() for name in ("test_summary.json", "eval_summary.json"))


def prune_checkpoints(result_dir: Path | None) -> int:
    if result_dir is None or env_bool("TRSO_KEEP_CHECKPOINTS", False):
        return 0
    removed = 0
    for pattern in ("checkpoint*.pth", "checkpoint*.pt"):
        for path in result_dir.rglob(pattern):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flatten_scalars(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten JSON-compatible values for the wide final-metrics CSV.

    Nested mappings use dot-separated keys. Lists are preserved as compact JSON
    strings so no reported value is silently discarded. Per-epoch history is
    exported separately by :func:`collect_reported_metrics`.
    """
    flat: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_scalars(child, child_prefix))
        return flat
    if isinstance(value, list):
        flat[prefix] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return flat
    if value is None or isinstance(value, (str, int, float, bool)):
        flat[prefix] = value
    else:
        flat[prefix] = str(value)
    return flat


def _write_union_csv(path: Path, rows: list[dict[str, Any]], preferred: tuple[str, ...] = ()) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = set().union(*(row.keys() for row in rows))
    fields = [key for key in preferred if key in keys]
    fields.extend(sorted(keys - set(fields)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def seed_completeness_rows(session_id: int, state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    completed = state.get("completed", {})
    failed = state.get("failed", {})
    timed_out = state.get("timed_out", {})
    for exp in SESSIONS[int(session_id)]:
        for seed in SEEDS:
            rid = run_id(exp, seed)
            if rid in completed:
                status = "completed"
            elif rid in timed_out:
                status = "timed_out"
            elif rid in failed:
                status = "failed"
            else:
                status = "pending"
            rows.append({
                "experiment_id": exp.experiment_id,
                "setting": exp.setting_label,
                "method": exp.method_label,
                "seed": int(seed),
                "status": status,
                "run_id": rid,
            })
    return rows


def collect_reported_metrics(session_id: int, state: dict[str, Any], output: Path) -> None:
    """Export every JSON metric/report produced by each completed training run.

    The original result directories are already included verbatim in the session
    ZIP (except optional checkpoint pruning). This function additionally creates
    machine-readable aggregate files so no reported metric is hidden inside a
    nested run directory.
    """
    bundles: dict[str, Any] = {}
    final_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    backwards_summaries: dict[str, Any] = {}

    for rid, meta in sorted(state.get("completed", {}).items()):
        result_dir = Path(meta.get("result_dir", ""))
        artifact_payloads: dict[str, Any] = {}
        row: dict[str, Any] = {
            "run_id": rid,
            "setting": meta.get("setting", ""),
            "method": meta.get("method", ""),
            "seed": meta.get("seed", ""),
            "actual_minutes": meta.get("actual_minutes", ""),
            "estimated_minutes_t4": meta.get("estimated_minutes_t4", ""),
        }
        summary = None

        if result_dir.is_dir():
            for artifact in sorted(result_dir.rglob("*.json")):
                rel = artifact.relative_to(result_dir).as_posix()
                parse_status = "ok"
                try:
                    payload = json.loads(artifact.read_text(encoding="utf-8"))
                except Exception as exc:
                    payload = {"parse_error": f"{type(exc).__name__}: {exc}"}
                    parse_status = "parse_error"
                artifact_payloads[rel] = payload
                inventory_rows.append({
                    "run_id": rid,
                    "setting": meta.get("setting", ""),
                    "method": meta.get("method", ""),
                    "seed": meta.get("seed", ""),
                    "artifact": rel,
                    "bytes": artifact.stat().st_size,
                    "sha256": _sha256(artifact),
                    "parse_status": parse_status,
                })

                stem = Path(rel).stem
                if stem == "history" and isinstance(payload, list):
                    for epoch_record in payload:
                        if not isinstance(epoch_record, dict):
                            continue
                        epoch_row = {
                            "run_id": rid,
                            "setting": meta.get("setting", ""),
                            "method": meta.get("method", ""),
                            "seed": meta.get("seed", ""),
                        }
                        epoch_row.update(_flatten_scalars(epoch_record))
                        epoch_rows.append(epoch_row)
                else:
                    row.update({
                        f"{stem}.{key}": value
                        for key, value in _flatten_scalars(payload).items()
                        if key
                    })

                if rel in {"test_summary.json", "eval_summary.json"}:
                    summary = payload

        bundles[rid] = {"metadata": meta, "artifacts": artifact_payloads}
        backwards_summaries[rid] = {"metadata": meta, "summary": summary}
        final_rows.append(row)

    (output / "all_reported_metrics.json").write_text(
        json.dumps({
            "schema_version": 1,
            "description": "Exact parsed contents of every JSON report in every completed run directory.",
            "runs": bundles,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "completed_summaries.json").write_text(
        json.dumps(backwards_summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_union_csv(
        output / "all_final_metrics.csv",
        final_rows,
        preferred=("run_id", "setting", "method", "seed", "actual_minutes", "estimated_minutes_t4"),
    )
    _write_union_csv(
        output / "all_epoch_metrics.csv",
        epoch_rows,
        preferred=("run_id", "setting", "method", "seed", "epoch"),
    )
    _write_union_csv(
        output / "metrics_inventory.csv",
        inventory_rows,
        preferred=("run_id", "setting", "method", "seed", "artifact", "bytes", "sha256", "parse_status"),
    )

    completeness = seed_completeness_rows(session_id, state)
    _write_union_csv(
        output / "seed_completeness.csv",
        completeness,
        preferred=("experiment_id", "setting", "method", "seed", "status", "run_id"),
    )
    by_group: dict[str, set[int]] = {}
    for item in completeness:
        if item["status"] == "completed":
            by_group.setdefault(item["experiment_id"], set()).add(int(item["seed"]))
    required = set(SEEDS)
    complete_groups = sorted(key for key, seeds in by_group.items() if seeds == required)
    expected_groups = [exp.experiment_id for exp in SESSIONS[int(session_id)]]
    coverage = {
        "session": int(session_id),
        "required_seeds_per_experiment": list(SEEDS),
        "expected_experiment_groups": len(expected_groups),
        "expected_seed_runs": len(expected_groups) * len(SEEDS),
        "completed_seed_runs": sum(item["status"] == "completed" for item in completeness),
        "fully_completed_three_seed_groups": len(complete_groups),
        "complete_experiment_ids": complete_groups,
        "incomplete_experiment_ids": sorted(set(expected_groups) - set(complete_groups)),
        "all_groups_have_exactly_three_completed_seeds": len(complete_groups) == len(expected_groups),
        "json_metric_artifacts_collected": len(inventory_rows),
        "epoch_metric_rows_collected": len(epoch_rows),
    }
    (output / "metrics_coverage.json").write_text(
        json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def create_archive(output: Path, zip_path: Path, state: dict[str, Any]) -> Path:
    session_id = int(state.get("session", 0))
    collect_reported_metrics(session_id, state, output)
    note = output / "ARCHIVE_NOTE.txt"
    note.write_text(
        "This archive contains every completed run available when the session ended, plus logs, "
        "manifests, runtime state, protocol files, all JSON reports, all final metrics, every "
        "per-epoch history metric, metric-file hashes, and seed-completeness validation.\n"
        f"Session complete: {state.get('session_complete', False)}\n"
        f"Stopped for time limit: {state.get('stopped_for_time_limit', False)}\n"
        "By default checkpoint .pth/.pt files are removed after a successful run to keep Kaggle "
        "archives manageable. Set TRSO_KEEP_CHECKPOINTS=1 before running to retain them.\n",
        encoding="utf-8",
    )
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(output.parent))
    return zip_path


def print_plan(session_id: int, gpu_name: str, scale: float) -> None:
    print("=" * 100)
    print(f"TRSO reviewer matrix — Session {session_id:02d}/{SESSION_COUNT}")
    print(f"GPU: {gpu_name}")
    print(f"T4 reference estimate: {session_estimated_minutes_t4(session_id)/60:.2f} h")
    print(f"GPU-adjusted initial estimate: {session_estimated_minutes_t4(session_id)*scale/60:.2f} h")
    print(f"Hard limit: {SESSION_HARD_LIMIT_MINUTES//60}h{SESSION_HARD_LIMIT_MINUTES%60:02d}m; ZIP reserve: {ZIP_RESERVE_MINUTES} min")
    print("Runs are ordered fastest -> slowest inside this session. Each experiment uses seeds 0,1,2.")
    print("-" * 100)
    for order, exp in enumerate(SESSIONS[session_id], 1):
        print(
            f"{order:02d}. {exp.estimated_minutes_t4_per_seed:>3} min/seed T4  "
            f"{exp.setting_label:<44} {exp.method_label}"
        )
    print("=" * 100, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=int, required=True, choices=range(1, SESSION_COUNT + 1))
    parser.add_argument("--repo", default=str(ROOT))
    args = parser.parse_args()

    session_id = int(args.session)
    repo = Path(args.repo).resolve()
    if not (repo / "tools" / "run_fair_suite.py").is_file():
        raise FileNotFoundError(f"Not a TRSO repository: {repo}")

    work = Path("/kaggle/working") if Path("/kaggle/working").exists() else repo / ".kaggle_work"
    work.mkdir(parents=True, exist_ok=True)
    data_dir = work / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output = work / f"trso_reviewer_matrix_session_{session_id:02d}_results"
    output_name = output.name
    zip_path = work / f"{output_name}.zip"

    # Recover a previously downloaded partial archive when supplied as Kaggle input.
    if not output.exists():
        resume_zip = find_resume_zip(session_id)
        if resume_zip:
            print(f"Restoring previous session progress from: {resume_zip}", flush=True)
            safe_extract(resume_zip, work, output_name)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifests").mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    (output / "runs").mkdir(exist_ok=True)

    gpu_name, gpu_factor = detect_gpu()
    commit = git_commit(repo)
    state_path = output / "session_state.json"
    state = load_state(state_path, session_id=session_id, commit=commit, gpu_name=gpu_name)
    state["git_commit"] = commit
    state["gpu_name"] = gpu_name
    state["initial_gpu_runtime_factor_vs_t4"] = gpu_factor

    protocol = session_payload(session_id)
    protocol["resolved_git_commit"] = commit
    protocol["resolved_gpu_name"] = gpu_name
    protocol["resolved_gpu_runtime_factor_vs_t4"] = gpu_factor
    (output / "session_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print_plan(session_id, gpu_name, gpu_factor)
    save_state(state, state_path)
    write_status_csv(state, output / "run_status.csv")

    try:
        budget_start_epoch = float(os.environ.get("TRSO_SESSION_BOOTSTRAP_START_EPOCH", str(time.time())))
    except ValueError:
        budget_start_epoch = time.time()
    session_start = time.monotonic()
    hard_seconds = SESSION_HARD_LIMIT_MINUTES * 60
    zip_reserve_seconds = ZIP_RESERVE_MINUTES * 60
    stop_on_error = env_bool("TRSO_STOP_ON_ERROR", False)

    for exp in SESSIONS[session_id]:
        for seed in SEEDS:
            rid = run_id(exp, seed)
            if rid in state.get("completed", {}):
                print(f"[skip complete] {rid}", flush=True)
                continue

            elapsed = max(0.0, time.time() - budget_start_epoch)
            scale = rolling_runtime_scale(state, gpu_factor)
            expected_seconds = exp.estimated_minutes_t4_per_seed * scale * 60
            safe_expected_seconds = max(expected_seconds * NO_NEW_RUN_SAFETY_MULTIPLIER, expected_seconds + 10 * 60)
            remaining = hard_seconds - elapsed
            if remaining <= zip_reserve_seconds + safe_expected_seconds:
                print(
                    f"[time guard] Not starting {rid}: remaining={remaining/60:.1f} min, "
                    f"safe expected={safe_expected_seconds/60:.1f} min, ZIP reserve={ZIP_RESERVE_MINUTES} min.",
                    flush=True,
                )
                state["stopped_for_time_limit"] = True
                save_state(state, state_path)
                break

            manifest = output / "manifests" / f"{rid}.json"
            run_root = output / "runs" / rid
            log_path = output / "logs" / f"{rid}.log"
            cmd = build_command(exp, seed, data_dir=data_dir, run_root=run_root, manifest=manifest)
            timeout_seconds = max(60, int(remaining - zip_reserve_seconds))
            started = utc_now()
            print(
                f"\n[start] {rid}\n"
                f"        {exp.setting_label} | {exp.method_label} | seed={seed}\n"
                f"        ETA now ~{expected_seconds/60:.1f} min; hard subprocess timeout ~{timeout_seconds/60:.1f} min\n"
                f"        log: {log_path}",
                flush=True,
            )
            run_started = time.monotonic()
            returncode = None
            error = ""
            timed_out = False
            try:
                returncode, timed_out = run_logged_with_timeout(
                    cmd,
                    cwd=repo,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "0"},
                    log_path=log_path,
                    timeout_seconds=timeout_seconds,
                )
                if timed_out:
                    error = "Killed with its full process group by the session timeout guard so completed runs could be archived before 11h50m."
            except Exception as exc:  # keep later experiments runnable
                returncode = -1
                error = f"{type(exc).__name__}: {exc}"

            actual_minutes = (time.monotonic() - run_started) / 60.0
            result_dir = resolve_result_dir(manifest)
            complete = (returncode == 0) and is_result_complete(result_dir)
            row = {
                "setting": exp.setting_label,
                "method": exp.method_label,
                "seed": seed,
                "estimated_minutes_t4": exp.estimated_minutes_t4_per_seed,
                "actual_minutes": round(actual_minutes, 3),
                "started_at_utc": started,
                "finished_at_utc": utc_now(),
                "manifest": str(manifest),
                "result_dir": str(result_dir) if result_dir else "",
                "log": str(log_path),
                "returncode": returncode,
                "error": error,
            }
            if complete:
                removed = prune_checkpoints(result_dir)
                row["checkpoints_removed"] = removed
                state.setdefault("completed", {})[rid] = row
                state.get("failed", {}).pop(rid, None)
                state.get("timed_out", {}).pop(rid, None)
                print(f"[done] {rid} in {actual_minutes:.1f} min", flush=True)
            elif timed_out:
                state.setdefault("timed_out", {})[rid] = row
                print(f"[timeout] {rid}; preserving partial logs/output and archiving completed runs.", flush=True)
            else:
                if not error:
                    error = "Training command failed or did not produce test_summary.json/eval_summary.json."
                    row["error"] = error
                state.setdefault("failed", {})[rid] = row
                print(f"[failed] {rid}: {error or f'return code {returncode}'}", flush=True)
                if log_path.is_file():
                    try:
                        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
                        print("\n".join(tail), flush=True)
                    except Exception:
                        pass

            save_state(state, state_path)
            write_status_csv(state, output / "run_status.csv")

            if timed_out:
                state["stopped_for_time_limit"] = True
                save_state(state, state_path)
                break
            if not complete and stop_on_error:
                print("TRSO_STOP_ON_ERROR=1: stopping after the failed run.", flush=True)
                break

        # Persist an immediately downloadable ZIP after each fully completed
        # 3-seed experiment group. The final ZIP is regenerated again below.
        group_ids = {run_id(exp, seed) for seed in SEEDS}
        if group_ids.issubset(set(state.get("completed", {}))):
            create_archive(output, zip_path, state)
            print(f"[archive checkpoint] {exp.experiment_id}: all seeds {list(SEEDS)} saved -> {zip_path}", flush=True)

        if state.get("stopped_for_time_limit") or (stop_on_error and state.get("failed")):
            break

    expected_ids = {
        run_id(exp, seed)
        for exp in SESSIONS[session_id]
        for seed in SEEDS
    }
    completed_ids = set(state.get("completed", {}))
    state["session_complete"] = expected_ids.issubset(completed_ids)
    state["completed_count"] = len(completed_ids & expected_ids)
    state["expected_count"] = len(expected_ids)
    state["failed_count"] = len(set(state.get("failed", {})) & expected_ids)
    state["timed_out_count"] = len(set(state.get("timed_out", {})) & expected_ids)
    state["elapsed_minutes_training_runner"] = round((time.monotonic() - session_start) / 60.0, 3)
    state["elapsed_minutes_from_onecell_start"] = round(max(0.0, time.time() - budget_start_epoch) / 60.0, 3)
    save_state(state, state_path)
    write_status_csv(state, output / "run_status.csv")

    print("\nCreating session archive with every completed run...", flush=True)
    create_archive(output, zip_path, state)
    print(
        f"Session {session_id:02d}: completed {state['completed_count']}/{state['expected_count']} seed-runs; "
        f"failed={state['failed_count']}; timed_out={state['timed_out_count']}\n"
        f"Archive: {zip_path}",
        flush=True,
    )
    # A partial session is not a process failure: the ZIP/state is intentionally
    # resumable in another Kaggle session. Real run failures are still explicit.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
