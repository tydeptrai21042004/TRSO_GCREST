"""Robust Kaggle runner for one reviewer-sensitivity session.

The runner executes one exact training run at a time, archives each completed
logical group, records source provenance, enforces an 11h50m wall-clock guard,
and can resume a partial ZIP without mixing Git commits.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from required_protocol import (  # noqa: E402
    EPOCHS, INPUT_SIZE, NO_NEW_RUN_SAFETY_MULTIPLIER, SESSION_COUNT,
    SESSION_HARD_LIMIT_MINUTES, SESSIONS, SPLIT_SEED, WARMUP_EPOCHS,
    ZIP_RESERVE_MINUTES, reviewer_coverage, session_estimated_minutes_t4,
    session_payload,
)
from tools.run_reviewer_revision import build_revision_specs  # noqa: E402

PUBLIC_REPO = "https://github.com/tydeptrai21042004/TRSO_GCREST.git"


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except Exception:
        return "unknown"


def git_remote(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=repo, text=True).strip()
    except Exception:
        return "unknown"


def detect_gpu() -> tuple[str, float]:
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
    else:
        factor = 1.00
    return name, factor


def safe_extract(zip_path: Path, destination: Path, expected_top: str) -> bool:
    if not zip_path.is_file():
        return False
    destination = destination.resolve()
    extracted = False
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
    base = Path("/kaggle/input")
    if not base.exists():
        return None
    pattern = f"*reviewer*required*session*{session_id:02d}*results*.zip"
    matches = sorted(base.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def run_logged_with_timeout(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path,
                            timeout_seconds: int) -> tuple[int, bool]:
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
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


def _job_args(group, seed: int, *, data_dir: Path, run_root: Path, manifest: Path):
    # Build exactly the reviewer-run specification through the same audited
    # planner used outside Kaggle.  Narrow scalar lists are supplied here and the
    # exact expected variant is selected below, avoiding hidden multi-run commands.
    from argparse import Namespace
    controls = group.controls
    return Namespace(
        study=group.study,
        dataset=group.dataset,
        data_path=str(data_dir),
        download="auto",
        dataset_args_json=json.dumps(group.dataset_args, separators=(",", ":")),
        task=group.task,
        backbone=group.backbone,
        model_source="auto",
        weights="DEFAULT",
        pretrained=True,
        seeds=str(seed),
        partition_seeds=(str(controls.get("partition_seed", "")) if controls.get("partition_mode") == "seeded_random" else ""),
        calibration_fractions=str(controls.get("calibration_fraction", 1.0)),
        batch_sizes=str(controls.get("calibration_batch_size", group.batch_size)),
        mode_rules=str(controls.get("mode_rule", "geometric")),
        r_scales=str(controls.get("r_scale", 1.0)),
        lora_ranks=str(controls.get("lora_rank", 8)),
        epochs=EPOCHS,
        batch_size=group.batch_size,
        num_workers=int(os.environ.get("TRSO_NUM_WORKERS", "4")),
        input_size=INPUT_SIZE,
        optimizer="adamw",
        lr=1e-3,
        weight_decay=1e-4,
        warmup_epochs=WARMUP_EPOCHS,
        min_lr=1e-6,
        split_seed=SPLIT_SEED,
        augmentation="strong",
        device="cuda",
        output_root=str(run_root),
        manifest=str(manifest),
        execute=False,
        max_runs=0,
        profile_efficiency=True,
        final_test=True,
        allow_val_as_test=False,
        svd_oversampling=0,
        svd_power_iterations=2,
    )


def exact_run_spec(group, seed: int, *, data_dir: Path, run_root: Path, manifest: Path):
    specs = build_revision_specs(_job_args(group, seed, data_dir=data_dir, run_root=run_root, manifest=manifest))
    matches = [spec for spec in specs if spec.name == group.variant and int(spec.parameters.get("seed", -1)) == int(seed)]
    if len(matches) != 1:
        names = [(spec.name, spec.parameters.get("seed")) for spec in specs]
        raise RuntimeError(f"Expected exactly one {group.variant}/seed{seed} run, found {matches!r}; generated={names!r}")

    spec = matches[0]
    # Protocol tables intentionally use provenance-qualified names such as
    # ``resnet18@torchvision``.  The audited planner must split those into the
    # two CLI fields consumed by main.py.  Fail here, before any dataset/model
    # download, if a future change accidentally reintroduces the qualified
    # token into --backbone.
    resolved_backbone = str(spec.parameters.get("backbone", ""))
    resolved_source = str(spec.parameters.get("model_source", "auto"))
    if "@" in resolved_backbone:
        raise RuntimeError(
            "Reviewer scheduler produced an unresolved qualified backbone "
            f"{resolved_backbone!r}; expected separate --backbone and --model_source fields."
        )
    if "@" in str(group.backbone):
        expected_source = str(group.backbone).rsplit("@", 1)[1].strip().lower()
        if resolved_source.lower() != expected_source:
            raise RuntimeError(
                f"Reviewer scheduler resolved {group.backbone!r} to model_source={resolved_source!r}; "
                f"expected {expected_source!r}."
            )
    return spec


def result_complete(path: Path) -> bool:
    return path.is_dir() and any((path / name).is_file() for name in ("test_summary.json", "eval_summary.json"))


def prune_checkpoints(path: Path) -> int:
    if not path.is_dir() or env_bool("TRSO_KEEP_CHECKPOINTS", False):
        return 0
    removed = 0
    for pattern in ("*.pt", "*.pth", "*.ckpt"):
        for item in path.rglob(pattern):
            try:
                item.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def rolling_runtime_scale(state: dict[str, Any], gpu_factor: float) -> float:
    ratios = []
    for row in state.get("completed", {}).values():
        base = float(row.get("estimated_minutes_t4", 0) or 0)
        actual = float(row.get("actual_minutes", 0) or 0)
        if base > 0 and actual > 0:
            ratios.append(actual / base)
    return max(0.45, min(2.25, statistics.median(ratios[-8:]))) if len(ratios) >= 2 else gpu_factor


def save_state(state: dict[str, Any], path: Path) -> None:
    state["last_updated_at_utc"] = utc_now()
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def write_status_csv(state: dict[str, Any], path: Path) -> None:
    rows = []
    for status in ("completed", "failed", "timed_out"):
        for job_id, row in state.get(status, {}).items():
            rows.append({"job_id": job_id, "status": status, **row})
    fields = [
        "job_id", "status", "group_id", "study", "setting", "variant", "seed",
        "reviewer_requirements", "estimated_minutes_t4", "actual_minutes", "result_dir",
        "manifest", "log", "returncode", "error", "started_at_utc", "finished_at_utc",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda x: x["job_id"]):
            row = dict(row)
            if isinstance(row.get("reviewer_requirements"), (list, tuple)):
                row["reviewer_requirements"] = ",".join(row["reviewer_requirements"])
            writer.writerow(row)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(item, name))
    elif isinstance(value, (str, int, float, bool)) or value is None:
        out[prefix] = value
    return out


def _write_union_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = ["job_id", "group_id", "study", "setting", "variant", "seed", "source"]
    all_fields = set().union(*(row.keys() for row in rows))
    fields = [x for x in preferred if x in all_fields] + sorted(all_fields - set(preferred))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_metrics(state: dict[str, Any], output: Path) -> None:
    final_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    bundle: dict[str, Any] = {}
    for job_id, status in sorted(state.get("completed", {}).items()):
        result_dir = Path(status["result_dir"])
        base = {
            "job_id": job_id, "group_id": status["group_id"], "study": status["study"],
            "setting": status["setting"], "variant": status["variant"], "seed": status["seed"],
        }
        reports: dict[str, Any] = {}
        for path in sorted(result_dir.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rel = path.relative_to(result_dir).as_posix()
            reports[rel] = payload
            inventory.append({**base, "source": rel, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            if rel.endswith("history.json") and isinstance(payload, list):
                for index, epoch in enumerate(payload):
                    if isinstance(epoch, dict):
                        epoch_rows.append({**base, "source": rel, "epoch_index": index, **_flatten(epoch)})
            elif isinstance(payload, dict):
                final_rows.append({**base, "source": rel, **_flatten(payload)})
        bundle[job_id] = {"metadata": base, "reports": reports}
    (output / "all_reported_metrics.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    _write_union_csv(output / "all_final_metrics.csv", final_rows)
    _write_union_csv(output / "all_epoch_metrics.csv", epoch_rows)
    _write_union_csv(output / "metrics_inventory.csv", inventory)


def write_coverage(session_id: int, state: dict[str, Any], output: Path) -> None:
    expected = []
    for group in SESSIONS[session_id]:
        for seed in group.seeds:
            expected.append({
                "job_id": group.job_id(seed), "group_id": group.group_id, "study": group.study,
                "setting": group.setting_label, "variant": group.variant, "seed": seed,
                "reviewer_requirements": list(group.reviewer_requirements),
            })
    completed = set(state.get("completed", {}))
    rows = [{**row, "completed": row["job_id"] in completed} for row in expected]
    _write_union_csv(output / "seed_completeness.csv", rows)
    payload = {
        "session": session_id,
        "expected_training_runs": len(expected),
        "completed_training_runs": sum(bool(row["completed"]) for row in rows),
        "session_complete": all(bool(row["completed"]) for row in rows),
        "reviewer_coverage_definition": reviewer_coverage(),
    }
    (output / "reviewer_coverage.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def create_archive(output: Path, zip_path: Path, state: dict[str, Any], session_id: int) -> Path:
    collect_metrics(state, output)
    write_coverage(session_id, state, output)
    tmp = zip_path.with_suffix(".tmp.zip")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(output.parent))
    tmp.replace(zip_path)
    return zip_path


def print_plan(session_id: int, gpu_name: str, factor: float) -> None:
    print("=" * 104)
    print(f"Reviewer REQUIRED sensitivity session {session_id:02d}/{SESSION_COUNT} | GPU: {gpu_name} | initial T4 factor: {factor:.2f}")
    print(f"Planned T4 time: {session_estimated_minutes_t4(session_id)} min | hard guard: {SESSION_HARD_LIMIT_MINUTES} min")
    for group in SESSIONS[session_id]:
        print(
            f"- {group.estimated_group_minutes_t4:>3} min | {group.study:<18} | {group.setting_label:<26} | "
            f"{group.variant:<24} | seeds={list(group.seeds)} | {','.join(group.reviewer_requirements)}"
        )
    print("=" * 104, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=int, required=True, choices=range(1, SESSION_COUNT + 1))
    parser.add_argument("--repo", default=str(ROOT))
    args = parser.parse_args()
    session_id = int(args.session)
    repo = Path(args.repo).resolve()
    if not (repo / "tools" / "run_reviewer_revision.py").is_file():
        raise FileNotFoundError(f"Not a TRSO_GCREST repository: {repo}")

    work = Path("/kaggle/working") if Path("/kaggle/working").exists() else repo / ".kaggle_work"
    work.mkdir(parents=True, exist_ok=True)
    data_dir = work / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output = work / f"trso_reviewer_required_session_{session_id:02d}_results"
    zip_path = work / f"{output.name}.zip"
    if not output.exists():
        resume_zip = find_resume_zip(session_id)
        if resume_zip:
            print(f"Restoring partial session from {resume_zip}", flush=True)
            safe_extract(resume_zip, work, output.name)
    output.mkdir(parents=True, exist_ok=True)
    for sub in ("runs", "logs", "manifests"):
        (output / sub).mkdir(exist_ok=True)

    gpu_name, gpu_factor = detect_gpu()
    commit = git_commit(repo)
    remote = git_remote(repo)
    state_path = output / "session_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        previous = str(state.get("git_commit", "unknown"))
        if previous not in {"unknown", commit} and not env_bool("TRSO_ALLOW_COMMIT_MISMATCH", False):
            raise RuntimeError(f"Resume ZIP commit {previous} != current commit {commit}. Pin the original commit or start a fresh session.")
    else:
        state = {
            "protocol": "reviewer_sensitivity_required_v2_kaggle_sessions",
            "session": session_id, "started_at_utc": utc_now(), "completed": {}, "failed": {}, "timed_out": {},
            "stopped_for_time_limit": False,
        }
    state.update({"git_commit": commit, "git_remote": remote, "gpu_name": gpu_name})
    protocol = session_payload(session_id)
    protocol.update({"resolved_git_commit": commit, "resolved_git_remote": remote, "expected_public_repo": PUBLIC_REPO,
                     "resolved_gpu_name": gpu_name, "resolved_gpu_runtime_factor_vs_t4": gpu_factor})
    (output / "session_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    provenance = {
        "repository_expected": PUBLIC_REPO,
        "repository_resolved": remote,
        "git_commit": commit,
        "git_ref_requested": os.environ.get("TRSO_GITHUB_REF", "main"),
        "git_commit_requested": os.environ.get("TRSO_GITHUB_COMMIT", ""),
        "generated_at_utc": utc_now(),
    }
    (output / "source_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    save_state(state, state_path)
    write_status_csv(state, output / "run_status.csv")
    print_plan(session_id, gpu_name, gpu_factor)

    try:
        budget_start = float(os.environ.get("TRSO_SESSION_BOOTSTRAP_START_EPOCH", str(time.time())))
    except ValueError:
        budget_start = time.time()
    hard_seconds = SESSION_HARD_LIMIT_MINUTES * 60
    reserve_seconds = ZIP_RESERVE_MINUTES * 60
    stop_on_error = env_bool("TRSO_STOP_ON_ERROR", False)

    for group in SESSIONS[session_id]:
        for seed in group.seeds:
            job_id = group.job_id(seed)
            if job_id in state.get("completed", {}):
                print(f"[skip complete] {job_id}", flush=True)
                continue
            scale = rolling_runtime_scale(state, gpu_factor)
            expected_seconds = group.estimated_minutes_t4_per_run * scale * 60
            safe_seconds = max(expected_seconds * NO_NEW_RUN_SAFETY_MULTIPLIER, expected_seconds + 10 * 60)
            elapsed = max(0.0, time.time() - budget_start)
            remaining = hard_seconds - elapsed
            if remaining <= reserve_seconds + safe_seconds:
                print(f"[time guard] not starting {job_id}: remaining={remaining/60:.1f} min; safe ETA={safe_seconds/60:.1f} min", flush=True)
                state["stopped_for_time_limit"] = True
                save_state(state, state_path)
                break

            run_root = output / "runs" / job_id
            manifest = output / "manifests" / f"{job_id}.json"
            spec = exact_run_spec(group, seed, data_dir=data_dir, run_root=run_root, manifest=manifest)
            # Persist an exact, human-readable command manifest before execution.
            manifest_payload = {
                "job": {**asdict(group), "seed": seed, "job_id": job_id},
                "run_spec": {"suite": spec.suite, "name": spec.name, "parameters": spec.parameters,
                             "output_dir": spec.output_dir, "command": list(spec.command)},
            }
            manifest.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
            log_path = output / "logs" / f"{job_id}.log"
            started = utc_now()
            run_started = time.monotonic()
            timeout_seconds = max(60, int(remaining - reserve_seconds))
            print(f"\n[start] {job_id} | ETA~{expected_seconds/60:.1f} min | log={log_path}", flush=True)
            error = ""
            try:
                returncode, timed_out = run_logged_with_timeout(
                    list(spec.command), cwd=repo,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "0"},
                    log_path=log_path, timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                returncode, timed_out, error = -1, False, f"{type(exc).__name__}: {exc}"
            actual = (time.monotonic() - run_started) / 60.0
            result_dir = Path(spec.output_dir)
            complete = returncode == 0 and result_complete(result_dir)
            row = {
                "group_id": group.group_id, "study": group.study, "setting": group.setting_label,
                "variant": group.variant, "seed": seed, "reviewer_requirements": list(group.reviewer_requirements),
                "estimated_minutes_t4": group.estimated_minutes_t4_per_run, "actual_minutes": round(actual, 3),
                "result_dir": str(result_dir), "manifest": str(manifest), "log": str(log_path),
                "returncode": returncode, "error": error, "started_at_utc": started, "finished_at_utc": utc_now(),
            }
            if complete:
                row["checkpoints_removed"] = prune_checkpoints(result_dir)
                state.setdefault("completed", {})[job_id] = row
                state.get("failed", {}).pop(job_id, None)
                state.get("timed_out", {}).pop(job_id, None)
                print(f"[done] {job_id} in {actual:.1f} min", flush=True)
            elif timed_out:
                row["error"] = "Killed by session timeout guard; completed prior runs remain resumable."
                state.setdefault("timed_out", {})[job_id] = row
                state["stopped_for_time_limit"] = True
                print(f"[timeout] {job_id}", flush=True)
            else:
                if not row["error"]:
                    row["error"] = "Training failed or did not produce test_summary.json/eval_summary.json."
                state.setdefault("failed", {})[job_id] = row
                print(f"[failed] {job_id}: {row['error']}", flush=True)
                if log_path.is_file():
                    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
                    print("\n".join(tail), flush=True)
            save_state(state, state_path)
            write_status_csv(state, output / "run_status.csv")
            if timed_out or (stop_on_error and not complete):
                break

        group_ids = {group.job_id(seed) for seed in group.seeds}
        if group_ids.issubset(set(state.get("completed", {}))):
            create_archive(output, zip_path, state, session_id)
            print(f"[archive checkpoint] {group.group_id} complete -> {zip_path}", flush=True)
        if state.get("stopped_for_time_limit") or (stop_on_error and state.get("failed")):
            break

    expected_ids = {group.job_id(seed) for group in SESSIONS[session_id] for seed in group.seeds}
    completed = set(state.get("completed", {}))
    state["expected_count"] = len(expected_ids)
    state["completed_count"] = len(completed & expected_ids)
    state["failed_count"] = len(set(state.get("failed", {})) & expected_ids)
    state["timed_out_count"] = len(set(state.get("timed_out", {})) & expected_ids)
    state["session_complete"] = expected_ids.issubset(completed)
    save_state(state, state_path)
    write_status_csv(state, output / "run_status.csv")
    create_archive(output, zip_path, state, session_id)
    print(
        f"Session {session_id:02d}: completed {state['completed_count']}/{state['expected_count']}; "
        f"failed={state['failed_count']}; timed_out={state['timed_out_count']}\nArchive: {zip_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
