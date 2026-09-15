SESSION_ID = 22
SESSION_TITLE = 'Energy-only / no-reliability full-core control'
EXPECTED_RUNS = 9

# Kaggle one-cell runner generated for the final targeted reviewer checks.
# Requirements: Kaggle GPU ON + Internet ON.
# This file clones the public repository, applies ONLY reviewer-experiment controls
# when needed, runs this session, aggregates every metric, and exports one result ZIP.

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

PUBLIC_REPO = os.environ.get(
    "TRSO_GITHUB_REPO",
    "https://github.com/tydeptrai21042004/TRSO_GCREST.git",
).strip() or "https://github.com/tydeptrai21042004/TRSO_GCREST.git"
GITHUB_REF = os.environ.get("TRSO_GITHUB_REF", "main").strip() or "main"
GITHUB_COMMIT_PIN = os.environ.get("TRSO_GITHUB_COMMIT", "").strip()

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
REPO = WORK / f"TRSO_FinalReviewer_S{SESSION_ID:02d}"
DATA = WORK / "trso_final_reviewer_data"
OUTPUT = WORK / f"trso_final_reviewer_session_{SESSION_ID:02d}_results"
RESULT_ZIP = WORK / f"trso_final_reviewer_session_{SESSION_ID:02d}_results.zip"

# Exact additive reviewer-only patch tested against the user's 2026-09-15 source ZIP.
# Default proposal remains: trso_ablation=full, geometric R, candidate cap=0,
# head_init_seed=-1.  The patch only exposes explicit non-default reviewer controls.
PATCH_SHA256 = "d9fb0abcff90d649a76d0973555d110c895cee9185d36978d423297b04b0d13a"
PATCH_ZLIB_B64 = "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"


def run(cmd, cwd=None, check=True):
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    if check and p.returncode:
        raise subprocess.CalledProcessError(p.returncode, cmd)
    return p


def clone_repo():
    if REPO.exists():
        shutil.rmtree(REPO)
    try:
        run([
            "git", "clone", "--depth", "1", "--single-branch",
            "--branch", GITHUB_REF, PUBLIC_REPO, REPO,
        ])
        if GITHUB_COMMIT_PIN:
            run(["git", "fetch", "--depth", "1", "origin", GITHUB_COMMIT_PIN], cwd=REPO)
            run(["git", "checkout", "--detach", GITHUB_COMMIT_PIN], cwd=REPO)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "GitHub clone failed. Turn Kaggle Internet ON and verify "
            f"{PUBLIC_REPO} @ {GITHUB_REF}."
        ) from exc
    required = [
        REPO / "main.py",
        REPO / "models/tuning_modules/mdl_tangent_core.py",
        REPO / "tools/run_reviewer_revision.py",
        REPO / "tools/aggregate_revision_results.py",
    ]
    missing = [str(x) for x in required if not x.is_file()]
    if missing:
        raise RuntimeError(f"Cloned repository has an unexpected layout; missing: {missing}")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def reviewer_controls_present():
    main = (REPO / "main.py").read_text(encoding="utf-8", errors="ignore")
    core = (REPO / "models/tuning_modules/mdl_tangent_core.py").read_text(encoding="utf-8", errors="ignore")
    runner = (REPO / "tools/run_reviewer_revision.py").read_text(encoding="utf-8", errors="ignore")
    protocol = REPO / "tools/final_reviewer_protocol.py"
    return (
        '"energy_only_core_matched"' in core
        and '"energy_only"' in core
        and '--trso_candidate_rank_cap' in main
        and '--head_init_seed' in main
        and '"fixed_cap_calibration"' in runner
        and '"head_init"' in runner
        and protocol.is_file()
    )


def apply_reviewer_patch():
    raw = zlib.decompress(base64.b64decode(PATCH_ZLIB_B64.encode("ascii")))
    actual = hashlib.sha256(raw).hexdigest()
    if actual != PATCH_SHA256:
        raise RuntimeError(f"Embedded patch checksum mismatch: {actual} != {PATCH_SHA256}")
    if reviewer_controls_present():
        print("Reviewer controls already present in cloned GitHub source; no patch needed.", flush=True)
        return "already_present"
    patch_file = WORK / f"trso_final_reviewer_s{SESSION_ID:02d}.patch"
    patch_file.write_bytes(raw)
    check = run(["git", "apply", "--check", str(patch_file)], cwd=REPO, check=False)
    if check.returncode != 0:
        raise RuntimeError(
            "The cloned GitHub source does not match the source version used to build "
            "these reviewer sessions, and the reviewer controls are not already present. "
            "Pin the matching commit with TRSO_GITHUB_COMMIT or update GitHub from the "
            "2026-09-15 source before running."
        )
    run(["git", "apply", str(patch_file)], cwd=REPO)
    run(["git", "diff", "--check"], cwd=REPO)
    if not reviewer_controls_present():
        raise RuntimeError("Reviewer patch applied but expected controls were not detected.")
    return "applied"


def install_runtime():
    run([
        sys.executable, "-m", "pip", "install", "-q",
        "timm>=0.9,<2", "pandas>=2", "scipy>=1.10",
        "scikit-learn>=1.3", "tqdm>=4.65", "medmnist>=3.0",
    ])
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator before running this session.")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)


def compact_results(root: Path):
    # Metrics/logs/manifests are preserved; large model checkpoints are excluded
    # from the downloadable result ZIP because they are not required for reviewer tables.
    skip_suffixes = {".pth", ".pt", ".ckpt"}
    if RESULT_ZIP.exists():
        RESULT_ZIP.unlink()
    with zipfile.ZipFile(RESULT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() in skip_suffixes:
                continue
            zf.write(path, path.relative_to(WORK))
    print(f"RESULT_ZIP={RESULT_ZIP}", flush=True)


CLONED_COMMIT = clone_repo()
PATCH_STATUS = apply_reviewer_patch()
install_runtime()

sys.path.insert(0, str(REPO))
from tools.final_reviewer_protocol import SESSIONS, session_payload

if SESSION_ID not in SESSIONS:
    raise RuntimeError(f"Session {SESSION_ID} is not defined by final_reviewer_protocol.py")
spec = SESSIONS[SESSION_ID]
if spec.expected_runs != EXPECTED_RUNS:
    raise RuntimeError(f"Expected {EXPECTED_RUNS} runs, protocol defines {spec.expected_runs}")

DATA.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)  # preserve completed runs on notebook restart

provenance = {
    "session": SESSION_ID,
    "title": SESSION_TITLE,
    "github_repo": PUBLIC_REPO,
    "github_ref": GITHUB_REF,
    "github_commit": CLONED_COMMIT,
    "requested_commit_pin": GITHUB_COMMIT_PIN,
    "reviewer_patch_sha256": PATCH_SHA256,
    "reviewer_patch_status": PATCH_STATUS,
    "proposal_default_unchanged": True,
    "expected_training_runs": EXPECTED_RUNS,
}
(OUTPUT / "source_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
(OUTPUT / "session_protocol.json").write_text(json.dumps(session_payload(SESSION_ID), indent=2), encoding="utf-8")

planned = 0
for index, command in enumerate(spec.commands, 1):
    manifest = OUTPUT / f"command_{index:02d}_manifest.json"
    cmd = [
        sys.executable, *command,
        "--data_path", str(DATA),
        "--output_root", str(OUTPUT / f"command_{index:02d}"),
        "--manifest", str(manifest),
    ]
    run(cmd, cwd=REPO)
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    planned += len(rows)

if planned != EXPECTED_RUNS:
    raise RuntimeError(f"Session planned {planned} runs, expected {EXPECTED_RUNS}")

summary_csv = OUTPUT / "final_reviewer_summary.csv"
run([
    sys.executable, "tools/aggregate_revision_results.py",
    "--root", str(OUTPUT), "--out_csv", str(summary_csv),
], cwd=REPO)

compact_results(OUTPUT)
print(json.dumps({
    "session": SESSION_ID,
    "title": SESSION_TITLE,
    "expected_runs": EXPECTED_RUNS,
    "github_commit": CLONED_COMMIT,
    "patch_status": PATCH_STATUS,
    "result_zip": str(RESULT_ZIP),
}, indent=2), flush=True)
