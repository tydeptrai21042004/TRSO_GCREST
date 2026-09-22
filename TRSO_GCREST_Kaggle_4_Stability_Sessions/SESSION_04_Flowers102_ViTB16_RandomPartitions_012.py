# Kaggle one-cell runner for TRSO/G-CREST editor-revision tensor-allocation stability.
# Requirements: Kaggle GPU ON + Internet ON.
# This script clones GitHub, verifies the revised stability-analysis code, plans exactly
# three runs for this session, executes them sequentially on one GPU, aggregates the
# results, and exports a compact result ZIP (checkpoints excluded).

SESSION_ID = 4
SESSION_TITLE = 'Flowers-102 / ViT-B/16 — stability across random partition seeds'
DATASET = 'flowers102'
BACKBONE = 'vit_b_16'
MODEL_SOURCE = "torchvision"
BATCH_SIZE = 16
PARTITION_MODE = 'seeded_random'   # "alternating" or "seeded_random"
EXPECTED_RUNS = 3

import json
import os
import shutil
import subprocess
import sys
import zipfile
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
REPO = WORK / f"TRSO_EditorStability_S{SESSION_ID:02d}"
DATA = WORK / f"trso_editor_stability_data_s{SESSION_ID:02d}"
OUTPUT = WORK / f"trso_editor_stability_session_{SESSION_ID:02d}_results"
MANIFEST = OUTPUT / "selected_manifest.json"
STATUS_JSON = OUTPUT / "execution_status.json"
SUMMARY_CSV = OUTPUT / "session_summary.csv"
RESULT_ZIP = WORK / f"trso_editor_stability_session_{SESSION_ID:02d}_results.zip"


def run(cmd, cwd=None, check=True):
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc


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
            "GitHub clone failed. Enable Kaggle Internet and verify TRSO_GITHUB_REPO/REF/COMMIT."
        ) from exc

    required = [
        REPO / "main.py",
        REPO / "tools/run_reviewer_revision.py",
        REPO / "tools/aggregate_revision_results.py",
        REPO / "models/tuning_modules/mdl_tangent_core.py",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(f"Unexpected repository layout; missing: {missing}")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    print("CLONED_COMMIT=", commit, flush=True)
    return commit


def install_runtime():
    # Kaggle already provides torch/torchvision. Avoid reinstalling them.
    run([
        sys.executable, "-m", "pip", "install", "-q",
        "timm>=0.9,<2", "pandas>=2", "scipy>=1.10",
        "scikit-learn>=1.3", "tqdm>=4.65", "fvcore>=0.1.5",
    ])
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator before executing this session.")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)


def verify_editor_revision_code():
    aggregator = (REPO / "tools/aggregate_revision_results.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    runner = (REPO / "tools/run_reviewer_revision.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    required_aggregator_markers = [
        "tensor_rank_spearman_mean",
        "tensor_rank_pair_mae_mean",
        "selected_tensor_jaccard_mean",
        "partition_stability",
    ]
    missing = [m for m in required_aggregator_markers if m not in aggregator]
    if missing:
        raise RuntimeError(
            "The cloned GitHub revision does not contain the editor-stability analysis patch. "
            f"Missing markers: {missing}. Push the revised files first or set TRSO_GITHUB_COMMIT "
            "to the commit containing the patch."
        )
    if '"calibration"' not in runner or '"seeded_random"' not in runner:
        raise RuntimeError("run_reviewer_revision.py does not contain the required calibration/partition study.")


def quick_preflight():
    tests = [
        REPO / "tests/test_mdl_tangent_core.py",
        REPO / "tests/test_reviewer_revision_protocol_fixes.py",
    ]
    existing = [str(p.relative_to(REPO)) for p in tests if p.is_file()]
    if existing:
        run([sys.executable, "-m", "pytest", "-q", *existing], cwd=REPO)


def build_selected_specs():
    sys.path.insert(0, str(REPO))
    from tools.run_reviewer_revision import get_parser, build_revision_specs

    parser = get_parser()
    argv = [
        "--study", "calibration",
        "--dataset", DATASET,
        "--data_path", str(DATA),
        "--download", "yes",
        "--task", "auto",
        "--backbone", BACKBONE,
        "--model_source", MODEL_SOURCE,
        "--weights", "DEFAULT",
        "--pretrained", "True",
        "--seeds", "0,1,2",
        "--partition_seeds", "0,1,2",
        "--calibration_fractions", "1.0",
        "--epochs", "30",
        "--batch_size", str(BATCH_SIZE),
        "--num_workers", "4",
        "--input_size", "224",
        "--optimizer", "adamw",
        "--lr", "1e-3",
        "--weight_decay", "1e-4",
        "--warmup_epochs", "3",
        "--min_lr", "1e-6",
        "--split_seed", "0",
        "--augmentation", "strong",
        "--device", "cuda",
        "--output_root", str(OUTPUT),
        "--manifest", str(MANIFEST),
        "--profile_efficiency", "True",
        "--final_test", "True",
        "--allow_val_as_test", "False",
        "--svd_oversampling", "0",
        "--svd_power_iterations", "2",
    ]
    args = parser.parse_args(argv)
    all_specs = build_revision_specs(args)
    selected = [
        spec for spec in all_specs
        if spec.parameters.get("trso_partition_mode") == PARTITION_MODE
    ]

    if len(selected) != EXPECTED_RUNS:
        raise RuntimeError(
            f"Session planned {len(selected)} selected runs, expected {EXPECTED_RUNS}."
        )

    if PARTITION_MODE == "alternating":
        seeds = sorted(int(spec.parameters["seed"]) for spec in selected)
        if seeds != [0, 1, 2]:
            raise RuntimeError(f"Alternating session has unexpected optimization seeds: {seeds}")
        if any(int(spec.parameters.get("trso_partition_seed", -1)) != 0 for spec in selected):
            raise RuntimeError("Alternating session unexpectedly varies the partition seed.")
    else:
        opt_seeds = sorted(set(int(spec.parameters["seed"]) for spec in selected))
        part_seeds = sorted(int(spec.parameters["trso_partition_seed"]) for spec in selected)
        if opt_seeds != [0] or part_seeds != [0, 1, 2]:
            raise RuntimeError(
                f"Random-partition session is not isolated correctly: optimization={opt_seeds}, "
                f"partition={part_seeds}"
            )
    return selected


def compact_results(root: Path):
    # Preserve metrics/manifests/JSON/CSV/logs; omit large checkpoints.
    skip_suffixes = {".pth", ".pt", ".ckpt"}
    if RESULT_ZIP.exists():
        RESULT_ZIP.unlink()
    with zipfile.ZipFile(RESULT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() in skip_suffixes:
                continue
            zf.write(path, path.relative_to(WORK))
    print("RESULT_ZIP=", RESULT_ZIP, flush=True)


CLONED_COMMIT = clone_repo()
install_runtime()
verify_editor_revision_code()
quick_preflight()
DATA.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

os.chdir(REPO)
from tools.experiment_grid import write_manifest, execute_specs

specs = build_selected_specs()
write_manifest(specs, MANIFEST)
print("\nSELECTED RUNS:")
for spec in specs:
    print(json.dumps({
        "name": spec.name,
        "seed": spec.parameters.get("seed"),
        "partition_mode": spec.parameters.get("trso_partition_mode"),
        "partition_seed": spec.parameters.get("trso_partition_seed"),
        "output_dir": spec.output_dir,
    }, indent=2))

statuses = execute_specs(specs, execute=True, skip_completed=True)
STATUS_JSON.write_text(json.dumps(statuses, indent=2), encoding="utf-8")

# Aggregate only this session. Alternating sessions produce cross-seed allocation
# stability; seeded-random sessions produce the partition-stability table.
run([
    sys.executable, "tools/aggregate_revision_results.py",
    "--root", str(OUTPUT),
    "--out_csv", str(SUMMARY_CSV),
], cwd=REPO)

provenance = {
    "session_id": SESSION_ID,
    "session_title": SESSION_TITLE,
    "dataset": DATASET,
    "backbone": BACKBONE,
    "partition_mode": PARTITION_MODE,
    "expected_runs": EXPECTED_RUNS,
    "github_repo": PUBLIC_REPO,
    "github_ref": GITHUB_REF,
    "github_commit": CLONED_COMMIT,
    "requested_commit_pin": GITHUB_COMMIT_PIN,
    "proposal_algorithm_changed": False,
}
(OUTPUT / "source_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

compact_results(OUTPUT)
print(json.dumps({
    "session": SESSION_ID,
    "title": SESSION_TITLE,
    "runs": EXPECTED_RUNS,
    "summary_csv": str(SUMMARY_CSV),
    "result_zip": str(RESULT_ZIP),
}, indent=2), flush=True)
