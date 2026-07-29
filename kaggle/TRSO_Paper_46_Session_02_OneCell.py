# Kaggle one-cell runner: TRSO 46-run comparison, Session 2 of 6.
# Settings: GPU enabled, Internet ON. Clones the tested G-CREST repository from GitHub main.

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

SESSION_ID = 2
WORK = Path("/kaggle/working")
REPO = WORK / "TRSO_Paper46"
GITHUB_REPO = os.environ.get(
    "TRSO_GITHUB_REPO",
    "https://github.com/tydeptrai21042004/trso_adapter.git",
).strip()
GITHUB_REF = os.environ.get("TRSO_GITHUB_REF", "main").strip() or "main"
GITHUB_COMMIT_PIN = os.environ.get("TRSO_GITHUB_COMMIT", "").strip()
DATA = WORK / "data"
OUTPUT = WORK / f"trso_paper_46_session_{SESSION_ID:02d}_results"
RESULT_ZIP = WORK / f"trso_paper_46_session_{SESSION_ID:02d}_results.zip"

SEED = "0"
SPLIT_SEED = 0
EPOCHS = 30
WARMUP = 3
INPUT_SIZE = 224
NUM_WORKERS = 4
RUN_TESTS = True
KEEP_CHECKPOINTS = False


def run(cmd, cwd=None, check=True):
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


def is_release(path: Path) -> bool:
    protocol = path / "tools/minimal_paper_protocol.py"
    proposal_core = path / "models/tuning_modules/mdl_tangent_core.py"
    method_file = path / "METHOD_GCREST_TRSO.md"
    return (
        (path / "main.py").is_file()
        and protocol.is_file()
        and proposal_core.is_file()
        and method_file.is_file()
        and "TOTAL_RUNS = MAIN_RUNS + len(ABLATIONS)" in protocol.read_text(encoding="utf-8", errors="ignore")
        and not (path / "models/tuning_modules/residual_adapter.py").exists()
        and not any(path.glob("*V7*.md")) and not any(path.glob("*V6*.md"))
    )


def clone_release() -> str:
    if REPO.exists():
        shutil.rmtree(REPO)

    command = [
        "git", "clone", "--depth", "1", "--single-branch",
        "--branch", GITHUB_REF, GITHUB_REPO, REPO,
    ]
    try:
        run(command)
        if GITHUB_COMMIT_PIN:
            run(["git", "fetch", "--depth", "1", "origin", GITHUB_COMMIT_PIN], cwd=REPO)
            run(["git", "checkout", "--detach", GITHUB_COMMIT_PIN], cwd=REPO)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Could not clone the TRSO GitHub repository. Enable Internet in Kaggle "
            "and confirm that the public repository/ref is valid: "
            f"{GITHUB_REPO} @ {GITHUB_REF}."
        ) from exc

    if not is_release(REPO):
        raise RuntimeError(
            "The cloned repository is not the tested G-CREST-TRSO "
            "46-run/six-session release."
        )

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    print(f"Cloned {GITHUB_REPO} ref={GITHUB_REF} commit={commit}", flush=True)
    return commit


CLONED_GITHUB_COMMIT = clone_release()
run([sys.executable, "-m", "pip", "install", "-q", "timm>=0.9,<2", "pandas>=2", "scipy>=1.10", "scikit-learn>=1.3", "tqdm>=4.65", "pytest>=8"])

import torch
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU accelerator.")
GPU_IDS = ",".join(str(i) for i in range(torch.cuda.device_count()))
PARALLEL = max(1, min(torch.cuda.device_count(), 2))

sys.path.insert(0, str(REPO))
from tools.minimal_paper_protocol import (
    ABLATIONS, ABLATION_SESSION, SESSION_TOTAL_RUNS, TOTAL_RUNS,
    payload, stages_for_session,
)
assert TOTAL_RUNS == 46
assert SESSION_ID in SESSION_TOTAL_RUNS
SESSION_STAGES = stages_for_session(SESSION_ID)
EXPECTED_SESSION_RUNS = SESSION_TOTAL_RUNS[SESSION_ID]
assert all("residual" not in stage.methods for stage in SESSION_STAGES)

DATA.mkdir(parents=True, exist_ok=True)
shutil.rmtree(OUTPUT, ignore_errors=True)
OUTPUT.mkdir(parents=True)
(OUTPUT / "minimal_paper_46_protocol.json").write_text(json.dumps(payload(), indent=2), encoding="utf-8")

if RUN_TESTS:
    run([sys.executable, "-m", "pytest", "-q"], cwd=REPO)

stage_manifests = []
reference_manifests = []
category_order = {"reference_control": 0, "paper_baseline": 1, "proposal": 2}
for stage in sorted(SESSION_STAGES, key=lambda item: (category_order[item.category], item.name)):
    stage_root = OUTPUT / stage.name
    manifest = stage_root / "fair_manifest.json"
    command = [
        sys.executable, "-m", "tools.run_fair_suite",
        "--dataset", stage.dataset,
        "--task", stage.task,
        "--data_path", DATA,
        "--download", "True",
        "--dataset_args_json", json.dumps(stage.dataset_args),
        "--backbones", stage.backbones,
        "--methods", stage.methods,
        "--trso_ablation", "full",
        "--seeds", SEED,
        "--split_seed", SPLIT_SEED,
        "--epochs", EPOCHS,
        "--warmup_epochs", WARMUP,
        "--batch_size", stage.batch_size,
        "--num_workers", NUM_WORKERS,
        "--input_size", INPUT_SIZE,
        "--optimizer", "adamw",
        "--augmentation", "strong",
        "--peft_lr", 1e-3,
        "--full_lr", 1e-4,
        "--linear_lr", 1e-3,
        "--weight_decay", 1e-4,
        "--min_lr", 1e-6,
        "--peft_head_lr_scale", 0.5,
        "--peft_freeze_head", "False",
        "--external_head_manifests", ",".join(reference_manifests) if stage.category != "reference_control" else "",
        "--output_root", stage_root / "runs",
        "--manifest", manifest,
        "--device", "cuda",
        "--gpu_ids", GPU_IDS,
        "--parallel_runs", PARALLEL,
        "--profile_efficiency", "True",
        "--measure_eval_latency", "True",
        "--execute",
    ]
    run(command, cwd=REPO)
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if len(rows) != stage.expected_runs:
        raise RuntimeError(f"{stage.name}: expected {stage.expected_runs} runs, got {len(rows)}")
    stage_manifests.append(str(manifest))
    if stage.category == "reference_control":
        reference_manifests.append(str(manifest))

ablation_manifest = None
if SESSION_ID == ABLATION_SESSION:
    all_rows = []
    for manifest_path in map(Path, stage_manifests):
        all_rows.extend(json.loads(manifest_path.read_text(encoding="utf-8")))
    linear_rows = [
        row for row in all_rows
        if row["parameters"].get("dataset") == "dtd"
        and row["parameters"].get("tuning_method") == "linear"
        and row["parameters"].get("backbone") == "resnet18"
        and int(row["parameters"].get("seed", -1)) == int(SEED)
    ]
    if len(linear_rows) != 1:
        raise RuntimeError(f"Expected one DTD ResNet-18 linear-probe row, got {len(linear_rows)}")
    head_path = Path(linear_rows[0]["output_dir"]) / "checkpoint-best.pth"
    if not head_path.is_file():
        raise FileNotFoundError(head_path)

    ablation_manifest = OUTPUT / "trso_proposal_ablation" / "manifest.json"
    run([
        sys.executable, "-m", "tools.run_trso_ablation",
        "--dataset", "dtd",
        "--data_path", DATA,
        "--download", "True",
        "--dataset_args_json", json.dumps({"dtd_partition": 1}),
        "--task", "auto",
        "--backbone", "resnet18",
        "--model_source", "torchvision",
        "--weights", "DEFAULT",
        "--pretrained", "True",
        "--seeds", SEED,
        "--ablations", ",".join(ABLATIONS),
        "--epochs", EPOCHS,
        "--warmup_epochs", WARMUP,
        "--batch_size", 32,
        "--num_workers", NUM_WORKERS,
        "--input_size", INPUT_SIZE,
        "--optimizer", "adamw",
        "--lr", 1e-3,
        "--weight_decay", 1e-4,
        "--min_lr", 1e-6,
        "--augmentation", "strong",
        "--split_seed", SPLIT_SEED,
        "--head_from", head_path,
        "--fair_protocol", "True",
        "--peft_head_lr_scale", 0.5,
        "--peft_freeze_head", "False",
        "--device", "cuda",
        "--output_root", OUTPUT / "trso_proposal_ablation" / "runs",
        "--manifest", ablation_manifest,
        "--profile_efficiency", "True",
        "--execute",
    ], cwd=REPO)
    ablation_rows = json.loads(ablation_manifest.read_text(encoding="utf-8"))
    if len(ablation_rows) != len(ABLATIONS):
        raise RuntimeError(f"Expected {len(ABLATIONS)} ablation runs, got {len(ablation_rows)}")

run([
    sys.executable, "-m", "tools.aggregate_revision_results",
    "--root", OUTPUT,
    "--out_csv", OUTPUT / "all_results.csv",
], cwd=REPO)

category_runs = {"paper_baseline": 0, "reference_control": 0, "proposal": 0}
for stage in SESSION_STAGES:
    category_runs[stage.category] += stage.expected_runs
summary = {
    "protocol_total_training_runs": 46,
    "session": SESSION_ID,
    "session_expected_training_runs": EXPECTED_SESSION_RUNS,
    "session_main_runs": sum(stage.expected_runs for stage in SESSION_STAGES),
    "session_category_runs": category_runs,
    "session_trso_ablations": len(ABLATIONS) if SESSION_ID == ABLATION_SESSION else 0,
    "seed": 0,
    "epochs": EPOCHS,
    "augmentation": "strong",
    "full_learning_rate": 1e-4,
    "linear_learning_rate": 1e-3,
    "stage_manifests": stage_manifests,
    "ablation_manifest": str(ablation_manifest) if ablation_manifest else None,
    "reference_controls_are_separate_from_paper_baselines": True,
    "residual_adapter_removed": True,
    "proposal_version": "global_cross_fitted_reproducibility_entropy_spectral_tangent_core",
    "source": "github_clone",
    "github_repository": GITHUB_REPO,
    "github_ref": GITHUB_REF,
    "github_commit": CLONED_GITHUB_COMMIT,
}
(OUTPUT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

if not KEEP_CHECKPOINTS:
    for suffix in ("*.pth", "*.pt", "*.ckpt"):
        for path in OUTPUT.rglob(suffix):
            path.unlink(missing_ok=True)

RESULT_ZIP.unlink(missing_ok=True)
with zipfile.ZipFile(RESULT_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in OUTPUT.rglob("*"):
        if path.is_file():
            archive.write(path, path.relative_to(OUTPUT.parent))

print(f"Completed Session {SESSION_ID}/6: {EXPECTED_SESSION_RUNS} training runs.")
print("Results:", OUTPUT)
print("Archive:", RESULT_ZIP)
