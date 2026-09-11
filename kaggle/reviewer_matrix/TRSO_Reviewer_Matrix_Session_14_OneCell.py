# Kaggle one-cell runner: TRSO reviewer-ready 204-run main matrix, Session 14 of 18.
# GPU ON + Internet ON. By default this cell clones the public TRSO_GCREST repo.

import os, shutil, subprocess, sys, time
from pathlib import Path

SESSION_ID = 14
PUBLIC_REPO = "https://github.com/tydeptrai21042004/TRSO_GCREST.git"
os.environ.setdefault("TRSO_SESSION_BOOTSTRAP_START_EPOCH", str(time.time()))
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
TARGET = WORK / "TRSO_GCREST"


def run(cmd, cwd=None):
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def valid_repo(path: Path) -> bool:
    return (
        (path / "tools" / "run_fair_suite.py").is_file()
        and (path / "baseline_recipes.py").is_file()
        and (path / "kaggle" / "reviewer_matrix" / "session_runner.py").is_file()
        and (path / "kaggle" / "reviewer_matrix" / "matrix_protocol.py").is_file()
    )


def clone_public_repo() -> Path:
    explicit = os.environ.get("TRSO_REPO_DIR", "").strip()
    if explicit:
        repo = Path(explicit).resolve()
        if not valid_repo(repo):
            raise RuntimeError(f"TRSO_REPO_DIR is not a corrected TRSO_GCREST repo: {repo}")
        return repo
    repo_url = os.environ.get("TRSO_GITHUB_REPO", PUBLIC_REPO).strip() or PUBLIC_REPO
    ref = os.environ.get("TRSO_GITHUB_REF", "main").strip() or "main"
    commit = os.environ.get("TRSO_GITHUB_COMMIT", "").strip()
    if TARGET.exists():
        shutil.rmtree(TARGET)
    run(["git", "clone", "--depth", "1", "--single-branch", "--branch", ref, repo_url, TARGET])
    if commit:
        run(["git", "fetch", "--depth", "1", "origin", commit], cwd=TARGET)
        run(["git", "checkout", "--detach", commit], cwd=TARGET)
    if not valid_repo(TARGET):
        raise RuntimeError("Cloned repo does not contain the corrected kaggle/reviewer_matrix workflow. Push this corrected package first.")
    print("Resolved commit:", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TARGET, text=True).strip())
    return TARGET


REPO = clone_public_repo()
run([
    sys.executable, "-m", "pip", "install", "-q",
    "timm>=0.9,<2", "pandas>=2", "scipy>=1.10", "scikit-learn>=1.3",
    "tqdm>=4.65", "medmnist>=3.0", "fvcore>=0.1.5",
])
run([
    sys.executable,
    REPO / "kaggle" / "reviewer_matrix" / "session_runner.py",
    "--session", str(SESSION_ID), "--repo", REPO,
], cwd=REPO)
