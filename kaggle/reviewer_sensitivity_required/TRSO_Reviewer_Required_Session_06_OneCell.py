# Kaggle one-cell runner: reviewer-required sensitivity scheduler, Session 06 of 9.
# GPU ON + Internet ON. Clone public repo by default; pin TRSO_GITHUB_COMMIT for archival runs.

import os, shutil, subprocess, sys, time
from pathlib import Path

SESSION_ID = 6
PUBLIC_REPO = "https://github.com/tydeptrai21042004/TRSO_GCREST.git"
os.environ.setdefault("TRSO_SESSION_BOOTSTRAP_START_EPOCH", str(time.time()))
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
TARGET = WORK / "TRSO_GCREST"

def run(cmd, cwd=None):
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)

def valid_repo(path: Path) -> bool:
    d = path / "kaggle" / "reviewer_sensitivity_required"
    return ((path / "tools" / "run_reviewer_revision.py").is_file()
            and (d / "required_protocol.py").is_file()
            and (d / "required_session_runner.py").is_file())

def clone_public_repo() -> Path:
    explicit = os.environ.get("TRSO_REPO_DIR", "").strip()
    if explicit:
        repo = Path(explicit).resolve()
        if not valid_repo(repo):
            raise RuntimeError(f"TRSO_REPO_DIR does not contain reviewer_sensitivity_required: {repo}")
        return repo
    repo_url = os.environ.get("TRSO_GITHUB_REPO", PUBLIC_REPO).strip() or PUBLIC_REPO
    ref = os.environ.get("TRSO_GITHUB_REF", "main").strip() or "main"
    commit = os.environ.get("TRSO_GITHUB_COMMIT", "").strip()
    if TARGET.exists(): shutil.rmtree(TARGET)
    run(["git", "clone", "--depth", "1", "--single-branch", "--branch", ref, repo_url, TARGET])
    if commit:
        run(["git", "fetch", "--depth", "1", "origin", commit], cwd=TARGET)
        run(["git", "checkout", "--detach", commit], cwd=TARGET)
    if not valid_repo(TARGET):
        raise RuntimeError("Cloned repo lacks kaggle/reviewer_sensitivity_required. Push this new folder first.")
    print("Resolved commit:", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TARGET, text=True).strip())
    return TARGET

REPO = clone_public_repo()
run([sys.executable, "-m", "pip", "install", "-q",
     "timm>=0.9,<2", "pandas>=2", "scipy>=1.10", "scikit-learn>=1.3",
     "tqdm>=4.65", "medmnist>=3.0", "fvcore>=0.1.5"])
run([sys.executable,
     REPO / "kaggle" / "reviewer_sensitivity_required" / "required_session_runner.py",
     "--session", str(SESSION_ID), "--repo", REPO], cwd=REPO)
