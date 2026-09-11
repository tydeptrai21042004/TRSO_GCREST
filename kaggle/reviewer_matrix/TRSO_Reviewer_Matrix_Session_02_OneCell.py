# Kaggle one-cell runner: TRSO reviewer-ready 204-run matrix, Session 02 of 18.
# GPU ON. Internet is required only when no updated project ZIP is attached.

import os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

SESSION_ID = 2

os.environ.setdefault('TRSO_SESSION_BOOTSTRAP_START_EPOCH', str(time.time()))
WORK = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
TARGET = WORK / 'TRSO_Reviewer_Matrix'


def run(cmd, cwd=None):
    cmd = [str(x) for x in cmd]
    print('+', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def valid_repo(path: Path) -> bool:
    return (
        (path / 'tools' / 'run_fair_suite.py').is_file()
        and (path / 'baseline_recipes.py').is_file()
        and (path / 'kaggle' / 'reviewer_matrix' / 'session_runner.py').is_file()
        and (path / 'kaggle' / 'reviewer_matrix' / 'matrix_protocol.py').is_file()
    )


def root_inside_extract(base: Path) -> Path | None:
    if valid_repo(base):
        return base
    for candidate in base.rglob('session_runner.py'):
        if candidate.as_posix().endswith('/kaggle/reviewer_matrix/session_runner.py'):
            repo = candidate.parents[2]
            if valid_repo(repo):
                return repo
    return None


def zip_has_active_runner(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            return any(name.endswith('/kaggle/reviewer_matrix/session_runner.py') for name in zf.namelist())
    except Exception:
        return False


def locate_repo() -> Path:
    explicit_dir = os.environ.get('TRSO_REPO_DIR', '').strip()
    if explicit_dir:
        path = Path(explicit_dir).resolve()
        if not valid_repo(path):
            raise RuntimeError(f'TRSO_REPO_DIR is not the updated project: {path}')
        return path

    try:
        here = Path(__file__).resolve()
        for parent in here.parents:
            if valid_repo(parent):
                return parent
    except NameError:
        pass

    explicit_zip = os.environ.get('TRSO_PROJECT_ZIP', '').strip()
    candidates = [Path(explicit_zip)] if explicit_zip else []
    kaggle_input = Path('/kaggle/input')
    if not explicit_zip and kaggle_input.exists():
        candidates.extend(sorted(kaggle_input.rglob('*.zip')))
    for archive in candidates:
        if archive.is_file() and zip_has_active_runner(archive):
            extract = WORK / 'TRSO_Reviewer_Matrix_Source'
            if extract.exists():
                shutil.rmtree(extract)
            extract.mkdir(parents=True)
            print(f'Using attached updated project ZIP: {archive}', flush=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(extract)
            repo = root_inside_extract(extract)
            if repo:
                return repo

    repo_url = os.environ.get('TRSO_GITHUB_REPO', 'https://github.com/tydeptrai21042004/trso_adapter.git').strip()
    ref = os.environ.get('TRSO_GITHUB_REF', 'main').strip() or 'main'
    commit = os.environ.get('TRSO_GITHUB_COMMIT', '').strip()
    if TARGET.exists():
        shutil.rmtree(TARGET)
    run(['git', 'clone', '--depth', '1', '--single-branch', '--branch', ref, repo_url, TARGET])
    if commit:
        run(['git', 'fetch', '--depth', '1', 'origin', commit], cwd=TARGET)
        run(['git', 'checkout', '--detach', commit], cwd=TARGET)
    if not valid_repo(TARGET):
        raise RuntimeError(
            'The cloned repository does not contain the new kaggle/reviewer_matrix runner. '
            'Push this corrected package to GitHub, set TRSO_PROJECT_ZIP to the updated ZIP, '
            'or attach the ZIP as a Kaggle Dataset.'
        )
    return TARGET


REPO = locate_repo()
run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'timm>=0.9,<2', 'pandas>=2', 'scipy>=1.10', 'scikit-learn>=1.3',
    'tqdm>=4.65', 'medmnist>=3.0', 'pytest>=8'
])
run([
    sys.executable,
    REPO / 'kaggle' / 'reviewer_matrix' / 'session_runner.py',
    '--session', str(SESSION_ID), '--repo', REPO,
], cwd=REPO)
