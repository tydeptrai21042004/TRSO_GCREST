"""Shared Kaggle runner for revision Sessions 07-14."""
from __future__ import annotations
import json, os, shutil, subprocess, sys, zipfile
from pathlib import Path


def run_revision_session(session_id: int):
    work = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
    repo = work / 'TRSO_Revision'
    url = os.environ.get('TRSO_GITHUB_REPO', 'https://github.com/tydeptrai21042004/trso_adapter.git')
    ref = os.environ.get('TRSO_GITHUB_REF', 'main')
    if repo.exists(): shutil.rmtree(repo)
    subprocess.run(['git','clone','--depth','1','--branch',ref,url,str(repo)], check=True)
    subprocess.run([sys.executable,'-m','pip','install','-q','timm>=0.9,<2','pandas>=2','scipy>=1.10','scikit-learn>=1.3','tqdm>=4.65','medmnist>=3.0'], check=True)
    sys.path.insert(0, str(repo))
    from tools.extended_paper_protocol import SESSIONS, session_payload
    spec = SESSIONS[int(session_id)]
    out = work / f'trso_revision_session_{session_id:02d}_results'
    out.mkdir(parents=True, exist_ok=True)
    (out/'session_protocol.json').write_text(json.dumps(session_payload(session_id), indent=2), encoding='utf-8')
    for index, command in enumerate(spec.commands, 1):
        cmd = [sys.executable, *command, '--output_root', str(out/f'command_{index:02d}'), '--manifest', str(out/f'command_{index:02d}_manifest.json')]
        print('+', ' '.join(cmd), flush=True)
        subprocess.run(cmd, cwd=repo, check=True)
    target = work / f'trso_revision_session_{session_id:02d}_results.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in out.rglob('*'):
            if path.is_file(): zf.write(path, path.relative_to(work))
    print(target)
