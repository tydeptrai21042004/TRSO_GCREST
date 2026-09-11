# Kaggle one-cell revision experiment session 20. GPU + Internet ON.
import sys
from pathlib import Path
import urllib.request
runner = Path('/kaggle/working/revision_session_runner.py')
url = 'https://raw.githubusercontent.com/tydeptrai21042004/trso_adapter/main/kaggle/revision_session_runner.py'
urllib.request.urlretrieve(url, runner)
sys.path.insert(0, str(runner.parent))
from revision_session_runner import run_revision_session
run_revision_session(20)
