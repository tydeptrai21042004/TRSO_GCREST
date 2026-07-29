# Non-training Kaggle utility: merge the six TRSO 46-run session archives.
# Add all six session output ZIPs as Kaggle input datasets, then run this cell.

import csv
import json
import shutil
import zipfile
from pathlib import Path

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
MERGED = WORK / "trso_paper_46_merged_results"
MERGED_ZIP = WORK / "trso_paper_46_merged_results.zip"

shutil.rmtree(MERGED, ignore_errors=True)
MERGED.mkdir(parents=True)

archives = []
for session in range(1, 7):
    matches = list(INPUT.rglob(f"trso_paper_46_session_{session:02d}_results.zip"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one Session {session} ZIP, found {len(matches)}")
    archives.append(matches[0])
    with zipfile.ZipFile(matches[0]) as handle:
        handle.extractall(MERGED)

summaries = []
result_csvs = []
for session in range(1, 7):
    root = MERGED / f"trso_paper_46_session_{session:02d}_results"
    summary_path = root / "run_summary.json"
    csv_path = root / "all_results.csv"
    if not summary_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError(f"Session {session} is incomplete: {root}")
    summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    result_csvs.append(csv_path)

if [row["session"] for row in summaries] != list(range(1, 7)):
    raise RuntimeError("Session summaries are missing or out of order.")
if sum(row["session_expected_training_runs"] for row in summaries) != 46:
    raise RuntimeError("The six session summaries do not total 46 runs.")

combined_rows = []
fieldnames = []
for path in result_csvs:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for field in reader.fieldnames or []:
            if field not in fieldnames:
                fieldnames.append(field)
        combined_rows.extend(reader)

combined_csv = MERGED / "all_46_results.csv"
with combined_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(combined_rows)

merged_summary = {
    "expected_training_runs": 46,
    "sessions": 6,
    "category_runs": {"paper_baseline": 12, "reference_control": 20, "proposal": 10},
    "additional_trso_ablations": 4,
    "session_summaries": summaries,
    "combined_result_rows": len(combined_rows),
}
(MERGED / "merged_run_summary.json").write_text(json.dumps(merged_summary, indent=2), encoding="utf-8")

MERGED_ZIP.unlink(missing_ok=True)
with zipfile.ZipFile(MERGED_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in MERGED.rglob("*"):
        if path.is_file():
            archive.write(path, path.relative_to(MERGED.parent))

print("Merged all six sessions.")
print("Combined CSV:", combined_csv)
print("Archive:", MERGED_ZIP)
