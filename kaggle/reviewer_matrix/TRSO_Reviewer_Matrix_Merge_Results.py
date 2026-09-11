# Kaggle merge-only utility for the 18 reviewer-matrix session archives.
# Attach downloaded session ZIPs as Kaggle inputs, then run this cell/script.
# The merged ZIP preserves every aggregate metric exported by each session.

import csv
import io
import json
from pathlib import Path
import shutil
import zipfile

WORK = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
INPUT = Path('/kaggle/input') if Path('/kaggle/input').exists() else WORK
OUT = WORK / 'trso_reviewer_matrix_merged'
ZIP_OUT = WORK / 'trso_reviewer_matrix_merged.zip'
EXPECTED_TOTAL_SEED_RUNS = 204
REQUIRED_SEEDS = {0, 1, 2}


def write_union_csv(path: Path, rows: list[dict], preferred=()):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys = set().union(*(row.keys() for row in rows))
    fields = [key for key in preferred if key in keys]
    fields.extend(sorted(keys - set(fields)))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def read_json_member(zf: zipfile.ZipFile, suffix: str, default):
    members = [name for name in zf.namelist() if name.endswith(suffix)]
    if len(members) != 1:
        return default
    return json.loads(zf.read(members[0]).decode('utf-8'))


def read_csv_member(zf: zipfile.ZipFile, suffix: str) -> list[dict]:
    members = [name for name in zf.namelist() if name.endswith(suffix)]
    if len(members) != 1:
        return []
    text = zf.read(members[0]).decode('utf-8')
    if not text.strip():
        return []
    return list(csv.DictReader(io.StringIO(text)))


if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

archives = sorted(INPUT.rglob('*reviewer*matrix*session*results*.zip'))
if not archives:
    archives = sorted(INPUT.rglob('trso_reviewer_matrix_session_*_results.zip'))
if not archives:
    raise FileNotFoundError('No reviewer-matrix session result ZIPs were found under /kaggle/input.')

rows = []
seen = set()
all_reported_runs = {}
session_states = {}
final_metric_rows = []
epoch_metric_rows = []
inventory_rows = []
seed_rows = []
coverage_by_session = {}

for archive in archives:
    print('Reading', archive)
    with zipfile.ZipFile(archive) as zf:
        states = [n for n in zf.namelist() if n.endswith('/session_state.json')]
        if len(states) != 1:
            print('  skip: could not identify one session_state.json')
            continue
        state = json.loads(zf.read(states[0]).decode('utf-8'))
        session = int(state['session'])
        session_states[str(session)] = state

        reported = read_json_member(zf, '/all_reported_metrics.json', {'runs': {}})
        coverage = read_json_member(zf, '/metrics_coverage.json', {})
        coverage_by_session[str(session)] = coverage

        for rid, meta in state.get('completed', {}).items():
            if rid in seen:
                continue
            seen.add(rid)
            rows.append({
                'session': session,
                'run_id': rid,
                'setting': meta.get('setting', ''),
                'method': meta.get('method', ''),
                'seed': meta.get('seed', ''),
                'estimated_minutes_t4': meta.get('estimated_minutes_t4', ''),
                'actual_minutes': meta.get('actual_minutes', ''),
                'git_commit': state.get('git_commit', ''),
                'gpu_name': state.get('gpu_name', ''),
                'source_archive': archive.name,
            })
            if rid in reported.get('runs', {}):
                all_reported_runs[rid] = reported['runs'][rid]

        # These aggregate CSVs contain all scalar final metrics, all per-epoch
        # history metrics, artifact hashes, and seed completeness respectively.
        for row in read_csv_member(zf, '/all_final_metrics.csv'):
            row['session'] = session
            row['source_archive'] = archive.name
            final_metric_rows.append(row)
        for row in read_csv_member(zf, '/all_epoch_metrics.csv'):
            row['session'] = session
            row['source_archive'] = archive.name
            epoch_metric_rows.append(row)
        for row in read_csv_member(zf, '/metrics_inventory.csv'):
            row['session'] = session
            row['source_archive'] = archive.name
            inventory_rows.append(row)
        for row in read_csv_member(zf, '/seed_completeness.csv'):
            row['session'] = session
            row['source_archive'] = archive.name
            seed_rows.append(row)

rows.sort(key=lambda x: (x['setting'], x['method'], int(x['seed']) if str(x['seed']).isdigit() else 999, x['run_id']))
write_union_csv(
    OUT / 'all_completed_runs.csv', rows,
    preferred=('session', 'run_id', 'setting', 'method', 'seed', 'actual_minutes', 'estimated_minutes_t4', 'gpu_name', 'git_commit', 'source_archive'),
)
write_union_csv(
    OUT / 'all_final_metrics.csv', final_metric_rows,
    preferred=('session', 'run_id', 'setting', 'method', 'seed', 'source_archive'),
)
write_union_csv(
    OUT / 'all_epoch_metrics.csv', epoch_metric_rows,
    preferred=('session', 'run_id', 'setting', 'method', 'seed', 'epoch', 'source_archive'),
)
write_union_csv(
    OUT / 'metrics_inventory.csv', inventory_rows,
    preferred=('session', 'run_id', 'setting', 'method', 'seed', 'artifact', 'bytes', 'sha256', 'parse_status', 'source_archive'),
)
write_union_csv(
    OUT / 'seed_completeness.csv', seed_rows,
    preferred=('session', 'experiment_id', 'setting', 'method', 'seed', 'status', 'run_id', 'source_archive'),
)

(OUT / 'all_reported_metrics.json').write_text(json.dumps({
    'schema_version': 1,
    'description': 'Merged exact JSON metric/report payloads from all session archives.',
    'runs': all_reported_runs,
}, indent=2), encoding='utf-8')
(OUT / 'session_states.json').write_text(json.dumps(session_states, indent=2), encoding='utf-8')
(OUT / 'metrics_coverage_by_session.json').write_text(json.dumps(coverage_by_session, indent=2), encoding='utf-8')

seed_sets = {}
for row in rows:
    key = (row['setting'], row['method'])
    try:
        seed_sets.setdefault(key, set()).add(int(row['seed']))
    except (TypeError, ValueError):
        pass
bad_seed_groups = [
    {'setting': setting, 'method': method, 'completed_seeds': sorted(seeds)}
    for (setting, method), seeds in sorted(seed_sets.items())
    if seeds != REQUIRED_SEEDS
]
expected_groups = EXPECTED_TOTAL_SEED_RUNS // len(REQUIRED_SEEDS)
complete_groups = sum(seeds == REQUIRED_SEEDS for seeds in seed_sets.values())
merge_summary = {
    'archives_found': len(archives),
    'sessions_found': sorted(int(x) for x in session_states),
    'unique_completed_seed_runs': len(rows),
    'expected_total_seed_runs': EXPECTED_TOTAL_SEED_RUNS,
    'required_seeds_per_experiment': sorted(REQUIRED_SEEDS),
    'expected_experiment_groups': expected_groups,
    'fully_complete_three_seed_groups': complete_groups,
    'incomplete_three_seed_groups': bad_seed_groups,
    'all_204_seed_runs_complete': len(rows) == EXPECTED_TOTAL_SEED_RUNS,
    'all_experiment_groups_have_seeds_0_1_2': len(seed_sets) == expected_groups and not bad_seed_groups,
    'complete': len(rows) == EXPECTED_TOTAL_SEED_RUNS and len(seed_sets) == expected_groups and not bad_seed_groups,
    'final_metric_rows': len(final_metric_rows),
    'epoch_metric_rows': len(epoch_metric_rows),
    'metric_artifact_inventory_rows': len(inventory_rows),
}
(OUT / 'merge_summary.json').write_text(json.dumps(merge_summary, indent=2), encoding='utf-8')

if ZIP_OUT.exists():
    ZIP_OUT.unlink()
with zipfile.ZipFile(ZIP_OUT, 'w', zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(OUT.rglob('*')):
        if path.is_file():
            zf.write(path, path.relative_to(OUT.parent))

print(f"Merged {len(rows)}/{EXPECTED_TOTAL_SEED_RUNS} completed seed-runs")
print(f"Three-seed groups: {complete_groups}/{expected_groups}")
print(OUT / 'all_final_metrics.csv')
print(OUT / 'all_epoch_metrics.csv')
print(OUT / 'all_reported_metrics.json')
print(ZIP_OUT)
