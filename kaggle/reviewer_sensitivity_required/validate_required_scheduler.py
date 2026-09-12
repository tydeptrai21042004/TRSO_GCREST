from __future__ import annotations
import importlib.util, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import required_protocol as rp
import required_session_runner as rr

def main():
    assert rp.total_group_count() == 39
    assert rp.total_run_count() == 105
    assert rp.SESSION_COUNT == 9
    assert max(rp.session_estimated_minutes_t4(i) for i in rp.SESSIONS) <= 480
    studies = {g.study for g in rp.groups()}
    assert studies == {"reliability","calibration","batch_sensitivity","mode_rules","r_scale","lora_rank_sweep"}
    # Validate that every logical group can resolve to one exact audited run spec.
    tmp = ROOT / ".scheduler_validate"
    for group in rp.groups():
        for seed in group.seeds[:1]:
            spec = rr.exact_run_spec(group, seed, data_dir=tmp/"data", run_root=tmp/"runs", manifest=tmp/"m.json")
            assert spec.name == group.variant
            assert int(spec.parameters.get("seed", -1)) == int(seed)
            assert "@" not in str(spec.parameters.get("backbone", ""))
            if "@" in group.backbone:
                assert spec.parameters.get("model_source") == group.backbone.rsplit("@", 1)[1]
    print("OK: 39 groups / 105 runs / 9 sessions")
    for sid in range(1, rp.SESSION_COUNT+1):
        print(f"Session {sid:02d}: {rp.session_run_count(sid):2d} runs, {rp.session_estimated_minutes_t4(sid):3d} T4 min")
    return 0

if __name__ == "__main__": raise SystemExit(main())
