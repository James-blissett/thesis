#!/usr/bin/env bash
# More-data run status in <= 5 lines (more_data_collection.md section 1, "Polling").
cd /ephemeral/code/thesis-introspection
source env.sh >/dev/null 2>&1
python - <<'PY'
import json, sys, time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, "collect_pro")
import common
cfg = common.load_config(); out = common.out_dir(cfg)
try:
    planned = len(common.jobs(cfg, common.load_assignment(cfg)))
except Exception:
    planned = 0
recs = []
p = out / "progress.jsonl"
if p.exists():
    for l in p.read_text().splitlines():
        try: recs.append(json.loads(l))
        except Exception: pass
done = sum(common.is_complete(d) for d in out.glob("*_task*_trial*") if not d.name.startswith(".")) if out.exists() else 0
fails = []
f = out / "failures.jsonl"
if f.exists():
    fails = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
now = datetime.now(timezone.utc)
recent = [r for r in recs if (now - datetime.fromisoformat(r["finished_utc"])).total_seconds() < 7200]
rate = len(recent) / 2.0 if recent else 0.0
eta = f"{(planned - done - len(fails)) / rate:.1f} h" if rate else "n/a"
succ = sum(r["success_ever"] for r in recs)
print(f"rollouts {done}/{planned} done ({succ} successes in progress.jsonl), {len(fails)} skipped/failed")
print(f"rate {rate:.1f}/h (last 2 h); est. remaining {eta}")
print(f"free disk /data {common.free_gb(out if out.exists() else Path('/data')):.0f} GB (headroom {cfg['HEADROOM_GB']})")
print("last error: " + (f"{fails[-1]['rollout_id']} {fails[-1]['error'][:100]}" if fails else "none"))
PY
echo "tmux: $(tmux ls 2>/dev/null | grep -o '^md-[^:]*' | tr '\n' ' ')"
