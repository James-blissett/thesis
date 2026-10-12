"""Write collect_pro/init_state_assignment_v3.json (Step 2). Refuses to overwrite."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

cfg = common.load_config()
path = common.assignment_path(cfg)
if path.exists() and "--force" not in sys.argv:
    raise SystemExit(f"{path} exists; not overwriting")
a = common.make_assignment(cfg)
path.write_text(json.dumps(a, indent=1))
js = common.jobs(cfg, a)
print(f"wrote {path}: {len(js)} jobs; first {[j['rollout_id'] for j in js[:3]]}")
