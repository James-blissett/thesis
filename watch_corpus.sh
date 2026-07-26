#!/usr/bin/env bash
# Progress snapshot for a running (or finished) gen_rollouts.py session.
#
# Usage:  ./watch_corpus.sh [corpus_dir]
#         watch -n 30 ./watch_corpus.sh      # auto-refreshing dashboard
#
# Reads session_summary.json, which gen_rollouts.py rewrites after every rollout, so this
# is safe to run at any time and never touches the generation process.

S="${1:-/data/corpus}/session_summary.json"
[ -f "$S" ] || { echo "no session_summary.json yet at $S"; exit 0; }

python3 - "$S" <<'PY'
import json, sys

d = json.load(open(sys.argv[1]))
r = d["rollouts"]
n, ns = len(r), d["n_success"]
avg = sum(x["wall_seconds"] for x in r) / n
eta = avg * (50 - n) / 60

print(f"rollouts   : {n}/50  ({100*n/50:.0f}%)")
print(f"success    : {ns}/{n} = {100*ns/n:.1f}%   [sanity gate: 25-80%]")
print(f"timesteps  : {d['total_timesteps']:,}")
print(f"avg rollout: {avg:.0f}s    ETA ~{eta:.0f} min")

# Loud failure signals. EGL breakage yields well-formed all-black JPEGs that pass every
# structural check, so the frame check is on pixel content.
black = [x["rollout_id"] for x in r if x["frame_stats"]["mean_pixel"] < 5]
if black:
    print(f"!! BLACK FRAMES: {black}")
oob = [x["rollout_id"] for x in r
       if not x.get("parity", {}).get("mismatches_within_tie_band", True)]
if oob:
    print(f"!! PARITY OUT OF BAND: {oob}")

print("\nper-task    " + "".join(f"{'t'+str(i):>3}" for i in range(10)))
print("done      " + "".join(f"{sum(1 for x in r if x['task_id']==t):>3}" for t in range(10)))
print("success   " + "".join(f"{sum(x['success'] for x in r if x['task_id']==t):>3}"
                             for t in range(10)))

print()
for x in r[-3:]:
    print(f"  {x['rollout_id']}  success={str(x['success']):5s} "
          f"steps={x['n_timesteps']:>3}  {x['wall_seconds']:.0f}s")
PY
