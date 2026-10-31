"""
scores.py

Per-row scores of every detector made in the follow-up (results_followup_handoff.md §0
outputs): results/thesis_results_final/scores/<training set>_seed<s>.npz, format in
fc.write_scores. Keys: F_B_noxl_h<H> (Check 1), F_B_common*_h<H> (Check 2, both training
sets), SM_<setting>__<row> (Check 4). Run with THESIS_POOL=D2; the original training set's
file is written by a subprocess.

Usage: source env.sh; python analysis/thesis_results/final/scores.py
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc  # noqa: E402

if __name__ == "__main__":
    keys = fc.new_keys()
    fc.write_scores(keys)
    fc.log(f"[scores] {fc.common.POOL_NAME}: {len(keys)} detectors")
    if fc.common.POOL == "D2":
        subprocess.run([sys.executable, __file__], env={**os.environ, "THESIS_POOL": "D0"}, check=True)
