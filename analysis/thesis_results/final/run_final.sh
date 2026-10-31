#!/usr/bin/env bash
# Results follow-up (results_followup_handoff.md, version 4), end to end. Resumable: every
# stage skips finished work files (/data/tmp/thesis_results_work/final/), so re-running is
# cheap. Check 4's GPU refits (about 12 minutes) are run_smoothing.sh, best launched in tmux:
#   tmux new-session -d -s tf-smoothing 'bash analysis/thesis_results/final/run_smoothing.sh 2>&1 | tee -a results/thesis_results_final/logs/smoothing.log'
set -eu
cd /ephemeral/code/thesis-introspection
source env.sh
export THESIS_POOL=D2 OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
P=analysis/thesis_results/final
mkdir -p results/thesis_results_final/logs
python $P/headline.py                     # §1 tables, paired, F1, figures, appendix
python $P/checks.py                       # §2-§4 and §6
[ -f results/thesis_results_final/logs/smoothing.done ] || bash $P/run_smoothing.sh   # §5
python $P/scores.py                       # per-row scores of every new detector
python $P/report.py                       # REPORT.md (headline text: report_notes.md)
