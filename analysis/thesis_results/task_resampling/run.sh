#!/usr/bin/env bash
# Task resampling (task_resampling_handoff.md, version 4), end to end. Resumable: the
# random-direction GPU pass skips finished layers (/data/tmp/thesis_results_work/task_resampling/
# random_dirs/); about 12 minutes on one L40S, best launched in tmux.
set -eu
cd /ephemeral/code/thesis-introspection
source env.sh
export THESIS_POOL=D2 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
P=analysis/thesis_results/task_resampling
mkdir -p results/thesis_results_final/task_resampling/logs
python $P/ranges.py check ranges                  # §1 self-check, §2-3 ranges
python $P/layer0.py                               # §6 extra: means without layer 0
python $P/ranges.py layer0                        #   and their ranges
python $P/false_alarms.py                         # §4
python $P/failure_types.py                        # §5 (descriptions: failure_descriptions.md, by hand)
python $P/random_directions.py check directions table figure   # §6
python $P/report.py                               # REPORT.md (headline text: report_notes.md)
