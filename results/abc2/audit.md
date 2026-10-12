# Corpus audit (more_data_collection.md Step 5)

Success = `success_ever`. Block `ori` = the 300 v2 rollouts. `failed_rollouts`: raised twice and logged in failures.jsonl; `not_run`: planned but never attempted (cell skipped, COLLECT_HOURS_MAX or disk stop).

## Per block

| block | role | planned | collected | successes | failures | success_rate | failed_rollouts | not_run | cells_skipped |
|---|---|---|---|---|---|---|---|---|---|
| ori | training | 300 | 300 | 137 | 163 | 0.4567 | 0 | 0 | 0 |
| ori2 | training | 200 | 200 | 91 | 109 | 0.455 | 0 | 0 | 0 |
| obj | training | 200 | 200 | 47 | 153 | 0.235 | 0 | 0 | 0 |
| lan | training | 200 | 200 | 85 | 115 | 0.425 | 0 | 0 | 0 |
| swap | stress | 50 | 50 | 0 | 50 | 0.0 | 0 | 0 | 0 |
| task | stress | 50 | 50 | 4 | 46 | 0.08 | 0 | 0 | 0 |
| all_new |  | 700 | 700 | 227 | 473 | 0.3243 | 0 | 0 | 0 |

## Collection

- started 2026-10-11T12:04:06+00:00, last rollout finished 2026-10-12T08:18:15.317564+00:00
- hours 20.236, rollouts finished in that window 696 (+ 4 smoke-test rollouts before the start), rollouts/hour 34.39
- COLLECT_HOURS_MAX 36 h; cap reached (last finish at or after the cap): no
- cells skipped (3 failures in a row): none
- collection stops logged (disk headroom): 0
- failures.jsonl entries: 0

## Per (block, task)

| block | task | planned | collected | successes | failures | success_rate | failed_rollouts | not_run | cell_skipped |
|---|---|---|---|---|---|---|---|---|---|
| ori | 0 | 30 | 30 | 11 | 19 | 0.3667 | 0 | 0 | 0 |
| ori | 1 | 30 | 30 | 24 | 6 | 0.8 | 0 | 0 | 0 |
| ori | 2 | 30 | 30 | 16 | 14 | 0.5333 | 0 | 0 | 0 |
| ori | 3 | 30 | 30 | 5 | 25 | 0.1667 | 0 | 0 | 0 |
| ori | 4 | 30 | 30 | 8 | 22 | 0.2667 | 0 | 0 | 0 |
| ori | 5 | 30 | 30 | 22 | 8 | 0.7333 | 0 | 0 | 0 |
| ori | 6 | 30 | 30 | 13 | 17 | 0.4333 | 0 | 0 | 0 |
| ori | 7 | 30 | 30 | 15 | 15 | 0.5 | 0 | 0 | 0 |
| ori | 8 | 30 | 30 | 10 | 20 | 0.3333 | 0 | 0 | 0 |
| ori | 9 | 30 | 30 | 13 | 17 | 0.4333 | 0 | 0 | 0 |
| ori2 | 0 | 20 | 20 | 13 | 7 | 0.65 | 0 | 0 | 0 |
| ori2 | 1 | 20 | 20 | 13 | 7 | 0.65 | 0 | 0 | 0 |
| ori2 | 2 | 20 | 20 | 9 | 11 | 0.45 | 0 | 0 | 0 |
| ori2 | 3 | 20 | 20 | 3 | 17 | 0.15 | 0 | 0 | 0 |
| ori2 | 4 | 20 | 20 | 3 | 17 | 0.15 | 0 | 0 | 0 |
| ori2 | 5 | 20 | 20 | 13 | 7 | 0.65 | 0 | 0 | 0 |
| ori2 | 6 | 20 | 20 | 13 | 7 | 0.65 | 0 | 0 | 0 |
| ori2 | 7 | 20 | 20 | 11 | 9 | 0.55 | 0 | 0 | 0 |
| ori2 | 8 | 20 | 20 | 5 | 15 | 0.25 | 0 | 0 | 0 |
| ori2 | 9 | 20 | 20 | 8 | 12 | 0.4 | 0 | 0 | 0 |
| obj | 0 | 20 | 20 | 0 | 20 | 0.0 | 0 | 0 | 0 |
| obj | 1 | 20 | 20 | 14 | 6 | 0.7 | 0 | 0 | 0 |
| obj | 2 | 20 | 20 | 9 | 11 | 0.45 | 0 | 0 | 0 |
| obj | 3 | 20 | 20 | 8 | 12 | 0.4 | 0 | 0 | 0 |
| obj | 4 | 20 | 20 | 6 | 14 | 0.3 | 0 | 0 | 0 |
| obj | 5 | 20 | 20 | 0 | 20 | 0.0 | 0 | 0 | 0 |
| obj | 6 | 20 | 20 | 4 | 16 | 0.2 | 0 | 0 | 0 |
| obj | 7 | 20 | 20 | 0 | 20 | 0.0 | 0 | 0 | 0 |
| obj | 8 | 20 | 20 | 4 | 16 | 0.2 | 0 | 0 | 0 |
| obj | 9 | 20 | 20 | 2 | 18 | 0.1 | 0 | 0 | 0 |
| lan | 0 | 20 | 20 | 7 | 13 | 0.35 | 0 | 0 | 0 |
| lan | 1 | 20 | 20 | 13 | 7 | 0.65 | 0 | 0 | 0 |
| lan | 2 | 20 | 20 | 11 | 9 | 0.55 | 0 | 0 | 0 |
| lan | 3 | 20 | 20 | 7 | 13 | 0.35 | 0 | 0 | 0 |
| lan | 4 | 20 | 20 | 1 | 19 | 0.05 | 0 | 0 | 0 |
| lan | 5 | 20 | 20 | 13 | 7 | 0.65 | 0 | 0 | 0 |
| lan | 6 | 20 | 20 | 5 | 15 | 0.25 | 0 | 0 | 0 |
| lan | 7 | 20 | 20 | 15 | 5 | 0.75 | 0 | 0 | 0 |
| lan | 8 | 20 | 20 | 4 | 16 | 0.2 | 0 | 0 | 0 |
| lan | 9 | 20 | 20 | 9 | 11 | 0.45 | 0 | 0 | 0 |
| swap | 0 | 5 | 5 | 0 | 5 | 0.0 | 0 | 0 | 0 |
| swap | 1 | 5 | 5 | 0 | 5 | 0.0 | 0 | 0 | 0 |
| swap | 2 | 5 | 5 | 0 | 5 | 0.0 | 0 | 0 | 0 |
| swap | 3 | 5 | 5 | 0 | 5 | 0.0 | 0 | 0 | 0 |
| swap | 4 | 5 | 5 | 0 | 5 | 0.0 | 0 | 0 | 0 |
| swap | 5 | 5 | 5 | 0 | 5 | 0.0 | 0 | 0 | 0 |
| swap | 6 | 5 | 5 | 0 | 5 | 0.0 | 0 | 0 | 0 |
| swap | 7 | 5 | 5 | 0 | 5 | 0.0 | 0 | 0 | 0 |
| swap | 8 | 5 | 5 | 0 | 5 | 0.0 | 0 | 0 | 0 |
| swap | 9 | 5 | 5 | 0 | 5 | 0.0 | 0 | 0 | 0 |
| task | 0 | 5 | 5 | 3 | 2 | 0.6 | 0 | 0 | 0 |
| task | 1 | 5 | 5 | 0 | 5 | 0.0 | 0 | 0 | 0 |
| task | 2 | 5 | 5 | 0 | 5 | 0.0 | 0 | 0 | 0 |
| task | 3 | 5 | 5 | 0 | 5 | 0.0 | 0 | 0 | 0 |
| task | 4 | 5 | 5 | 0 | 5 | 0.0 | 0 | 0 | 0 |
| task | 5 | 5 | 5 | 0 | 5 | 0.0 | 0 | 0 | 0 |
| task | 6 | 5 | 5 | 0 | 5 | 0.0 | 0 | 0 | 0 |
| task | 7 | 5 | 5 | 0 | 5 | 0.0 | 0 | 0 | 0 |
| task | 8 | 5 | 5 | 1 | 4 | 0.2 | 0 | 0 | 0 |
| task | 9 | 5 | 5 | 0 | 5 | 0.0 | 0 | 0 | 0 |
