# Step 8 regression gate (D0 on T0 vs results/abc)

**PASSED** — run 2026-10-11T13:28:48Z. Stage wall times (timings.csv, all runs; a re-run skips finished work): stage2_seeds 30 s, stage2_total 30 s, stage3_grid 72 s, stage3_total 72 s, all 103 s, stage4_grid 3797 s, stage4_final 118 s, stage4_final 56 s, stage4_final 86 s, stage4_total 4058 s, stage6 1 s, all 4059 s, stage6 2 s, all 2 s.

Unseen = T0, mean over seeds 0-2. Port: analysis/abc2/pipeline.py on a v2-only index, old cache /data/tmp/abc_features, splits from data2.make_split2 (equal to results/abc/splits).

| Quantity | Reference | Port | Diff | Tolerance | Pass | Port per seed |
|---|---|---|---|---|---|---|
| emb_temp[28] alone, M1-per-task | 0.7250 | 0.7250 | +0.0000 | ±0.005 | yes | 0.6625, 0.7656, 0.7471 |
| C final, M1-per-task | 0.7169 | 0.7169 | +0.0000 | ±0.005 | yes | 0.6901, 0.7733, 0.6874 |
| A final, M1-per-task | 0.7163 | 0.7163 | +0.0000 | ±0.005 | yes | 0.7520, 0.6901, 0.7068 |
| emb_temp[28], balanced accuracy, alpha 0.1, cut | 0.5559 | 0.5559 | -0.0000 | ±0.01 | yes | 0.4896, 0.6780, 0.5000 |
| A, balanced accuracy, alpha 0.1, cut | 0.5368 | 0.5368 | +0.0000 | ±0.01 | yes | 0.5476, 0.5628, 0.5000 |
| C, balanced accuracy, alpha 0.1, cut | 0.5009 | 0.5009 | +0.0000 | ±0.01 | yes | 0.5000, 0.5000, 0.5028 |

## Informational parity (not gated)

| Selection | abc | port | same |
|---|---|---|---|
| beta_action | 0.96 | 0.96 | True |
| beta_emb_temp | none | none | True |
| beta_cross_layer | 0.91 | 0.91 | True |
| emb_layer | emb_temp[28] | emb_temp[28] | True |
| c_C_probes | 1 | 1 | True |
| c_C_final | 1e-05 | 1e-05 | True |
| c_A | 1e-05 | 1e-05 | True |
| A_probe_seed0 | layer01 | layer01 | True |
| A_probe_seed1 | layer10 | layer10 | True |
| A_probe_seed2 | layer02 | layer02 | True |

- Constraint M1 per seed (eval_seen and unseen, pooled and per-task): 2508 rows, max |diff| 0.00005 (abc CSV rounded to 4 d.p.); sign strings differing: 0.
- C probes, unseen per-task per seed: max |diff| 0.00005.
- A probes, unseen per-task per seed: max |diff| 0.00005 (mean 0.00002; GPU L-BFGS).
- CP balanced accuracy, every alpha and seed (cut): A max |diff| 0.0000 (n=48); C max |diff| 0.0000 (n=48); emb max |diff| 0.0000 (n=48).

Split check: {"seed0": {"train": true, "eval_seen": true, "unseen": true, "unseen_tasks": true, "stress_empty": true}, "seed1": {"train": true, "eval_seen": true, "unseen": true, "unseen_tasks": true, "stress_empty": true}, "seed2": {"train": true, "eval_seen": true, "unseen": true, "unseen_tasks": true, "stress_empty": true}}
