# Selections: original vs full training set

*Background run (results_collection_handoff.md §6). Every selection was re-made on each training set's own eval-seen rollouts by the rules the Methods describe; unseen and stress rollouts chose nothing. The thesis Methods quotes the original-set values. Source: `tables/selections_comparison.csv`.*

| Selection | Original training set | Full training set | Same |
|---|---|---|---|
| smoothing beta, action family | 0.96 | 0.83 | no |
| smoothing beta, emb_temp family | none | none | yes |
| smoothing beta, cross-layer family | 0.91 | 0.96 | no |
| untrained constraint layer, pooled rule (earlier) | emb_temp[28] | emb_temp[29] | no |
| c, C's probes | 1 | 0.001 | no |
| c, C's 132-input logistic regression | 1e-05 | 0.0001 | no |
| c, A's probes | 1e-05 | 1e-06 | no |
| A's single best probe per seed | layer01 / layer10 / layer02 | layer10 / layer31 / layer11 | no |
| sign-unstable constraints at the chosen beta | 14: emb_temp[15] emb_temp[1] xl_adj[10] xl_adj[11] xl_adj[12] xl_adj[14] xl_adj[15] xl_adj[17] xl_adj[21] xl_adj[22] xl_adj[24] xl_adj[6] xl_final[17] xl_final[18] | 20: emb_temp[11] emb_temp[12] emb_temp[13] xl_adj[10] xl_adj[14] xl_adj[15] xl_adj[16] xl_adj[17] xl_adj[19] xl_adj[29] xl_adj[30] xl_adj[6] xl_final[18] xl_final[19] xl_final[20] xl_final[24] xl_final[26] xl_final[28] xl_final[29] xl_final[30] | no |
| c, A's trained combiner | 0.001 | 1e-05 | no |
| c, C's trained combiner | 0.0001 | 0.001 | no |
| c, B's trained combiner, H = 0 | 0.01 | 1.0 | no |
| c, B's trained combiner, H = 25 | 1e-05 | 0.0001 | no |
| c, B's trained combiner, H = 10 (refit) | 0.001 | 0.01 | no |
| C's single best probe per seed | layer00 / action / layer03 | action / action / layer05 | no |
| B's single best probe per seed, H = 10 | L20:emb_temp:H10 / L12:xl_final:H10 / L00:emb_temp:H10 | L30:emb_temp:H10 / L15:emb_temp:H10 / L00:emb_temp:H10 | no |
| B's single best probe per seed, H = 0 | L20:emb_temp:H0 / L12:xl_final:H0 / L00:emb_temp:H0 | L29:emb_temp:H0 / L00:emb_temp:H0 / L00:act_rep:H0 | no |
| B's single best probe per seed, H = 25 | L07:emb_temp:H25 / L09:emb_temp:H25 / L27:emb_temp:H25 | L00:emb_temp:H25 / L17:emb_temp:H25 / L18:emb_temp:H25 | no |
| best untrained constraint, per-task rule | emb_temp[26] | emb_temp[26] | yes |
| smoothing beta, token mean entropy | 0.96 | 0.83 | no |
| smoothing beta, token max-probability | 0.83 | 0.83 | yes |
| B's grid c | 0.0001 | 0.0001 | yes |
| B's probe count per H | 124 | 116 | no |
| B's skipped (sign-unstable) targets | emb_temp[15] emb_temp[1] xl_final[17] xl_final[18] | emb_temp[11] emb_temp[12] emb_temp[13] xl_final[18] xl_final[19] xl_final[20] xl_final[24] xl_final[26] xl_final[28] xl_final[29] xl_final[30] | no |
| weight decay, C's 2-layer MLP | 0.01 | 0.0001 | no |
| weight decay, B's 2-layer MLP | 1.0 | 1.0 | yes |
