# thesis-introspection

Runtime failure detection for OpenVLA (7B, 33 layers, d_model=4096) on LIBERO-10.

## Layout

    collect/     produces the corpus  (writes /data/corpus)
    analysis/    consumes the corpus  (writes ./results)

Nothing in `analysis/` imports from `collect/`, or vice versa. The only coupling is
the corpus on disk and its manifest schema.

### collect/
    capture.py                schema v2 lib: teacher-forced re-forward, 8 positions
    gen_rollouts.py           v1 driver  -> capture.py
    capture_v2.py             v2 lib: states straight from generate(), 7 positions
    gen_rollouts_v2.py        v2 driver  -> capture_v2.py
    make_init_assignment.py   writes collect/init_state_assignment.json

The v1 pair is **frozen, not dead**: it is what the existing 50-rollout corpus and its
analysis were produced with. v2 is a methodological rewrite (no re-forward, so no
fp-nondeterminism and no parity check) — see the header of `capture_v2.py`.

### analysis/
    probe_layer.py            shared lib + single-layer probe (load_features, fit_and_score)
    control_diagnostic.py     permutation/split null distributions   -> probe_layer
    analyse_control.py        significance tests over those distributions
    probe_late_window.py      scheme B, final 20% of each rollout by normalised time
    extract_all_layers.py     caches per-layer features to /data/tmp
    plot_auroc_by_layer.py    layer sweep -> figure + csv
    project_hidden_states.py  projection visualisations

## Running

**Always run from the repo root**, with the path prefix:

    source env.sh
    python analysis/probe_layer.py

Results dirs (`results/probe_pilot`, ...) are resolved relative to the *current working
directory*, not the script, so running from inside `analysis/` silently writes results
to `analysis/results/`. The layer sweep assumes the same:

    python analysis/extract_all_layers.py    # fill the feature cache first
    ./run_all_layers.sh                      # 33 layers, 8 at a time
    python analysis/plot_auroc_by_layer.py
