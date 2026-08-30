"""
compute_constraints.py

Step 1 of the consistency-constraint study: one scalar per timestep per rollout, for
each of 130 constraint series, over the full 300-rollout v2 corpus.

Reads /data/rollouts_v2 (143.4 GB of hidden.pt) exactly once. The corpus disk saturates
at ~0.31 GB/s regardless of reader concurrency, so this is I/O-bound end to end and runs
single-threaded on purpose: per-rollout compute is ~0.4 s against ~1.6 s of read, and
mmap'ing a position slice measured *slower* than a full load (readahead pulls the whole
extent anyway). Expect ~10 minutes.

Memory discipline: exactly one rollout is resident at a time. Nothing accumulates across
the loop except per-rollout summary scalars and a 10-rollout parquet buffer.

Layer semantics (verified, see checkpoint 0)
--------------------------------------------
capture_v2 stored `generate()`'s `output_hidden_states`. In transformers 4.40.1
`LlamaModel.forward` appends `hidden_states` at the TOP of the block loop and appends
once more after `self.norm`. Therefore, for the 33 stored indices:

    index 0        embedding output
    index 1..31    RAW residual stream, output of decoder block 1..31
    index 32       model.norm(output of decoder block 32)   <- POST-NORM, not raw

Confirmed numerically: ||h_32 / g||_2 = 63.9989 +- 0.02 against an expected exactly
sqrt(4096) = 64, where g = language_model.model.norm.weight. Block 32's raw output is
not stored anywhere in the corpus.

Consequences, per the checkpoint-0 decision:
  * emb_temp  covers L 0..31   (index 32 excluded)
  * xl_adj    covers l 0..30   (the 31->32 pair would straddle the norm)
  * xl_final  anchors to layer 31, not 32
  * xl_final_hN is a secondary single-position variant that recovers block 32's raw
    DIRECTION as h_32/g, exact because RMSNorm is y = x/rms(x) * g so y/g = x/rms(x).
    It is single-position only: the pooled read is a mean of seven normed vectors, and
    mean(x_p/rms_p) is not proportional to mean(x_p), so the pooled anchor is NOT
    recoverable this way.

Sink dimension
--------------
Dim 1512 is a massive-activation / attention-sink channel carrying 30-180x the typical
per-dim magnitude from layer 1 onward; g[1512] = 5.3e-05, i.e. the model suppresses it
before lm_head. It compresses the raw-residual cosines (xl_adj mean 0.078 -> 0.102 and
emb_temp 0.101 -> 0.155 when dropped). Sink-excluded variants are computed alongside the
primary ones at negligible cost and stored in the per-rollout .npz ONLY -- they are kept
out of all.parquet to avoid doubling it. They exist so the choice can be made at
checkpoint 1 without paying for a second 143 GB pass.

Usage:
    source env.sh
    python analysis/compute_constraints.py            # full corpus
    python analysis/compute_constraints.py --limit 4  # smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# --- Locked configuration -----------------------------------------------------------
CORPUS_ROOT = Path("/data/rollouts_v2")
INDEX_JSON = Path("corpus_v2_index.json")
OUT_DIR = Path("constraints")
CKPT_HUB = Path("/data/hf-cache/hub/models--openvla--openvla-7b-finetuned-libero-10")
NORM_WEIGHT_KEY = "language_model.model.norm.weight"

D_MODEL = 4096
N_STORED = 33          # indices 0..32 as written by capture_v2
N_RAW = 32             # indices 0..31 are raw residual reads; 32 is post-norm
SINK_DIM = 1512
N_BINS = 256
VOCAB_SIZE = 32000
DELTA_EPS = 1e-6       # act_dir is undefined below this delta norm
COS_EPS = 1e-12
PARQUET_BUFFER_ROLLOUTS = 10

# Label convention matches probe_layer.py: 1 = failure.
LABEL_NOTE = "1 = failure (i.e. NOT success), matching probe_layer.py"


# --- Small helpers ------------------------------------------------------------------
def load_norm_weight() -> torch.Tensor:
    """g = language_model.model.norm.weight, straight from safetensors. No model load."""
    from safetensors import safe_open

    snaps = sorted((CKPT_HUB / "snapshots").glob("*"))
    if not snaps:
        raise SystemExit(f"no snapshot under {CKPT_HUB}/snapshots")
    snap = snaps[-1]
    weight_map = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    shard = weight_map[NORM_WEIGHT_KEY]
    with safe_open(str(snap / shard), framework="pt", device="cpu") as f:
        g = f.get_tensor(NORM_WEIGHT_KEY).float()
    if g.shape != (D_MODEL,):
        raise SystemExit(f"{NORM_WEIGHT_KEY} has shape {tuple(g.shape)}, expected ({D_MODEL},)")
    return g


def normalised_actions(action_token_ids: np.ndarray) -> np.ndarray:
    """(T, 7) continuous actions in [-1, 1], exactly as ActionTokenizer decodes them.

    Pure numpy; no model and no norm_stats needed. Verified against action_raw: the
    affine fit back to un-normalised units has max|residual| == 0.0 on every dim.
    """
    bins = np.linspace(-1.0, 1.0, N_BINS)
    centers = (bins[:-1] + bins[1:]) / 2.0
    disc = VOCAB_SIZE - action_token_ids.astype(np.int64)
    return centers[np.clip(disc - 1, 0, centers.shape[0] - 1)].astype(np.float32)


def cos_dist_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(1 - cos) over all dims, and over all dims except the sink, in one pass.

    The sink-excluded value is obtained by subtracting the sink term from the dot product
    and from both squared norms, rather than materialising a masked copy of a and b.
    """
    dot = (a * b).sum(-1)
    na2 = (a * a).sum(-1)
    nb2 = (b * b).sum(-1)
    full = 1.0 - dot / (na2.sqrt() * nb2.sqrt()).clamp_min(COS_EPS)

    asd = a[..., SINK_DIM]
    bsd = b[..., SINK_DIM]
    dot_x = dot - asd * bsd
    na2_x = (na2 - asd * asd).clamp_min(0.0)
    nb2_x = (nb2 - bsd * bsd).clamp_min(0.0)
    nosink = 1.0 - dot_x / (na2_x.sqrt() * nb2_x.sqrt()).clamp_min(COS_EPS)
    return full, nosink


def pooled_and_hN(x: torch.Tensor, n_positions: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(T, 33, P, 4096) fp16 -> pooled (T, 33, 4096) fp32 and h[N] (T, 33, 4096) fp32.

    P=3 index 0 is already the mean over the seven decision states, computed by
    capture_v2 as decision.float().mean(0).to(fp16). For P=7 the same mean is recomputed
    here and put through the identical fp32 -> fp16 -> fp32 round trip, so that the 30
    P=7 rollouts (trials 0-2 of every task) are treated bit-for-bit like the other 270
    rather than differing by one rounding step.
    """
    T = x.shape[0]
    if n_positions == 3:
        pooled = x[:, :, 0, :].float()
        hN = x[:, :, 1, :].float()
    elif n_positions == 7:
        acc = torch.zeros(T, N_STORED, D_MODEL, dtype=torch.float32)
        for p in range(7):
            acc += x[:, :, p, :].float()
        acc /= 7.0
        pooled = acc.half().float()          # match the P=3 storage round trip
        hN = x[:, :, 0, :].float()
    else:
        raise ValueError(f"unexpected n_positions {n_positions}")
    return pooled, hN


# --- Constraint families ------------------------------------------------------------
def action_constraints(norm_act: np.ndarray, grip_executed: np.ndarray) -> dict:
    """act_mag / act_dir on normalised dims 0:6; grip_flip on executed[:, 6] in {-1, +1}.

    Index 0 is NaN for all three. act_dir additionally needs two deltas, so index 1 is
    NaN too, and any timestep whose current or previous delta norm falls below
    DELTA_EPS is NaN.
    """
    T = norm_act.shape[0]
    a = norm_act[:, :6].astype(np.float64)

    d = np.full((T, 6), np.nan)
    d[1:] = a[1:] - a[:-1]
    dn = np.linalg.norm(d, axis=1)

    act_mag = dn.astype(np.float32)                      # NaN at t=0 by construction

    act_dir = np.full(T, np.nan)
    ok = np.zeros(T, dtype=bool)
    ok[2:] = (dn[2:] > DELTA_EPS) & (dn[1:-1] > DELTA_EPS)
    idx = np.where(ok)[0]
    if idx.size:
        num = (d[idx] * d[idx - 1]).sum(1)
        act_dir[idx] = 1.0 - num / (dn[idx] * dn[idx - 1])

    grip_flip = np.full(T, np.nan)
    s = np.sign(grip_executed.astype(np.float64))
    grip_flip[1:] = (s[1:] != s[:-1]).astype(np.float64)

    return {
        "act_mag": act_mag,
        "act_dir": act_dir.astype(np.float32),
        "grip_flip": grip_flip.astype(np.float32),
    }


def hidden_constraints(pooled: torch.Tensor, hN: torch.Tensor, g: torch.Tensor) -> dict:
    """All embedding-temporal and cross-layer series. See the module docstring for why
    index 32 is excluded from the primary families."""
    T = pooled.shape[0]
    raw = pooled[:, :N_RAW, :]                      # (T, 32, 4096), indices 0..31

    out: dict[str, np.ndarray] = {}

    # (b) emb_temp[L][t] = 1 - cos(h_t^L, h_{t-1}^L), L in 0..31. Row 0 is NaN.
    full, nosink = cos_dist_pair(raw[1:], raw[:-1])
    for key, val in (("emb_temp", full), ("emb_temp_nosink", nosink)):
        arr = np.full((T, N_RAW), np.nan, dtype=np.float32)
        arr[1:] = val.numpy()
        out[key] = arr

    # (c) xl_adj[l][t] = 1 - cos(h_t^l, h_t^{l+1}), l in 0..30.
    full, nosink = cos_dist_pair(raw[:, :N_RAW - 1, :], raw[:, 1:N_RAW, :])
    out["xl_adj"] = full.numpy().astype(np.float32)
    out["xl_adj_nosink"] = nosink.numpy().astype(np.float32)

    # anchored to layer 31 (NOT 32, which is post-norm), l in 0..30.
    anchor = raw[:, N_RAW - 1:N_RAW, :]
    full, nosink = cos_dist_pair(raw[:, :N_RAW - 1, :], anchor)
    out["xl_final"] = full.numpy().astype(np.float32)
    out["xl_final_nosink"] = nosink.numpy().astype(np.float32)

    out["xl_spread"] = out["xl_adj"].mean(axis=1).astype(np.float32)
    out["xl_spread_nosink"] = out["xl_adj_nosink"].mean(axis=1).astype(np.float32)

    # Secondary variant: single-position h[N], anchored to block 32's raw DIRECTION,
    # recovered exactly as h_32 / g. l in 0..31, so this one DOES span all raw layers.
    u = hN[:, N_RAW, :] / g
    full, nosink = cos_dist_pair(hN[:, :N_RAW, :], u.unsqueeze(1))
    out["xl_final_hN"] = full.numpy().astype(np.float32)
    out["xl_final_hN_nosink"] = nosink.numpy().astype(np.float32)

    return out


# --- Parquet emission ---------------------------------------------------------------
# Only the primary families reach the parquet; *_nosink stay in the .npz.
PARQUET_SERIES = [
    ("act_mag", False), ("act_dir", False), ("grip_flip", False), ("xl_spread", False),
    ("emb_temp", True), ("xl_adj", True), ("xl_final", True), ("xl_final_hN", True),
]


def rollout_rows(series: dict, meta: dict) -> dict:
    """Long form for one rollout: 130 series x T timesteps."""
    T = meta["T"]
    t = np.arange(T, dtype=np.int16)
    t_norm = t.astype(np.float32) / float(T)   # literally t/T, per the brief

    names, layers, values, ts = [], [], [], []
    for name, layered in PARQUET_SERIES:
        arr = series[name]
        cols = arr.shape[1] if layered else 1
        for c in range(cols):
            v = arr[:, c] if layered else arr
            values.append(v.astype(np.float32))
            ts.append(t)
            layers.append(np.full(T, c if layered else -1, dtype=np.int16))
            names.extend([name] * T)

    n = len(names)
    return {
        "rollout_id": [meta["rollout_id"]] * n,
        "task_id": np.full(n, meta["task_id"], dtype=np.int16),
        "outcome": np.full(n, meta["outcome"], dtype=np.int8),
        "outcome_final": np.full(n, meta["outcome_final"], dtype=np.int8),
        "outcome_group": [meta["outcome_group"]] * n,
        "t_first_success": np.full(n, meta["t_first_success"], dtype=np.float32),
        "t": np.concatenate(ts),
        "T": np.full(n, T, dtype=np.int16),
        "t_norm": np.tile(t_norm, n // T),
        "constraint_name": names,
        "layer": np.concatenate(layers),
        "value": np.concatenate(values),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", type=str, default=str(CORPUS_ROOT))
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--limit", type=int, default=None, help="first N rollouts (smoke test)")
    ap.add_argument("--force", action="store_true", help="recompute even if the .npz exists")
    ap.add_argument("--no-parquet", action="store_true")
    args = ap.parse_args()

    pa = pq = None
    if not args.no_parquet:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise SystemExit(
                "pyarrow is not installed in this venv and all.parquet is a required "
                "output.\n    pip install pyarrow\n"
                "Or pass --no-parquet to write only the per-rollout .npz files."
            )

    root = Path(args.corpus_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = json.loads(INDEX_JSON.read_text())
    entries = index["rollouts"]
    if args.limit:
        entries = entries[: args.limit]

    g = load_norm_weight()
    print(f"[*] g = {NORM_WEIGHT_KEY}: mean {g.mean():.4f} sd {g.std():.4f} "
          f"min {g.min():.2e} max {g.max():.4f}")
    print(f"[*] {len(entries)} rollouts -> {out_dir}   ({LABEL_NOTE})")

    writer = None
    buf: list[dict] = []
    summary: list[dict] = []
    checked_g = False
    t_start = time.time()

    for i, ent in enumerate(entries, 1):
        rid = ent["rollout_id"]
        d = root / rid
        npz_path = out_dir / f"{rid}.npz"

        success_ever = bool(ent["success_ever"])
        success_final = bool(ent["success_final"])
        meta = {
            "rollout_id": rid,
            "task_id": int(ent["task_idx"]),
            "T": int(ent["T_total"]),
            # 1 = failure, matching probe_layer.py.
            "outcome": int(not success_ever),
            "outcome_final": int(not success_final),
            "outcome_group": ("success_clean" if success_ever and success_final
                              else "success_lost" if success_ever
                              else "never"),
            "t_first_success": (float(ent["t_success"]) if ent.get("t_success") is not None
                                else float("nan")),
        }

        if npz_path.exists() and not args.force:
            z = np.load(npz_path)
            series = {k: z[k] for k in z.files if not k.startswith("_")}
        else:
            man = json.loads((d / "manifest.json").read_text())
            P = int(man["n_positions"])
            T = int(man["T_total"])
            if T != meta["T"] or int(man["layers_stored"]) != N_STORED:
                raise SystemExit(f"{rid}: manifest disagrees with the index")

            acts = np.load(d / "actions.npz")
            pe = acts["policy_step_env_t"]
            expect = np.arange(man["num_steps_wait"], man["num_steps_wait"] + T)
            if not np.array_equal(pe, expect):
                raise SystemExit(f"{rid}: policy_step_env_t is not contiguous from the wait offset")

            series = action_constraints(
                normalised_actions(acts["action_token_ids"]),
                acts["executed"][pe][:, 6],
            )

            x = torch.load(d / "hidden.pt", map_location="cpu")
            if tuple(x.shape) != (T, N_STORED, P, D_MODEL):
                raise SystemExit(f"{rid}: hidden.pt is {tuple(x.shape)}, expected {(T, N_STORED, P, D_MODEL)}")
            pooled, hN = pooled_and_hN(x, P)
            del x

            if not checked_g:
                # RMSNorm identity: ||h_32 / g||_2 must be sqrt(4096) = 64 exactly.
                nrm = (hN[:, N_RAW, :] / g).norm(dim=-1)
                print(f"[*] RMSNorm check on {rid}: ||h_32/g|| mean {nrm.mean():.4f} "
                      f"sd {nrm.std():.5f} (expect 64.0000)")
                if abs(float(nrm.mean()) - np.sqrt(D_MODEL)) > 0.1:
                    raise SystemExit("index 32 is not RMSNorm(x)*g with this g -- stop and re-verify")
                checked_g = True

            series.update(hidden_constraints(pooled, hN, g))
            del pooled, hN
            np.savez_compressed(npz_path, **series)

        am = series["act_mag"][1:]
        summary.append({
            "rollout_id": rid, "task_id": meta["task_id"],
            "outcome_group": meta["outcome_group"],
            "act_mag_median": float(np.nanmedian(am)),
            "act_mag_p95": float(np.nanpercentile(am, 95)),
            "grip_flip_rate": float(np.nanmean(series["grip_flip"])),
            "xl_spread_mean": float(np.nanmean(series["xl_spread"])),
        })

        if not args.no_parquet:
            buf.append(rollout_rows(series, meta))
            if len(buf) >= PARQUET_BUFFER_ROLLOUTS or i == len(entries):
                cols = {k: (pa.array(sum((b[k] for b in buf), []))
                            if isinstance(buf[0][k], list)
                            else pa.array(np.concatenate([b[k] for b in buf])))
                        for k in buf[0]}
                table = pa.table(cols)
                if writer is None:
                    writer = pq.ParquetWriter(out_dir / "all.parquet", table.schema,
                                              compression="zstd")
                writer.write_table(table)
                buf = []

        if i % 25 == 0 or i == len(entries):
            el = time.time() - t_start
            rate = el / i
            print(f"[{i:3d}/{len(entries)}] elapsed {el/60:5.1f} min  "
                  f"{rate:.2f} s/rollout  eta {(len(entries)-i)*rate/60:5.1f} min",
                  flush=True)

    if writer is not None:
        writer.close()

    med = float(np.median([s["act_mag_median"] for s in summary]))
    for s in summary:
        s["act_mag_median_ratio"] = s["act_mag_median"] / med if med else float("nan")
        s["act_mag_flagged"] = bool(s["act_mag_median_ratio"] > 3.0)
    flagged = [s["rollout_id"] for s in summary if s["act_mag_flagged"]]

    (out_dir / "summary.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_rollouts": len(summary),
        "label_note": LABEL_NOTE,
        "layer_semantics": {
            "index_32": "post-model.norm; excluded from emb_temp/xl_adj, not used as xl_final anchor",
            "emb_temp_layers": "0..31", "xl_adj_layers": "0..30",
            "xl_final_anchor": 31, "xl_final_hN_layers": "0..31",
        },
        "sink_dim": SINK_DIM,
        "act_mag_corpus_median": med,
        "act_mag_flagged_rollouts": flagged,
        "per_rollout": summary,
    }, indent=1))

    print(f"\n[*] act_mag corpus median {med:.4f}; "
          f"{len(flagged)} rollouts flagged at >3x: {flagged if flagged else '(none)'}")
    print(f"[*] done in {(time.time()-t_start)/60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
