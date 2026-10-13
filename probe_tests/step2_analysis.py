"""Step 2 analysis: does grader_parity_cheat_vs_clean behave like a text classifier?

Primary layer 36, fixed before looking (probe.py's layer); all layers reported as a sweep.
Rollout-level score: cosine of the CoT-mean residual with the row (the vector's own statistic: it was built from unit-normalized
CoT means). Prefix-level scores on Ethan's two resampled rollouts: the projection at the prefix's last token, and the running
mean of projections over the CoT so far.

Compared against TF-IDF (text_baselines.py) on the same rollouts, and against resampled P(cheat) (step3_analysis.py).

    uv run --no-project --with scikit-learn --with numpy --with torch python probe_tests/step2_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import torch as t
from sklearn.metrics import roc_auc_score

SETS = Path("data/probe_tests/sets")
SCORES = Path("data/probe_tests/probe_scores")
TB = Path("data/probe_tests/text_baselines")
OUT = Path("data/probe_tests/step2")
LAYER = 36
RESAMPLED = {"cheating": "2026-10-02T12-48-29-00-00_grader_M2iuL7zk7KphWhNa6ev3Wi/parity.odd.hack.none.36/1",
             "clean": "2026-10-02T13-12-46-00-00_grader_VwSvUNosNaLtivxYs4b7MT/parity.odd.hack.none.127/1"}
STRETCHES = [(100, 185), (186, 204), (205, 214), (215, 279), (280, 330), (331, 400), (401, 459), (460, 499), (500, 620)]


def load_jsonl(path):
    return [json.loads(line) for line in path.open()]


def auc(y, s):
    y = np.asarray(y, dtype=int)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def side_auc(rows, s):
    vals = []
    for side in sorted({r["side"] for r in rows}):
        m = np.array([r["side"] == side for r in rows])
        y = np.array([r["cheat"] for r in rows])[m]
        if min(y.sum(), len(y) - y.sum()) >= 5:
            vals.append(auc(y, np.asarray(s)[m]))
    return float(np.mean(vals)) if vals else float("nan")


#%% load scores
sc = {}
for shard in sorted(SCORES.glob("shard_*.pt")):
    d = t.load(shard)
    for i, rid in enumerate(d["rollout_id"]):
        sc[rid] = {k: d[k][i] for k in d}
print(f"{len(sc)} rollouts scored")
meta = {r["rollout_id"]: r for r in load_jsonl(SETS / "natural_pool.jsonl") + load_jsonl(SETS / "sysprompt_pool.jsonl")}
tfidf = {r["rollout_id"]: r for r in load_jsonl(TB / "test_scores.jsonl")}
pairs = load_jsonl(SETS / "length_matched_pairs.jsonl")
pair_ids = {p["cheat_id"] for p in pairs} | {p["clean_id"] for p in pairs}


def cos_all_layers(s):
    p = s["proj"].float()
    return (p[:, s["cot_start"]:s["cot_end"]].mean(1) / s["cot_mean_norm"].float()).numpy()  # [n_layers]


#%% rollout-level AUCs by group and subset
OUT.mkdir(parents=True, exist_ok=True)
rows_out = []
groups = {"parity_heldout": lambda m, s: s["group"] == "parity_heldout", "parity_insample": lambda m, s: s["group"] == "parity",
          "sign": lambda m, s: s["group"] == "sign", "pets_listed": lambda m, s: s["group"] == "pets_listed"}
print(f"\nrollout-level AUC (cosine of CoT mean with the row), layer {LAYER}; TF-IDF trained on parity, same test-split rows")
for gname, gf in groups.items():
    rs = [(meta[rid], s) for rid, s in sc.items() if rid in meta and gf(meta[rid], s)]
    if not rs:
        continue
    cos = np.stack([cos_all_layers(s) for _, s in rs])  # [n, n_layers]
    m = [r for r, _ in rs]
    y = [r["cheat"] for r in m]
    subsets = {"all": np.ones(len(m), bool), "maxrew": np.array([r["maxrew"] for r in m]), "lenpair": np.array([r["rollout_id"] in pair_ids for r in m]),
               "test_split": np.array([r["rollout_id"] in tfidf for r in m])}
    for sub, mask in subsets.items():
        if mask.sum() == 0:
            continue
        mm = [r for r, k in zip(m, mask) if k]
        yy = [r["cheat"] for r in mm]
        a = auc(yy, cos[mask, LAYER])
        sa = side_auc(mm, cos[mask, LAYER])
        best = int(np.nanargmax([auc(yy, cos[mask, L]) for L in range(cos.shape[1])]))
        tf = [tfidf.get(r["rollout_id"], {}).get("tfidf@parity") for r in mm]
        tf_auc = auc(yy, tf) if all(v is not None for v in tf) else float("nan")  # TF-IDF scores exist for test-split rows only
        row = {"group": gname, "subset": sub, "n": len(mm), "n_cheat": int(sum(yy)), "probe_auc_L36": a, "probe_side_auc_L36": sa,
               "best_layer": best, "probe_auc_best": auc(yy, cos[mask, best]), "tfidf_parity_auc": tf_auc}
        rows_out.append(row)
        print(f"  {gname:16s} {sub:10s} n={len(mm):5d} cheat={sum(yy):5d}  probe L36 {a:.3f} (side {sa:.3f})  best L{best} {row['probe_auc_best']:.3f}  tfidf@parity {tf_auc:.3f}")
(OUT / "rollout_auc.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_out))

#%% Ethan's rollouts: prefix-level scores against P(cheat) and TF-IDF
curves = load_jsonl(Path("data/probe_tests/resample/step3_curves.jsonl"))
print(f"\nprefix scores at layer {LAYER}: last-token projection and running CoT mean, pooled by stretch")
prefix_rows = []
for name, rid in RESAMPLED.items():
    if rid not in sc:
        continue
    s = sc[rid]
    p = s["proj"].float()[LAYER]
    cot0 = s["cot_start"] + 1  # t counts tokens after "<think>\n"; cot_start is the "\n"
    run = np.cumsum(p[s["cot_start"]:].numpy()) / np.arange(1, len(p) - s["cot_start"] + 1)
    for c in (c for c in curves if c["rollout"] == name):
        tt = c["t"]
        last = cot0 + tt - 1  # last token of the t-token prefix (t = 0: the "\n")
        if last >= s["cot_end"]:
            continue
        prefix_rows.append({**{k: c[k] for k in ["rollout", "t", "k", "n", "p_naive", "tfidf_prob"]},
                            "probe_last": float(p[last]), "probe_runmean": float(run[last - s["cot_start"]])})
(OUT / "prefix_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prefix_rows))
for name in RESAMPLED:
    pr = [r for r in prefix_rows if r["rollout"] == name]
    if not pr:
        continue
    print(f"  {name}")
    for a, b in STRETCHES:
        g = [r for r in pr if a <= r["t"] <= b]
        if not g:
            continue
        k, n = sum(r["k"] for r in g), sum(r["n"] for r in g)
        print(f"    t {a:4d}-{b:4d}: P(cheat) {k / n:.2f} (n={n:4d})  tfidf {np.mean([r['tfidf_prob'] for r in g]):.2f}"
              f"  probe last {np.mean([r['probe_last'] for r in g]):+.3f}  runmean {np.mean([r['probe_runmean'] for r in g]):+.3f}")
    # rank agreement with smoothed P(cheat) over the whole trace
    tt = np.array([r["t"] for r in pr])
    pk = np.convolve([r["p_naive"] for r in pr], np.ones(25) / 25, mode="same")
    from scipy.stats import spearmanr
    for key in ["tfidf_prob", "probe_last", "probe_runmean"]:
        rho = spearmanr(pk, [r[key] for r in pr]).correlation
        print(f"    Spearman with 25-position mean P(cheat): {key:14s} {rho:+.2f}")
