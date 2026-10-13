"""Step 2b: how much of grader_parity_cheat_vs_clean is a word detector?

Lexical model: the mean per-token projection of each token id, fit on the CoT tokens of the in-sample parity rollouts (the
vector's own training data). A rollout's lexical expectation is the probe score that model predicts from its tokens alone.
On rollouts the vector never saw (held-out parity, sign, pets_listed) compare cheat-vs-clean AUC of:
  probe     the actual score (CoT-mean projection divided by the CoT-mean norm, the vector's cosine statistic)
  lexical   the lexical expectation, normalized the same way
  residual  probe minus lexical
If residual carries the separation, the probe reads more than word identity. Layer 36, fixed in advance; other layers as a sweep.

    /tmp/john-scratch/rs-venv/bin/python probe_tests/step2_lexical.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch as t
from sklearn.metrics import roc_auc_score

SETS = Path("data/probe_tests/sets")
SCORES = Path("data/probe_tests/probe_scores")
GRADER = Path("data/inspect/grader")
TB = Path("data/probe_tests/text_baselines")
OUT = Path("data/probe_tests/step2")
LAYERS = [20, 36, 51, 63]
PRIMARY = 36
MIN_COUNT = 5


def load_jsonl(path):
    return [json.loads(line) for line in path.open()]


def auc(y, s):
    y = np.asarray(y, dtype=int)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


#%% load the layers of interest only
proj, span, norm, group = {}, {}, {}, {}
for shard in sorted(SCORES.glob("shard_*.pt")):
    d = t.load(shard)
    for i, rid in enumerate(d["rollout_id"]):
        proj[rid] = d["proj"][i][LAYERS].float().numpy()
        span[rid] = (d["cot_start"][i], d["cot_end"][i])
        norm[rid] = d["cot_mean_norm"][i][LAYERS].float().numpy()
        group[rid] = d["group"][i]
meta = {r["rollout_id"]: r for r in load_jsonl(SETS / "natural_pool.jsonl") + load_jsonl(SETS / "sysprompt_pool.jsonl")}
ids = {}
for f in sorted({meta[r]["file"] for r in proj if r in meta}):
    for line in (GRADER / f).open():
        rec = json.loads(line)
        if rec["rollout_id"] in proj:
            ids[rec["rollout_id"]] = rec["ids"]
print(f"{len(proj)} rollouts; groups {dict((g, sum(v == g for v in group.values())) for g in sorted(set(group.values())))}")

#%% lexical model on in-sample parity CoT tokens
fit = [r for r in proj if group[r] == "parity"]
sums = [defaultdict(float) for _ in LAYERS]
counts = defaultdict(int)
for rid in fit:
    a, b = span[rid]
    toks = ids[rid][a:b]
    for j, tid in enumerate(toks):
        counts[tid] += 1
        for li in range(len(LAYERS)):
            sums[li][tid] += proj[rid][li, a + j]
glob_mean = [sum(s.values()) / sum(counts.values()) for s in sums]
print(f"lexical model: {len(fit)} parity rollouts, {sum(counts.values())} tokens, {sum(c >= MIN_COUNT for c in counts.values())} token ids with >= {MIN_COUNT}")


def lexical_scores(rid):
    a, b = span[rid]
    toks = ids[rid][a:b]
    out = []
    for li in range(len(LAYERS)):
        pred = [sums[li][tid] / counts[tid] if counts[tid] >= MIN_COUNT else glob_mean[li] for tid in toks]
        out.append(np.mean(pred))
    return np.array(out)


#%% evaluate on rollouts the vector never saw
tfidf = {r["rollout_id"]: r for r in load_jsonl(TB / "test_scores.jsonl")}
pairs = load_jsonl(SETS / "length_matched_pairs.jsonl")
pair_ids = {p["cheat_id"] for p in pairs} | {p["clean_id"] for p in pairs}
results = []
for g in ["parity_heldout", "sign", "pets_listed", "resampled"]:
    rids = [r for r in proj if group[r] == g and (r in meta or g == "resampled")]
    if g == "resampled":
        for r in rids:
            a, b = span[r]
            print(f"  resampled {r[-28:]}: probe cos L36 {proj[r][LAYERS.index(36), a:b].mean() / norm[r][LAYERS.index(36)]:.4f}")
        continue
    y = np.array([meta[r]["cheat"] for r in rids])
    actual = np.stack([np.array([proj[r][li, span[r][0]:span[r][1]].mean() for li in range(len(LAYERS))]) for r in rids])
    lex = np.stack([lexical_scores(r) for r in rids])
    nrm = np.stack([norm[r] for r in rids])
    subsets = {"all": np.ones(len(rids), bool), "maxrew": np.array([meta[r]["maxrew"] for r in rids]),
               "lenpair": np.array([r in pair_ids for r in rids]), "test_split": np.array([r in tfidf for r in rids])}
    for sub, m in subsets.items():
        if m.sum() < 10:
            continue
        for li, L in enumerate(LAYERS):
            pc, lc = actual[m, li] / nrm[m, li], lex[m, li] / nrm[m, li]
            row = {"group": g, "subset": sub, "layer": L, "n": int(m.sum()), "n_cheat": int(y[m].sum()),
                   "auc_probe": auc(y[m], pc), "auc_lexical": auc(y[m], lc), "auc_residual": auc(y[m], pc - lc),
                   "corr_probe_lexical": float(np.corrcoef(pc, lc)[0, 1])}
            if sub == "test_split":
                row["auc_tfidf_parity"] = auc(y[m], [tfidf[r]["tfidf@parity"] for r, k in zip(rids, m) if k])
            results.append(row)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "lexical_residual.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
print("\nAUC cheat vs clean on rollouts the vector never saw (cosine statistic)")
for r in results:
    if r["layer"] == PRIMARY or r["subset"] == "all":
        extra = f"  tfidf@parity {r['auc_tfidf_parity']:.3f}" if "auc_tfidf_parity" in r else ""
        print(f"  {r['group']:15s} {r['subset']:10s} L{r['layer']:2d} n={r['n']:5d} cheat={r['n_cheat']:5d}  probe {r['auc_probe']:.3f}"
              f"  lexical {r['auc_lexical']:.3f}  residual {r['auc_residual']:.3f}  corr(probe, lexical) {r['corr_probe_lexical']:.2f}{extra}")
