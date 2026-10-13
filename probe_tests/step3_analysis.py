"""Step 3 analysis: resampled P(cheat | first t reasoning tokens) along the cheating and clean rollouts, against a TF-IDF text
classifier scored on the same prefixes. Positions where the two disagree are where a probe could show it reads more than text.

TF-IDF: the truncated-CoT model of text_baselines.py (parity train split, cut at 10..90% of each CoT, label = final outcome),
refit here with the same settings. Both resampled rollouts are excluded from its training data (build_sets.py).

Run from the repo root after resample_step3.py run:
    /tmp/john-scratch/rs-venv/bin/python probe_tests/step3_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from mechtools.resample import Resampler
from mechtools.stats import wilson

import resample_step3 as rs3  # same prompt, ids, config and judge

SETS = Path("data/probe_tests/sets")
GRADER = Path("data/inspect/grader")
OUT = Path("data/probe_tests/resample")
FIG = Path("data/probe_tests/figures")
CUT_FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BLUE, ORANGE = "#2a78d6", "#eb6834"  # reference palette slots 1-2


def load_jsonl(path):
    return [json.loads(line) for line in path.open()]


#%% TF-IDF on truncated parity CoTs
pool = [r for r in load_jsonl(SETS / "natural_pool.jsonl") if not r["exclude"] and r["family"] == "parity"]
text = {}
for f in sorted({r["file"] for r in pool}):
    for line in (GRADER / f).open():
        rec = json.loads(line)
        text[rec["rollout_id"]] = rec["turns"][0]["reasoning"] or ""
assert not {rs3.IDS["cheating"], rs3.IDS["clean"]} & {r["rollout_id"] for r in pool}
train = [(text[r["rollout_id"]][: int(f * len(text[r["rollout_id"]]))], r["cheat"]) for r in pool if r["split"] == "train" for f in CUT_FRACS]
val = [(text[r["rollout_id"]][: int(f * len(text[r["rollout_id"]]))], r["cheat"]) for r in pool if r["split"] == "val" for f in CUT_FRACS]
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
Xtr = vec.fit_transform([x for x, _ in train])
Xva = vec.transform([x for x, _ in val])
fits = [(roc_auc_score([y for _, y in val], c.decision_function(Xva)), C, c) for C in [0.1, 1.0, 10.0]
        for c in [LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr, [y for _, y in train])]]
val_auc, C, clf = max(fits, key=lambda z: z[0])
print(f"truncated-CoT TF-IDF: C={C}, val AUC {val_auc:.3f} on {len(val)} truncated val CoTs")

#%% curves
tok = AutoTokenizer.from_pretrained(rs3.MODEL_ID)
rsm = rs3.resamplers(tok)
rows, summary = [], {}
for name, r in rsm.items():
    by_t = {}
    for x in r.rollouts:
        by_t.setdefault(x["t"], []).append(x.get("cheat"))
    reuse = {s["t"]: s for s in r.scores("cheat", method="reuse")}
    ts = sorted(by_t)
    probs = clf.predict_proba(vec.transform([r.strs[t] for t in ts]))[:, 1]
    for t, p_text in zip(ts, probs):
        v = [y for y in by_t[t] if y is not None]
        k = sum(v)
        lo, hi = wilson(k, len(v)) if v else (float("nan"), float("nan"))
        rows.append({"rollout": name, "t": t, "n": len(v), "k": k, "p_naive": k / len(v) if v else float("nan"), "ci_lo": lo, "ci_hi": hi,
                     "p_reuse": reuse[t]["p"] if t in reuse else float("nan"), "n_reuse": reuse[t]["n"] if t in reuse else 0,
                     "tfidf_prob": float(p_text), "prefix_tail": r.strs[t][-120:]})
(OUT / "step3_curves.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))

#%% disagreement on the dense grid (n >= 10)
print("\nDense grid (n >= 10 per position): naive P(cheat) with Wilson 95% CI vs TF-IDF")
for name in rsm:
    g = [x for x in rows if x["rollout"] == name and x["n"] >= 10]
    hi_text = [x for x in g if x["tfidf_prob"] > x["ci_hi"]]
    lo_text = [x for x in g if x["tfidf_prob"] < x["ci_lo"]]
    print(f"  {name}: {len(g)} positions, t {g[0]['t']}..{g[-1]['t']}; TF-IDF above the CI at {len(hi_text)}, below the CI at {len(lo_text)}")
    for x in g:
        flag = "TEXT>P" if x["tfidf_prob"] > x["ci_hi"] else "TEXT<P" if x["tfidf_prob"] < x["ci_lo"] else ""
        print(f"    t={x['t']:4d} P={x['k']:2d}/{x['n']:2d}={x['p_naive']:.2f} [{x['ci_lo']:.2f},{x['ci_hi']:.2f}]  tfidf {x['tfidf_prob']:.2f} {flag:6s} ...{x['prefix_tail'][-70:]!r}")

#%% clean rollout along its whole length (stride 1, reuse estimate)
c = [x for x in rows if x["rollout"] == "clean"]
k = 25
cheat1 = np.array([x["p_naive"] for x in c])
smooth = np.convolve(np.nan_to_num(cheat1), np.ones(k) / k, mode="same")
print("\nclean rollout, 25-position mean of naive P(cheat) vs TF-IDF at selected t")
for t in [0, 100, 186, 200, 250, 289, 300, 322, 350, 400, 500, 800, 1200, 1600]:
    if t < len(c):
        print(f"  t={t:4d} P~{smooth[t]:.2f}  reuse {c[t]['p_reuse']:.2f} (n_eff {c[t]['n_reuse']})  tfidf {c[t]['tfidf_prob']:.2f}")

#%% figure: one panel per rollout
fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.12,
                    subplot_titles=["Cheating rollout (Ethan's, densified over t = 100-400)", "Clean rollout (forks from the cheating one at t = 186)"])
for i, name in enumerate(["cheating", "clean"], start=1):
    g = [x for x in rows if x["rollout"] == name]
    ts = [x["t"] for x in g]
    dense = [x for x in g if x["n"] >= 10]
    sm = np.convolve(np.nan_to_num([x["p_naive"] for x in g]), np.ones(k) / k, mode="same")
    fig.add_trace(go.Scatter(x=ts, y=sm, mode="lines", line=dict(color=BLUE, width=2), name="P(cheat), resampled (25-position mean)",
                             legendgroup="p", showlegend=i == 1, hovertemplate="t=%{x}<br>P(cheat) %{y:.2f}<extra></extra>"), row=i, col=1)
    if dense:
        fig.add_trace(go.Scatter(x=[x["t"] for x in dense], y=[x["p_naive"] for x in dense], mode="markers",
                                 marker=dict(color=BLUE, size=8, line=dict(color="#fcfcfb", width=2)),
                                 error_y=dict(type="data", symmetric=False, array=[x["ci_hi"] - x["p_naive"] for x in dense],
                                              arrayminus=[x["p_naive"] - x["ci_lo"] for x in dense], color=BLUE, thickness=1, width=0),
                                 name="P(cheat) at n = 10, Wilson 95% CI", legendgroup="d", showlegend=i == 1,
                                 hovertemplate="t=%{x}<br>P(cheat) %{y:.2f}<extra></extra>"), row=i, col=1)
    fig.add_trace(go.Scatter(x=ts, y=[x["tfidf_prob"] for x in g], mode="lines", line=dict(color=ORANGE, width=2, dash="dash"),
                             name="TF-IDF text classifier on the prefix", legendgroup="t", showlegend=i == 1,
                             hovertemplate="t=%{x}<br>TF-IDF %{y:.2f}<extra></extra>"), row=i, col=1)
    fig.add_vline(x=186, line=dict(color="#898781", width=1, dash="dot"), row=i, col=1)
    fig.update_yaxes(range=[-0.03, 1.03], title_text="probability", gridcolor="#e6e5e0", row=i, col=1)
    fig.update_xaxes(title_text="reasoning tokens kept (t)", gridcolor="#e6e5e0", row=i, col=1)
fig.update_layout(template="plotly_white", paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb", height=760, width=1000,
                  font=dict(color="#3d3c39"), legend=dict(orientation="h", y=1.08, x=0),
                  title=dict(text="Does a text classifier track P(cheat | prefix)? Odd hack prompt, Qwen3.6-27B", y=0.98))
FIG.mkdir(parents=True, exist_ok=True)
fig.write_html(FIG / "step3_curves.html", include_plotlyjs="cdn")
print(f"\nwrote {OUT / 'step3_curves.jsonl'} and {FIG / 'step3_curves.html'}")
