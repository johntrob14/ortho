"""Step 1c: text baselines on CoTs cut before the first committed answer, and the top TF-IDF features.

Full CoTs state the decision, so text classifiers separate them almost perfectly (text_baselines.py). The test set a probe
can win on is the CoT before the model commits. Cut point: the first match of COMMIT (a first-person statement of the
answer, or an "Output:"/"Final answer" line). Label: the rollout's final outcome (not a resampled P(cheat)).

Run from the repo root (after build_sets.py):
    uv run --no-project --with scikit-learn --with numpy python probe_tests/precommit.py
"""
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SETS = Path("data/probe_tests/sets")
OUT = Path("data/probe_tests/text_baselines")
GRADER = Path("data/inspect/grader")
FAMILIES = ["parity", "sign", "pets_listed"]
COMMIT = re.compile(
    r"(I'll|I will|I'm going to|I should|Let's|let me)\s+(just\s+)?(output|go with|stick with|pick|choose|answer|give|write|respond with|select)\b"
    r"|\b(final answer|output)\s*[:=]",
    re.I,
)
MIN_CHARS = 200  # rollouts that commit within the first 200 characters are dropped (no deliberation to read)


def load_jsonl(path):
    return [json.loads(line) for line in path.open()]


def auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < sum(y) < len(y) else float("nan")


pool = [r for r in load_jsonl(SETS / "natural_pool.jsonl") if not r["exclude"]]
text = {}
for path in sorted({r["file"] for r in pool}):
    for line in (GRADER / path).open():
        rec = json.loads(line)
        text[rec["rollout_id"]] = rec["turns"][0]["reasoning"] or ""
rows = []
for r in pool:
    m = COMMIT.search(text[r["rollout_id"]])
    cut = m.start() if m else None
    if cut is not None and cut >= MIN_CHARS:
        rows.append(dict(r, pre=text[r["rollout_id"]][:cut], cut_frac=cut / len(text[r["rollout_id"]])))
print(f"usable {len(pool)}, with a commitment at >= {MIN_CHARS} chars: {len(rows)}")
for f in FAMILIES:
    fr = [r for r in rows if r["family"] == f]
    c = [r["cut_frac"] for r in fr if r["cheat"]]
    k = [r["cut_frac"] for r in fr if not r["cheat"]]
    print(f"  {f:11s} n={len(fr)} cheat={len(c)}  median cut fraction cheat {np.median(c):.2f} clean {np.median(k):.2f}")

by = {(f, s): [r for r in rows if r["family"] == f and r["split"] == s] for f in FAMILIES for s in ["train", "val", "test"]}
results = []
print("\npre-commitment TF-IDF (trained on pre-commitment text), AUC [cheat/n]; length-only AUC alongside")
for tr in FAMILIES:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform([r["pre"] for r in by[tr, "train"]])
    y = [r["cheat"] for r in by[tr, "train"]]
    yv = [r["cheat"] for r in by[tr, "val"]]
    best = max(((auc(yv, c.decision_function(vec.transform([r["pre"] for r in by[tr, "val"]]))), C, c) for C in [0.1, 1.0, 10.0]
                for c in [LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(X, y)]), key=lambda z: z[0])
    clf = best[2]
    lclf = LogisticRegression(class_weight="balanced").fit(np.log([[len(r["pre"])] for r in by[tr, "train"]]), y)
    for te in FAMILIES:
        test = by[te, "test"]
        yt = [r["cheat"] for r in test]
        a = auc(yt, clf.decision_function(vec.transform([r["pre"] for r in test])))
        al = auc(yt, lclf.decision_function(np.log([[len(r["pre"])] for r in test])))
        results.append({"train": tr, "test": te, "C": best[1], "n": len(test), "n_cheat": sum(yt), "auc_tfidf": a, "auc_len": al})
        print(f"  {tr:11s} -> {te:11s} tfidf {a:.3f}  length {al:.3f}  [{sum(yt)}/{len(test)}]")
    if tr == "sign":
        names = np.array(vec.get_feature_names_out())
        order = np.argsort(clf.coef_[0])
        print("  sign pre-commit top cheat features:", list(names[order[-20:]][::-1]))
        print("  sign pre-commit top clean features:", list(names[order[:20]]))
(OUT / "precommit_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
(SETS / "precommit_cuts.jsonl").write_text("".join(json.dumps({"rollout_id": r["rollout_id"], "cut_chars": len(r["pre"]), "cheat": r["cheat"], "split": r["split"]}) + "\n" for r in rows))

# full-CoT top features for comparison
for fam in ["sign", "parity"]:
    tr = [r for r in pool if r["family"] == fam and r["split"] == "train"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(vec.fit_transform([text[r["rollout_id"]] for r in tr]), [r["cheat"] for r in tr])
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    print(f"\nfull-CoT {fam} top cheat features:", list(names[order[-20:]][::-1]))
    print(f"full-CoT {fam} top clean features:", list(names[order[:20]]))
