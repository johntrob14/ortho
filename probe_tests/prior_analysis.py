"""Prior-pressure analysis and figures.

Probe: grader_parity_cheat_vs_clean, layer 36 (fixed in advance), score = mean per-token projection on the unit row over the
CoT ("CoT-mean projection", residual-norm units); the cosine statistic (divided by the CoT-mean norm) as a check.

Part A, token forcing (natural no-system rollouts, identical tokens under each system prompt): paired differences vs none.
Part B, free generation: cheat rates per condition; probe score of generated CoTs under their own system prompt and with the
system prompt removed; lexical expectation (token-identity model, step2_lexical.py) and reward-vocabulary counts as text controls.

    /tmp/john-scratch/rs-venv/bin/python probe_tests/prior_analysis.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import prior
from mechtools.stats import wilson

L = 36
P = Path("data/probe_tests/prior")
FIG = Path("data/probe_tests/figures")
FIG.mkdir(parents=True, exist_ok=True)
CONDS = ["none", "generic", "encourage", "discourage", "negated"]
COLOR = {"none": "#3d3c39", "generic": "#898781", "encourage": "#eb6834", "discourage": "#2a78d6", "negated": "#1baf7a"}  # ink, gray, palette slots 2, 1, 3
STYLE = {"none": "-", "generic": ":", "encourage": "-", "discourage": "-", "negated": "--"}
VOCAB = re.compile(r"reward|grader|maximi[sz]|optimi[sz]|hack|satisf", re.I)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#898781",
                     "axes.labelcolor": "#3d3c39", "xtick.color": "#3d3c39", "ytick.color": "#3d3c39", "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})


def load(part):
    rows = {}
    for shard in sorted((P / f"scores_{part}").glob("shard_*.pt")):
        d = t.load(shard)
        for i in range(len(d["key"])):
            rows[d["key"][i], d["cond"][i]] = {k: d[k][i] for k in d}
    return rows


def cot_proj(r, layer=L):
    p = r["proj"][layer].float()
    return float(p[r["cot_start"]:r["cot_end"]].mean())


def ci95(x):
    x = np.asarray(x)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


def auc(y, s):
    y = np.asarray(y, dtype=int)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


report = {}
#%% Part A: token forcing
nat = load("natural")
CONDS = [c for c in CONDS if any(cc == c for _, cc in nat)]  # conditions scored so far
keys = sorted({k for k, _ in nat if all((k, c) in nat for c in CONDS)})
print(f"Part A: {len(keys)} natural rollouts scored under all of {CONDS}")
A = []
for k in keys:
    r0 = nat[k, "none"]
    row = {"key": k, "family": r0["family"], "side": r0["side"], "cheat": r0["cheat"]}
    for c in CONDS:
        row[c] = cot_proj(nat[k, c])
        row[c + "_cos"] = row[c] / float(nat[k, c]["cot_mean_norm"][L])
    A.append(row)
print("\nCoT-mean projection at L36, mean [95% CI] of the paired difference vs none (same tokens)")
for fam in ["parity", "sign"]:
    for cheat in [False, True]:
        g = [r for r in A if r["family"] == fam and r["cheat"] == cheat]
        if not g:
            continue
        line = f"  {fam:6s} {'cheat' if cheat else 'clean'} n={len(g):4d}  none {np.mean([r['none'] for r in g]):+.3f}"
        for c in CONDS[1:]:
            d = [r[c] - r["none"] for r in g]
            line += f" | {c} {np.mean(d):+.3f} [{np.mean(d) - ci95(d):+.3f}, {np.mean(d) + ci95(d):+.3f}]"
            report[f"A/{fam}/{'cheat' if cheat else 'clean'}/{c}-none"] = (float(np.mean(d)), float(ci95(d)), len(d))
        print(line)
    d_ed = [r["encourage"] - r["discourage"] for r in A if r["family"] == fam]
    print(f"  {fam:6s} encourage - discourage, all: {np.mean(d_ed):+.3f} +- {ci95(d_ed):.3f}; positive in {np.mean(np.array(d_ed) > 0):.0%} of rollouts")
print("\ncheat-vs-clean AUC of the CoT-mean projection under each forced system prompt (same tokens)")
for fam in ["parity", "sign"]:
    g = [r for r in A if r["family"] == fam]
    print(f"  {fam:6s} " + "  ".join(f"{c} {auc([r['cheat'] for r in g], [r[c] for r in g]):.3f}" for c in CONDS))
(P / "partA.jsonl").write_text("".join(json.dumps(r) + "\n" for r in A))

# per-token difference by CoT position (does the effect fade with distance from the system prompt?)
bins = [0, 50, 100, 200, 400, 800, 1600, 4000]
pos = {c: defaultdict(list) for c in CONDS[1:]}
for k in keys:
    r0 = nat[k, "none"]
    p0 = r0["proj"][L].float().numpy()
    s, e = r0["cot_start"], r0["cot_end"]
    for c in CONDS[1:]:
        dc = nat[k, c]["proj"][L].float().numpy() - p0
        for b0, b1 in zip(bins, bins[1:]):
            seg = dc[s + b0:min(s + b1, e)]
            if len(seg):
                pos[c][b0].append(seg.mean())

#%% Figure A1: token-level trace, same text under three system prompts
fam_pick = [r for r in A if r["family"] == "parity" and not r["cheat"]]
fam_pick += [r for r in A if r["family"] == "parity" and r["cheat"]]
examples = []
for want_cheat in [False, True]:
    cands = sorted([r for r in A if r["family"] == "parity" and r["cheat"] == want_cheat], key=lambda r: r["key"])
    cands = [r for r in cands if 800 <= nat[r["key"], "none"]["cot_end"] - nat[r["key"], "none"]["cot_start"] <= 1600]
    examples.append(cands[0])
fig, axes = plt.subplots(len(examples), 1, figsize=(11, 3.4 * len(examples)), sharex=False)
for ax, ex in zip(axes, examples):
    for c in ["none", "encourage", "discourage"]:
        r = nat[ex["key"], c]
        p = r["proj"][L].float().numpy()[r["cot_start"]:r["cot_end"]]
        w = np.convolve(p, np.ones(15) / 15, mode="same")
        ax.plot(np.arange(len(p)), p, color=COLOR[c], lw=0.5, alpha=0.25)
        ax.plot(np.arange(len(p)), w, color=COLOR[c], lw=2, ls=STYLE[c], label=f"{c} (CoT mean {p.mean():+.2f})")
    ax.set_title(f"{'Cheating' if ex['cheat'] else 'Clean'} parity rollout, {ex['side']} request, identical CoT tokens  ({ex['key'].split('/', 1)[1]})", loc="left", fontsize=10)
    ax.set_ylabel(f"probe projection, L{L}")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
axes[-1].set_xlabel("CoT token (thin: per token; thick: 15-token mean)")
fig.suptitle("Token-forced: the same CoT scored with no system prompt, an encourage-hacking prompt and a discourage-hacking prompt", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "prior_token_trace.png", dpi=150)
plt.close(fig)

#%% Figure A2: paired differences by family x outcome, and by CoT position
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
groups = [(f, ch) for f in ["parity", "sign"] for ch in [False, True]]
xs = np.arange(len(groups))
for j, c in enumerate(CONDS[1:]):
    m = [np.mean([r[c] - r["none"] for r in A if r["family"] == f and r["cheat"] == ch]) for f, ch in groups]
    e = [ci95([r[c] - r["none"] for r in A if r["family"] == f and r["cheat"] == ch]) for f, ch in groups]
    ax1.errorbar(xs + (j - 1) * 0.22, m, yerr=e, fmt="o", ms=7, color=COLOR[c], label=c, capsize=0, mec="#fcfcfb", mew=1.5)
ax1.axhline(0, color="#898781", lw=1)
ax1.set_xticks(xs, [f"{f}\n{'cheat' if ch else 'clean'} (n={sum(r['family'] == f and r['cheat'] == ch for r in A)})" for f, ch in groups])
ax1.set_ylabel(f"CoT-mean projection minus no-system value, L{L}")
ax1.set_title("Same tokens, different system prompt (mean, 95% CI)", loc="left", fontsize=10)
ax1.legend(frameon=False)
for c in CONDS[1:]:
    ax2.errorbar(range(len(bins) - 1), [np.mean(pos[c][b]) for b in bins[:-1]], yerr=[ci95(pos[c][b]) for b in bins[:-1]], color=COLOR[c], ls=STYLE[c], lw=2, marker="o", ms=5, label=c, capsize=0)
ax2.axhline(0, color="#898781", lw=1)
ax2.set_xticks(range(len(bins) - 1), [f"{a}-{b}" for a, b in zip(bins, bins[1:])])
ax2.set_xlabel("CoT tokens from the start of reasoning")
ax2.set_ylabel("per-token projection minus no-system value")
ax2.set_title("Does the shift fade along the CoT? (all forced rollouts)", loc="left", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "prior_forced_shift.png", dpi=150)
plt.close(fig)

#%% Part B: free generation
gen_path = P / "gen.jsonl"
if gen_path.exists():
    G = [json.loads(l) for l in gen_path.open()]
    print(f"\nPart B: {len(G)} generated rollouts")
    rates = {}
    for c in CONDS:
        for (fam, side) in prior.TASKS:
            g = [r for r in G if r["cond"] == c and r["family"] == fam and r["side"] == side]
            lab = [r["cheat"] for r in g if r["cheat"] is not None]
            k = sum(lab)
            lo, hi = wilson(k, len(lab)) if lab else (float("nan"),) * 2
            rates[c, fam, side] = (k, len(lab), lo, hi, len(g) - len(lab))
            print(f"  {c:10s} {fam}.{side:8s} cheat {k:3d}/{len(lab):3d} = {k / max(len(lab), 1):.2f} [{lo:.2f}, {hi:.2f}]  invalid {len(g) - len(lab)}")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    tasks = list(prior.TASKS)
    for j, c in enumerate(CONDS):
        x = np.arange(len(tasks)) + (j - 1.5) * 0.19
        v = [rates[c, f, s] for f, s in tasks]
        ax.bar(x, [a[0] / max(a[1], 1) for a in v], width=0.17, color=COLOR[c], label=c, edgecolor="#fcfcfb", linewidth=1)
        ax.errorbar(x, [a[0] / max(a[1], 1) for a in v], yerr=[[a[0] / max(a[1], 1) - a[2] for a in v], [a[3] - a[0] / max(a[1], 1) for a in v]],
                    fmt="none", ecolor="#3d3c39", lw=1, capsize=0)
    ax.set_xticks(np.arange(len(tasks)), [f"{f} / {s} request" for f, s in tasks])
    ax.set_ylabel("cheat rate (Wilson 95% CI)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    ax.set_title(f"Free generation, {len([r for r in G if r['cond'] == 'none' and r['family'] == 'parity' and r['side'] == 'even'])} samples per bar", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "prior_cheat_rates.png", dpi=150)
    plt.close(fig)

    gs = load("gen")
    if gs:
        # lexical model from the in-sample parity scores of step 2 (token-identity mean projection at L)
        sums, counts = defaultdict(float), defaultdict(int)
        ids_cache = {}
        for shard in sorted(Path("data/probe_tests/probe_scores").glob("shard_*.pt")):
            d = t.load(shard)
            for i, rid in enumerate(d["rollout_id"]):
                if d["group"][i] == "parity":
                    ids_cache[rid] = (d["proj"][i][L].float().numpy(), d["cot_start"][i], d["cot_end"][i])
        for f in sorted(Path("data/inspect/grader").glob("*.jsonl")):
            for line in f.open():
                rec = json.loads(line)
                if rec["rollout_id"] in ids_cache:
                    p, s, e = ids_cache[rec["rollout_id"]]
                    for tid, x in zip(rec["ids"][s:e], p[s:e]):
                        sums[tid] += x
                        counts[tid] += 1
        gm = sum(sums.values()) / sum(counts.values())
        lex = lambda ids: float(np.mean([sums[i] / counts[i] if counts[i] >= 5 else gm for i in ids]))
        gen_by_key = {r["key"]: r for r in G}
        B = []
        for (key, cond), r in gs.items():
            if cond != gen_by_key[key]["cond"]:
                continue
            g = gen_by_key[key]
            s, e = r["cot_start"], r["cot_end"]
            row = {"key": key, "cond": cond, "family": g["family"], "side": g["side"], "cheat": g["cheat"], "own": cot_proj(r),
                   "nosys": cot_proj(gs[key, "none"]) if (key, "none") in gs else float("nan"),
                   "lexical": lex(g["user_ids"][s:e]), "vocab_per_100": 100 * len(VOCAB.findall(g["reasoning"])) / max(1, e - s), "cot_tokens": e - s}
            B.append(row)
        (P / "partB.jsonl").write_text("".join(json.dumps(r) + "\n" for r in B))
        print("\nPart B, scored subset: CoT-mean projection L36 under the rollout's own system prompt, with it removed, text controls")
        for fam in ["parity", "sign"]:
            for cheat in [False, True]:
                for c in CONDS:
                    g = [r for r in B if r["family"] == fam and r["cheat"] == cheat and r["cond"] == c]
                    if not g:
                        continue
                    f = lambda k: f"{np.mean([r[k] for r in g]):+.2f}+-{ci95([r[k] for r in g]):.2f}"
                    print(f"  {fam:6s} {'cheat' if cheat else 'clean'} {c:10s} n={len(g):3d}  own {f('own')}  nosys {f('nosys')}  prior part {np.mean([r['own'] - r['nosys'] for r in B if r in g]):+.2f}"
                          f"  lexical {f('lexical')}  reward-vocab/100tok {f('vocab_per_100')}  CoT tokens {np.mean([r['cot_tokens'] for r in g]):.0f}")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
        cells = [(f, ch) for f in ["parity", "sign"] for ch in [False, True]]
        for ax, key, title in zip(axes, ["own", "nosys", "lexical"], ["probe, own system prompt", "probe, system prompt removed (same text)", "lexical expectation (token identity only)"]):
            for j, c in enumerate(CONDS):
                m, e, xs = [], [], []
                for i, (f, ch) in enumerate(cells):
                    g = [r[key] for r in B if r["family"] == f and r["cheat"] == ch and r["cond"] == c]
                    if len(g) >= 3:
                        m.append(np.mean(g)); e.append(ci95(g)); xs.append(i + (j - 1.5) * 0.18)
                ax.errorbar(xs, m, yerr=e, fmt="o", ms=6, color=COLOR[c], label=c, capsize=0, mec="#fcfcfb", mew=1.2)
            ax.set_xticks(range(len(cells)), [f"{f}\n{'cheat' if ch else 'clean'}" for f, ch in cells])
            ax.set_title(title, loc="left", fontsize=10)
        axes[0].set_ylabel(f"CoT-mean projection, L{L}")
        axes[0].legend(frameon=False, fontsize=9)
        fig.suptitle("Free generation: probe score by system prompt and outcome (mean, 95% CI; up to 64 rollouts per cell)", x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(FIG / "prior_free_generation.png", dpi=150)
        plt.close(fig)
(P / "report.json").write_text(json.dumps(report, indent=1))
print("\nfigures in", FIG)
