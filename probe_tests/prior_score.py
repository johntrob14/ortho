"""Prior pressure, probe scores: grader_parity_cheat_vs_clean per token on the same tokens under different system prompts.

Items, each a rollout's user-turn ids scored under a list of system prompts (prior.SYSTEM keys):
  natural   held-out parity (log 2026-10-02T21-52-56, not in the vector's training data) and a seeded sign sample of 150 cheat /
            150 clean, all no-system rollouts, token-forced under none, generic, encourage, discourage
  gen       the rollouts of prior_generate.py, scored under their own system prompt and under none
Per (item, condition): projection of resid_post on each layer's unit row at every user-turn token (positions aligned across
conditions; the sys block itself is not stored), float16 [n_layers, len(user ids)], plus the CoT span and the norm of the
CoT-mean residual per layer. Shards in data/probe_tests/prior/scores_<part>/; reruns skip what is on disk.

    PYTHONPATH=. uv run python probe_tests/prior_score.py natural|gen
"""
import json
import random
import sys
import time
from pathlib import Path

import torch as t

from mechtools import *

from utils import load_vector

sys.path.insert(0, str(Path(__file__).parent))
import prior

t.set_grad_enabled(False)
MODEL_ID = "Qwen/Qwen3.6-27B"
PART = sys.argv[1]
assert PART in ("natural", "gen"), PART
SETS = Path("data/probe_tests/sets")
GRADER = Path("data/inspect/grader")
OUT = Path(f"data/probe_tests/prior/scores_{PART}")
SHARD = 200
HELD_OUT_PARITY = "2026-10-02T21-52-56"

model = load_bridge(MODEL_ID, device_map="cuda")
tok = model.tokenizer
n_layers = model.cfg.n_layers
v, _ = load_vector(Path("data/vectors/Qwen3.6-27B"), "grader_parity_cheat_vs_clean")
units = (v / v.norm(dim=-1, keepdim=True)).to(model.device)

#%% items
items = []  # (key, family, side, cheat, user ids, conditions)
if PART == "natural":
    pool = [json.loads(l) for l in (SETS / "natural_pool.jsonl").open()]
    usable = [r for r in pool if not r["exclude"]]
    parity = [r for r in usable if r["family"] == "parity" and r["file"].startswith(HELD_OUT_PARITY)]
    sign = [r for r in usable if r["family"] == "sign"]
    rng = random.Random(0)
    sign = rng.sample([r for r in sign if r["cheat"]], 150) + rng.sample([r for r in sign if not r["cheat"]], 150)
    want = {r["rollout_id"]: r for r in parity + sign}
    for f in sorted({r["file"] for r in want.values()}):
        for line in (GRADER / f).open():
            rec = json.loads(line)
            if rec["rollout_id"] in want:
                m = want[rec["rollout_id"]]
                items.append((rec["rollout_id"], m["family"], m["side"], m["cheat"], prior.user_ids(tok, rec["ids"]), list(prior.SYSTEM)))
else:
    # Scoring every generated rollout would take hours at this throughput, so per (condition, task) a seeded subset:
    # up to GEN_PER_CLASS cheats and GEN_PER_CLASS clean rollouts (cheat rates still use every rollout).
    GEN_PER_CLASS = 64
    cells = {}
    for line in Path("data/probe_tests/prior/gen.jsonl").open():
        r = json.loads(line)
        if r["cheat"] is not None:
            cells.setdefault((r["cond"], r["family"], r["side"], r["cheat"]), []).append(r)
    rng = random.Random(0)
    for (cond, family, side, cheat), rs in sorted(cells.items()):
        for r in rng.sample(rs, min(GEN_PER_CLASS, len(rs))):
            conds = ["none"] if cond == "none" else [cond, "none"]
            items.append((r["key"], family, side, cheat, r["user_ids"], conds))
        print(f"  {cond:10s} {family}.{side:8s} cheat={cheat}: {len(rs)} rollouts, scoring {min(GEN_PER_CLASS, len(rs))}")
OUT.mkdir(parents=True, exist_ok=True)
done = set()
for shard in sorted(OUT.glob("shard_*.pt")):
    d = t.load(shard)
    done |= set(zip(d["key"], d["cond"]))
work = [(it, c) for it in items for c in it[5] if (it[0], c) not in done]
print(f"{len(items)} items, {len(done)} (item, condition) pairs on disk, {len(work)} to go", flush=True)

#%% hooks
captured, span = {}, {}


def make_hook(layer):
    def hook(act, hook):
        a = act[0, span["off"]:].float()  # user-turn positions only
        captured[layer] = (a @ units[layer], a[span["start"]:span["end"]].mean(0).norm())
        return act
    return hook


hooks = [(f"blocks.{layer}.hook_resid_post", make_hook(layer)) for layer in range(n_layers)]
sys_ids = {c: prior.sys_block_ids(tok, c) for c in prior.SYSTEM}
rows, shard_i, t0, n_tok = [], len(list(OUT.glob("shard_*.pt"))), time.time(), 0


def flush():
    global rows, shard_i
    if rows:
        t.save({k: [r[k] for r in rows] for k in rows[0]}, OUT / f"shard_{shard_i:04d}.pt")
        rows, shard_i = [], shard_i + 1


for k, ((key, family, side, cheat, uids, _), cond) in enumerate(work):
    ids = sys_ids[cond] + uids
    start, end = prior.cot_span(uids)
    span.update(off=len(sys_ids[cond]), start=start, end=end)
    captured.clear()
    with model.hooks(fwd_hooks=hooks):
        model(t.tensor([ids], device=model.device), return_type=None)
    rows.append({"key": key, "cond": cond, "family": family, "side": side, "cheat": cheat,
                 "proj": t.stack([captured[L][0] for L in range(n_layers)]).half().cpu(),
                 "cot_start": start, "cot_end": end, "cot_mean_norm": t.stack([captured[L][1] for L in range(n_layers)]).cpu()})
    n_tok += len(ids)
    if (k + 1) % 100 == 0:
        el = time.time() - t0
        print(f"{k + 1}/{len(work)}, {n_tok / el:.0f} tok/s, eta {el / (k + 1) * (len(work) - k - 1) / 60:.0f} min", flush=True)
    if len(rows) >= SHARD:
        flush()
flush()
print("done", flush=True)
