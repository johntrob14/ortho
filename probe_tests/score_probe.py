"""Step 2: per-token scores of the grader_parity_cheat_vs_clean direction on the eval rollouts. GPU box (Qwen3.6-27B in bf16).

For every rollout: resid_post at every layer, projected on that layer's unit row of the vector, at every token of the served ids
(prompt, CoT, answer). Stored as float16 [n_layers, seq] with the CoT span (as utils.cot_span: after <think>, up to the last
</think>) and the norm of the CoT-mean residual per layer, so both a per-token probe and the vector's own statistic (cosine of
the CoT mean with the row) can be computed offline.

Order: Ethan's two resampled rollouts, then sign, pets_listed, held-out parity (log 2026-10-02T21-52-56, not in the vector's
training data), the system-prompt pool, then in-sample parity. Shards under data/probe_tests/probe_scores/; a rerun skips
rollout_ids already on disk.

    uv run python probe_tests/score_probe.py
"""
import json
import sys
import time
from pathlib import Path

import torch as t

from mechtools import *

from utils import load_vector

t.set_grad_enabled(False)
MODEL_ID = "Qwen/Qwen3.6-27B"
VECTOR = "grader_parity_cheat_vs_clean"
THINK_OPEN, THINK_CLOSE = 248068, 248069
SETS = Path("data/probe_tests/sets")
GRADER = Path("data/inspect/grader")
OUT = Path("data/probe_tests/probe_scores")
SHARD_SIZE = 200
RESAMPLED = ["2026-10-02T12-48-29-00-00_grader_M2iuL7zk7KphWhNa6ev3Wi/parity.odd.hack.none.36/1",
             "2026-10-02T13-12-46-00-00_grader_VwSvUNosNaLtivxYs4b7MT/parity.odd.hack.none.127/1"]
HELD_OUT_PARITY = "2026-10-02T21-52-56"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None  # for a smoke run


def load_jsonl(path):
    return [json.loads(line) for line in path.open()]


#%% work list
pool = [r for r in load_jsonl(SETS / "natural_pool.jsonl") if not r["exclude"]]
sysp = [r for r in load_jsonl(SETS / "sysprompt_pool.jsonl") if not r["exclude"]]
by_file = {}
for r in pool + sysp:
    by_file.setdefault(r["file"], []).append(r)
rank = {"resampled": 0, "sign": 1, "pets_listed": 2, "parity_heldout": 3, "parity_sysprompt": 4, "parity": 5}
work = [(0, rid.split("/")[0] + ".jsonl", rid, "resampled") for rid in RESAMPLED]
for r in pool + sysp:
    g = "parity_heldout" if r["group"] == "parity" and r["file"].startswith(HELD_OUT_PARITY) else r["group"]
    work.append((rank[g], r["file"], r["rollout_id"], g))
work.sort(key=lambda w: (w[0], w[2]))
OUT.mkdir(parents=True, exist_ok=True)
done = set()
for shard in sorted(OUT.glob("shard_*.pt")):
    done |= set(t.load(shard)["rollout_id"])
work = [w for w in work if w[2] not in done]
if LIMIT:
    work = work[:LIMIT]
print(f"{len(done)} rollouts already scored, {len(work)} to go")

ids_of = {}
for f in sorted({w[1] for w in work}):
    want = {w[2] for w in work if w[1] == f}
    for line in (GRADER / f).open():
        rec = json.loads(line)
        if rec["rollout_id"] in want:
            ids_of[rec["rollout_id"]] = rec["ids"]
assert len(ids_of) == len(work), (len(ids_of), len(work))

#%% model and vector
model = load_bridge(MODEL_ID, device_map="cuda")
n_layers = model.cfg.n_layers
v, layers = load_vector(Path("data/vectors/Qwen3.6-27B"), VECTOR)
assert v.shape[0] == n_layers and layers == list(range(n_layers)), (v.shape, layers)
units = (v / v.norm(dim=-1, keepdim=True)).to(model.device)  # [n_layers, d_model] float32

captured, span = {}, {}


def make_hook(layer):
    def hook(act, hook):
        a = act[0].float()
        captured[layer] = (a @ units[layer], a[span["start"]:span["end"]].mean(0).norm())  # projection per token, norm of the CoT-mean residual
        return act
    return hook


hooks = [(f"blocks.{layer}.hook_resid_post", make_hook(layer)) for layer in range(n_layers)]

#%% score
rows, shard_i, t0 = [], len(list(OUT.glob("shard_*.pt"))), time.time()


def flush():
    global rows, shard_i
    if rows:
        t.save({k: [r[k] for r in rows] for k in rows[0]}, OUT / f"shard_{shard_i:04d}.pt")
        print(f"  wrote shard {shard_i} ({len(rows)} rollouts)", flush=True)
        rows, shard_i = [], shard_i + 1


n_tok = 0
for k, (_, f, rid, group) in enumerate(work):
    ids = ids_of[rid]
    start, end = ids.index(THINK_OPEN) + 1, len(ids) - 1 - ids[::-1].index(THINK_CLOSE)
    captured.clear()
    span["start"], span["end"] = start, end
    with model.hooks(fwd_hooks=hooks):
        model(t.tensor([ids], device=model.device), return_type=None)
    proj = t.stack([captured[layer][0] for layer in range(n_layers)])  # [n_layers, seq]
    mean_norm = t.stack([captured[layer][1] for layer in range(n_layers)])
    rows.append({"rollout_id": rid, "group": group, "proj": proj.half().cpu(), "cot_start": start, "cot_end": end, "cot_mean_norm": mean_norm.cpu()})
    n_tok += len(ids)
    if (k + 1) % 50 == 0:
        el = time.time() - t0
        print(f"{k + 1}/{len(work)} rollouts, {n_tok / el:.0f} tok/s, eta {el / (k + 1) * (len(work) - k - 1) / 60:.0f} min", flush=True)
    if len(rows) >= SHARD_SIZE or (group == "resampled" and k == 1):
        flush()
flush()
print("done", flush=True)
