"""Prior pressure, free generation: sample the grader prompts under each system prompt (prior.SYSTEM) with vLLM offline.

Sampling as the rhenvs grader runs: temperature 1, top_p 0.95, top_k 20, max_tokens 4096, reasoning on. Prompt token ids are
built as sys block + the user turn's ids, so they match what the server would render. One seed per (condition, task).
Writes data/probe_tests/prior/gen.jsonl, one row per rollout (appended per condition and task, skipped if present).

    /workspace/venvs/vllm/bin/python probe_tests/prior_generate.py [N_PER_TASK]
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
import prior

MODEL_ID = "Qwen/Qwen3.6-27B"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
OUT = Path("data/probe_tests/prior/gen.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    have = set()
    if OUT.exists():
        for line in OUT.open():
            r = json.loads(line)
            have.add((r["cond"], r["family"], r["side"]))

    jobs = []
    for ci, cond in enumerate(prior.SYSTEM):
        for ti, ((family, side), text) in enumerate(prior.TASKS.items()):
            if (cond, family, side) in have:
                continue
            uprompt = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            uids = tok(uprompt, add_special_tokens=False)["input_ids"]
            jobs.append((cond, family, side, prior.sys_block_ids(tok, cond), uids, 1000 * ci + ti))
    print(f"{len(jobs)} (condition, task) cells to sample, {N} each")

    if jobs:
        llm = LLM(MODEL_ID, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.92, seed=0, enforce_eager=True, trust_remote_code=True, max_num_seqs=1024)  # as scripts/serve.sh; a larger batch for throughput
        prompts = [{"prompt_token_ids": s + u} for _, _, _, s, u, _ in jobs]
        params = [SamplingParams(n=N, temperature=1.0, top_p=0.95, top_k=20, max_tokens=4096, seed=seed) for *_, seed in jobs]
        outs = llm.generate(prompts, params)
        with OUT.open("a") as f:
            for (cond, family, side, sids, uids, seed), out in zip(jobs, outs):
                for i, c in enumerate(out.outputs):
                    comp = list(c.token_ids)
                    closed = prior.THINK_CLOSE in comp
                    k = comp.index(prior.THINK_CLOSE) if closed else len(comp)
                    answer = tok.decode(comp[k + 1:], skip_special_tokens=True) if closed else ""
                    reasoning = tok.decode(comp[:k], skip_special_tokens=True)
                    row = {"key": f"{cond}/{family}.{side}/{i}", "cond": cond, "family": family, "side": side, "i": i, "seed": seed,
                           "finish": c.finish_reason, "closed": closed, "n_close": comp.count(prior.THINK_CLOSE),
                           "answer": answer.strip(), "cheat": prior.judge(family, side, answer) if closed and c.finish_reason == "stop" else None,
                           "reasoning": reasoning, "user_ids": uids + comp, "n_sys": len(sids)}
                    f.write(json.dumps(row) + "\n")
        print("done")


if __name__ == "__main__":  # vLLM spawns worker processes, which re-import this module
    main()
