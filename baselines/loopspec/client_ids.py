"""Drive one running sglang_recurrent server (AR baseline or LoopSpec) over a prompt file, batch 1, greedy.

Same as client.py, with one difference that matters: the payload carries **input_ids**, not text.

Our own prompt files store the context already tokenised (`ids`), and that is what our engine decoded.
Handing the server the same token list makes the comparison literally the same prompt, instead of
decoding to a string and letting the other harness re-tokenise it — a round trip that can change the
prompt length by a token or two and so move the measured speed.

    python client_ids.py --prompts P.jsonl --out DIR --tag T --max-new N --server-log LOG
                         [--start S --end E --limit L --stride K] [--probe]

`--stride` exists because in the multi-subtask sets (BBH, MMLU-Pro) doc_id is ordered by subtask, so a
contiguous slice would sample one subject and its prompt lengths. Striding spreads the subset.

Per doc writes {doc_id, text, output_ids, completion_tokens, finish, wall}; afterwards joins the server's
per-request `generation_metrics` lines (the last N of the log == the N docs; requests are strictly
sequential) and writes gens_<tag>.jsonl + metrics_<tag>.json.
"""
import argparse, json, os, re, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=10**9)
ap.add_argument("--limit", type=int, default=0, help="keep at most this many docs after start/end + stride")
ap.add_argument("--stride", type=int, default=1, help="keep every K-th doc (spreads a subset over subtasks)")
ap.add_argument("--max-new", type=int, required=True); ap.add_argument("--server-log", required=True)
ap.add_argument("--url", default="http://127.0.0.1:30000"); ap.add_argument("--probe", action="store_true")
ap.add_argument("--deadline", type=float, default=0, help="epoch seconds; stop starting new docs after this")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
EOS_STR = "<|endoftext|>"
METRICS_RE = re.compile(r"generation_metrics (\{.*\})")


def post(payload, timeout=3600):
    req = urllib.request.Request(a.url + "/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def metric_lines():
    with open(a.server_log, errors="replace") as f:
        return [json.loads(m.group(1)) for m in (METRICS_RE.search(l) for l in f if "generation_metrics" in l) if m]


docs = [json.loads(l) for l in open(a.prompts)]
docs = [d for d in docs if a.start <= d["doc_id"] < a.end]
if a.stride > 1:
    docs = docs[::a.stride]
if a.limit:
    docs = docs[:a.limit]
assert docs, "no docs selected"
missing = [d["doc_id"] for d in docs if not d.get("ids")]
assert not missing, f"{len(missing)} docs have no `ids` (this client sends input_ids, not text): {missing[:5]}"
print(f"[client] {len(docs)} docs, prompt tokens {min(len(d['ids']) for d in docs)}-{max(len(d['ids']) for d in docs)}", flush=True)

summary = {}
if a.probe:
    time.sleep(2); n0 = len(metric_lines())
    pids = docs[0]["ids"]
    for _ in range(3):
        post({"input_ids": pids, "sampling_params": {"temperature": 0, "max_new_tokens": 256, "ignore_eos": True}})
    time.sleep(3); pm = metric_lines()[n0:]
    summary["probe"] = pm
    toks = sum(m["decode_tokens"] for m in pm); secs = sum(m["decode_seconds"] for m in pm)
    summary["probe_ms_per_token"] = 1000 * secs / max(toks, 1)
    if pm and "pipeline_cycles" in pm[0]:
        summary["probe_ms_per_cycle"] = 1000 * secs / max(sum(m["pipeline_cycles"] for m in pm), 1)
    print(f"[client] probe: {json.dumps({k: v for k, v in summary.items() if k != 'probe'})}", flush=True)

recs, t_all = [], time.time()
for i, d in enumerate(docs):
    if a.deadline and time.time() > a.deadline:
        print(f"[client] deadline hit after {i} docs", flush=True); break
    stop = list(d["until"]) + ([EOS_STR] if EOS_STR not in d["until"] else [])
    t0 = time.time()
    r = post({"input_ids": d["ids"], "sampling_params": {"temperature": 0, "max_new_tokens": a.max_new, "stop": stop}})
    wall = time.time() - t0
    mi = r.get("meta_info", {})
    fr = mi.get("finish_reason")
    recs.append(dict(doc_id=d["doc_id"], text=r["text"], output_ids=r.get("output_ids"), completion_tokens=mi.get("completion_tokens"),
                     prompt_tokens=mi.get("prompt_tokens"), finish=(fr or {}).get("type") if isinstance(fr, dict) else fr,
                     matched=(fr or {}).get("matched") if isinstance(fr, dict) else None, wall=wall))
    if i % 25 == 0 or i == len(docs) - 1:
        print(f"[client] {i + 1}/{len(docs)} doc {d['doc_id']} toks {mi.get('completion_tokens')} wall {wall:.2f}s "
              f"elapsed {time.time() - t_all:.0f}s", flush=True)

time.sleep(3)
ms = metric_lines()
ms = ms[len(ms) - len(recs):] if recs else []
assert len(ms) == len(recs), (len(ms), len(recs))
bad = 0
for r, m in zip(recs, ms):
    r["metrics"] = m
    if r["completion_tokens"] is not None and m["decode_tokens"] not in (r["completion_tokens"] - 1, 0):
        bad += 1
summary.update(n=len(recs), start=a.start, end=a.end, limit=a.limit, stride=a.stride, max_new=a.max_new,
               metric_token_mismatch=bad, prompts=a.prompts,
               decode_tokens=sum(m["decode_tokens"] for m in ms), decode_seconds=sum(m["decode_seconds"] for m in ms),
               wall=sum(r["wall"] for r in recs), completion_tokens=sum(r["completion_tokens"] or 0 for r in recs))
summary["decode_tok_s"] = summary["decode_tokens"] / max(summary["decode_seconds"], 1e-9)
summary["e2e_tok_s"] = summary["completion_tokens"] / max(summary["wall"], 1e-9)
for k in ("pipeline_cycles", "pre_rows", "first_accept", "first_reject", "second_skip", "second_accept", "second_reject"):
    if ms and k in ms[0]:
        summary[k] = sum(m[k] for m in ms)
with open(f"{a.out}/gens_{a.tag}.jsonl", "w") as f:
    for r in recs: f.write(json.dumps(r) + "\n")
json.dump(summary, open(f"{a.out}/metrics_{a.tag}.json", "w"), indent=1)
print(f"[client] DONE {json.dumps({k: v for k, v in summary.items() if k != 'probe'})}", flush=True)
