"""client_late.py -- drive a running sglang_late server over a prompt file (batch 1, sequential requests) and write records in late2's
run_late.py result format (so late2/score_tasks.py and tests/compare.py read them).

  python tests/client_late.py --url http://127.0.0.1:30000 --prompts <gsm8k_v2_prompts.jsonl> --docs 0:100 --max-new 1024 \
      --key sgl:l2:lkv:am:latt:4 --stats /tmp/late_stats.jsonl --model /tmp/sgl_v2 --out results/<tag>.json

Payload: the doc's prompt token ids (late2 data, same as late.py), temperature 0, max_new_tokens, stop = the doc's lm-eval `until`
strings with skip_special_tokens=False (late.py's has_stop decodes with special tokens), stop_token_ids = EOS {0, 2}, rid = doc<id>.
Per doc: gen (EOS removed), tokens_eff (first prefix whose decode contains a stop string; late.py's eff_len), text, wall time, and from
the worker's step log (LATE_STATS): decode seconds = last decode step end - prefill end, steps, per-step commits, GPU/host split.
"""
import argparse, json, os, statistics, time, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:30000"); ap.add_argument("--prompts", required=True); ap.add_argument("--docs", default="0:100")
ap.add_argument("--max-new", type=int, default=1024); ap.add_argument("--key", required=True); ap.add_argument("--stats", default="")
ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--eos", default="0,2")
a = ap.parse_args()
from tokenizers import Tokenizer
tokzr = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
EOS = [int(x) for x in a.eos.split(",")]
def has_stop(ids, until): return any(s in tokzr.decode(ids[-64:], skip_special_tokens=False) for s in until)
def eff_len(gen, until):
    if not until: return len(gen)
    for k in range(1, len(gen) + 1):
        if has_stop(gen[:k], until): return k
    return len(gen)


def post(payload, timeout=3600):
    req = urllib.request.Request(a.url + "/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stats_lines():
    if not a.stats or not os.path.exists(a.stats): return []
    out = []
    for l in open(a.stats):
        try: out.append(json.loads(l))
        except Exception: pass
    return out


docs = [json.loads(l) for l in open(a.prompts)]
d0, d1 = map(int, a.docs.split(":")); docs = [d for d in docs if d0 <= d["doc_id"] < d1]
tag = f"{int(time.time())}"
for i in range(a.warmup):
    post({"input_ids": docs[0]["ids"], "sampling_params": {"temperature": 0, "max_new_tokens": 64}, "rid": f"warm{i}_{tag}"})
recs = []; t_all = time.time()
for i, d in enumerate(docs):
    until = list(d.get("until") or [])
    sp = {"temperature": 0, "max_new_tokens": a.max_new, "stop": until, "stop_token_ids": EOS, "skip_special_tokens": False}
    rid = f"doc{d['doc_id']}_{tag}"
    t0 = time.perf_counter(); r = post({"input_ids": d["ids"], "sampling_params": sp, "rid": rid}); wall = time.perf_counter() - t0
    mi = r.get("meta_info", {}); raw = list(r.get("output_ids") or [])
    gen = [t for t in raw if t not in EOS]
    te = min(eff_len(gen, until), a.max_new)
    fr = mi.get("finish_reason"); frt = fr.get("type") if isinstance(fr, dict) else fr
    recs.append(dict(doc_id=d["doc_id"], rid=rid, prompt_tokens=len(d["ids"]), tokens=len(gen), tokens_eff=te, gen=gen, gen_raw=raw, wall=wall,
                     text=tokzr.decode(gen[:te], skip_special_tokens=False), finish_reason=fr,
                     finish="eos" if len(raw) > len(gen) else ("stop" if frt == "stop" else "length"),
                     completion_tokens=mi.get("completion_tokens"), e2e_latency=mi.get("e2e_latency")))
    if i % 20 == 0 or i == len(docs) - 1:
        print(f"[client] {i + 1}/{len(docs)} doc {d['doc_id']} tok {len(gen)} eff {te} wall {wall:.3f}s elapsed {time.time() - t_all:.0f}s", flush=True)
time.sleep(1)
S = stats_lines(); by = {}
for s in S: by.setdefault(s["rid"], []).append(s)
for r in recs:
    ss = by.get(r["rid"], [])
    x = [s for s in ss if s["k"] == "x"]; dd = [s for s in ss if s["k"] == "d"]
    r["steps"] = len(dd); r["log"] = [s["n"] for s in dd]; r["c"] = [s["c"] for s in dd]
    if x and dd:
        r["prefill_end"] = x[-1]["t1"]; r["prefill_ms"] = 1000 * (x[-1]["t1"] - x[-1]["t0"])
        r["decode_sec"] = dd[-1]["t_out"] - x[-1]["t1"]
        r["sec"] = r["decode_sec"] + r["prefill_ms"] / 1000
        r["step_ms"] = [1000 * (s["t_out"] - s["t_in"]) for s in dd]
        r["gpu_ms"] = [1000 * (s["t_gpu"] - s["t_in"]) for s in dd]
        r["gap_ms"] = [1000 * s["gap"] for s in dd[1:] if s.get("gap") is not None]
        r["first_gap_ms"] = 1000 * (dd[0]["t_in"] - x[-1]["t1"])
    else:
        r["sec"] = r["wall"]; r["prefill_ms"] = 0.0
OUT = dict(meta=dict(url=a.url, key=a.key, docs=a.docs, max_new=a.max_new, time=time.time()), modes={a.key: recs}, summary={})
ok = [r for r in recs if "decode_sec" in r]
if ok:
    tok = sum(r["tokens_eff"] - 1 for r in ok); dsec = sum(r["decode_sec"] for r in ok)
    allst = [x for r in ok for x in r["step_ms"]]; allg = [x for r in ok for x in r["gpu_ms"]]; allgap = [x for r in ok for x in r["gap_ms"]]
    nst = sum(r["steps"] for r in ok)
    summ = dict(docs=len(recs), tokens=sum(r["tokens"] for r in recs), tokens_eff=sum(r["tokens_eff"] for r in recs), steps=nst,
                decode_tok_per_s=tok / dsec, tok_per_step=sum(r["tokens"] for r in ok) / max(1, nst),
                ms_per_step_wall=1000 * dsec / max(1, nst - len(ok)) if nst > len(ok) else None,
                worker_step_ms_median=statistics.median(allst), worker_gpu_ms_median=statistics.median(allg),
                sched_gap_ms_median=statistics.median(allgap) if allgap else None, sched_gap_ms_mean=statistics.mean(allgap) if allgap else None,
                first_gap_ms_median=statistics.median(r["first_gap_ms"] for r in ok), prefill_ms_mean=statistics.mean(r["prefill_ms"] for r in ok),
                wall_tok_per_s=sum(r["tokens_eff"] for r in recs) / sum(r["wall"] for r in recs))
    OUT["summary"][a.key] = summ
    print("[client] SUMMARY " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items()}), flush=True)
tmp = a.out + ".tmp"; json.dump(OUT, open(tmp, "w")); os.replace(tmp, a.out)
