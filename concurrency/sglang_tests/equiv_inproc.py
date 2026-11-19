"""equiv_inproc.py -- GPU, one process: our executor (harness.Harness = the server's LateStepExecutor / ARStepExecutor + LateReqState +
prefill) vs the standalone late2 decoders (late.LateLattice, engine.Lattice B=0) on the real v2 weights.

  python tests/equiv_inproc.py --model /tmp/sgl_v2 --out results/equiv_<tag>.json [--fp32 1] [--bf16 1] [--docs 0:100]
                               [--modes l2:lkv:am:latt:4,l2:lkv:am:latt:8] [--gve 10] [--ar-docs 0:20]

--fp32: fp32 weights, torch attention (FA3 has no fp32 path), no graphs, 4 prompts x 64 new tokens, modes b0 / latt:4 / latt:8 / ar:
        token-for-token equality (acceptance (a), GPU part).
--bf16: bf16, paged FA3, CUDA graphs, graph prefill (pad 64) on both sides, W = S = 2560, GSM8K docs, lm-eval stop strings, 1024 new:
        tokens + per-step log identical per doc (acceptance (b) core), decode tok/s of both in the same process (acceptance (c) core),
        graph == eager on --gve docs.
"""
import argparse, json, os, statistics, sys, time
import torch
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--prompts", default="decode/data/gsm8k_v2_prompts.jsonl")
ap.add_argument("--fp32", type=int, default=1); ap.add_argument("--bf16", type=int, default=1)
ap.add_argument("--docs", default="0:100"); ap.add_argument("--modes", default="l2:lkv:am:latt:4,l2:lkv:am:latt:8")
ap.add_argument("--ar-docs", default="0:20"); ap.add_argument("--gve", type=int, default=10); ap.add_argument("--max-new", type=int, default=1024)
ap.add_argument("--burn-s", type=float, default=15.0)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "decode"))      # job snapshot: <code>/late2 = copies of late2/*.py
import engine as E
import late as LT
from sglang_late.core_ops import Cfg, OuroCore
from sglang_late.harness import Harness
T0 = time.time()
def log(*x): print(f"[equiv {time.time() - T0:7.1f}s]", *x, flush=True)
OUT = dict(meta=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, argv=sys.argv), fp32=[], bf16={}, gve=[], timing={})
def save(): tmp = a.out + ".tmp"; json.dump(OUT, open(tmp, "w")); os.replace(tmp, a.out)
meta = json.load(open(os.path.join(os.path.dirname(a.prompts), "meta_" + os.path.basename(a.prompts).replace(".jsonl", ".json"))))
NV, EOS, MASK = meta["nv"], meta["eos"], meta["mask_id"]
docs = [json.loads(l) for l in open(a.prompts)]
from tokenizers import Tokenizer
tokzr = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
def has_stop(ids, until): return any(s in tokzr.decode(ids[-64:], skip_special_tokens=False) for s in until)
def eff_len(gen, until):
    for k in range(1, len(gen) + 1):
        if has_stop(gen[:k], until): return k
    return len(gen)
S = 2560


def ref_dec(eng, mode, attn, graphs):
    if mode == "ar": return E.Lattice(eng, 0, NV, EOS, MASK, attn=attn, graphs=graphs)
    p = LT.parse_late(mode)
    return LT.LateLattice(eng, p["B"], p["sets"], NV, EOS, MASK, read=p["read"], ver="am", attn=attn, graphs=graphs)


def first_diff(x, y):
    return next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), None if len(x) == len(y) else min(len(x), len(y)))


# ------------------------------------------------------------------ fp32 (acceptance (a), GPU part)
if a.fp32:
    W, cfg = E.load_weights(a.model, "cuda", torch.float32)
    eng = LT.LateEngine(W, cfg, "cuda", torch.float32, S=S, fused=True, log=log, M=8, Bmax=16)
    core = OuroCore(Cfg(json.load(open(os.path.join(a.model, "config.json")))), "cuda", torch.float32, W=W, fused=True, log=log); del W
    log(f"fp32 ops late {eng.ops.chosen} ours {core.ops.chosen}")
    for mode in ["l2:lkv:am:b0", "l2:lkv:am:latt:4", "l2:lkv:am:latt:8", "ar"]:
        ref = ref_dec(eng, mode, "torch", False)
        hz = Harness(core, mode, S, NV, EOS, MASK, size=S + 64, attn="torch", graphs=False, perm_seed=11, prefill_pad=0)
        for d in docs[:4]:
            r0 = ref.decode(d["ids"], 64, prefill_attn="torch"); r1 = hz.decode(d["ids"], 64)
            eq = r0["gen_raw"] == r1["gen_raw"] and r0["log"] == r1["log"]
            OUT["fp32"].append(dict(mode=mode, doc=d["doc_id"], eq=eq, n=len(r1["gen_raw"]), steps=r1["steps"], first_diff=first_diff(r0["gen_raw"], r1["gen_raw"])))
            log(f"fp32 {mode:20s} doc {d['doc_id']}: identical {eq} ({len(r1['gen_raw'])} tok, {r1['steps']} steps)")
        del hz; torch.cuda.empty_cache()
    save(); del eng, core; torch.cuda.empty_cache()

# ------------------------------------------------------------------ bf16 (acceptance (b)/(c) core)
if a.bf16:
    W, cfg = E.load_weights(a.model, "cuda", torch.bfloat16)
    eng = LT.LateEngine(W, cfg, "cuda", torch.bfloat16, S=S, fused=True, log=log, M=8, Bmax=16)
    core = OuroCore(Cfg(json.load(open(os.path.join(a.model, "config.json")))), "cuda", torch.bfloat16, W=W, fused=True, log=log); del W
    log(f"bf16 ops late {eng.ops.chosen} ours {core.ops.chosen}"); OUT["meta"]["ops"] = [eng.ops.chosen, core.ops.chosen]
    d0, d1 = map(int, a.docs.split(":")); sel = [d for d in docs if d0 <= d["doc_id"] < d1]
    eng.prefill_graphs = True
    for Pb in sorted({-(-len(d["ids"]) // 64) * 64 for d in sel}): eng._pgraph(Pb, NV)
    x = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16); te = time.time() + a.burn_s
    while time.time() < te:
        for _ in range(20): torch.nn.functional.linear(x, eng.layers[0]["gu"])
        torch.cuda.synchronize()
    ar0, ar1 = map(int, a.ar_docs.split(":"))
    for mode in a.modes.split(",") + (["ar"] if ar1 > ar0 else []):
        dsel = sel if mode != "ar" else [d for d in docs if ar0 <= d["doc_id"] < ar1]
        ref = ref_dec(eng, mode, "paged", True); ref.capture()
        hz = Harness(core, mode, S, NV, EOS, MASK, size=4096, attn="paged", graphs=True, perm_seed=5, prefill_pad=64)
        ref.decode(dsel[0]["ids"], 32); hz.decode(dsel[0]["ids"], 32)                       # warm-up
        recs = []
        for i, d in enumerate(dsel):
            until = list(d.get("until") or []); sf = (lambda g, u=until: has_stop(g, u)) if until else None
            runs = {}
            for who in (("ref", "ours") if i % 2 == 0 else ("ours", "ref")):
                dec = ref if who == "ref" else hz
                torch.cuda.synchronize(); t0 = time.perf_counter()
                r = dec.decode(d["ids"], a.max_new, stop_fn=sf)                                  # (alternating order per doc)
                torch.cuda.synchronize(); r["sec"] = time.perf_counter() - t0; runs[who] = r
            r0, r1 = runs["ref"], runs["ours"]
            te0 = min(eff_len(r0["gen"], until), a.max_new); te1 = min(eff_len(r1["gen"], until), a.max_new)
            eq = r0["gen_raw"] == r1["gen_raw"] and r0["log"] == r1["log"]
            recs.append(dict(doc=d["doc_id"], eq=eq, first_diff=first_diff(r0["gen_raw"], r1["gen_raw"]), n=len(r1["gen_raw"]), steps=r1["steps"],
                             tok_eff=[te0, te1], sec=[r0["sec"], r1["sec"]], prefill_ms=[r0["prefill_ms"], r1["prefill_ms"]],
                             step_med=[statistics.median(r0["step_ms"]) if r0["step_ms"] else None, statistics.median(r1["step_ms"]) if r1["step_ms"] else None]))
            if not eq or i % 20 == 0:
                log(f"bf16 {mode} doc {d['doc_id']}: identical {eq} first diff {recs[-1]['first_diff']} n {len(r1['gen_raw'])} "
                    f"sec {r0['sec']:.3f}/{r1['sec']:.3f}")
        OUT["bf16"][mode] = recs
        tm = {}
        for j, who in enumerate(("late2", "ours")):
            tok = sum(r["tok_eff"][j] - 1 for r in recs); sec = sum(r["sec"][j] - r["prefill_ms"][j] / 1000 for r in recs)
            tm[who] = dict(decode_tok_s=tok / sec, step_ms_median=statistics.median([r["step_med"][j] for r in recs if r["step_med"][j]]),
                           prefill_ms_mean=statistics.mean(r["prefill_ms"][j] for r in recs))
        tm["identical_docs"] = sum(r["eq"] for r in recs); tm["docs"] = len(recs); tm["ratio_ours_over_late2"] = tm["ours"]["decode_tok_s"] / tm["late2"]["decode_tok_s"]
        OUT["timing"][mode] = tm
        log(f"MODE {mode}: identical {tm['identical_docs']}/{len(recs)}; decode tok/s late2 {tm['late2']['decode_tok_s']:.1f} ours {tm['ours']['decode_tok_s']:.1f} "
            f"(x{tm['ratio_ours_over_late2']:.3f}); step median {tm['late2']['step_ms_median']:.3f} / {tm['ours']['step_ms_median']:.3f} ms")
        save()
        if mode != "ar" and a.gve > 0:                                                       # graph == eager on our side
            he = Harness(core, mode, S, NV, EOS, MASK, size=4096, attn="paged", graphs=False, perm_seed=5, prefill_pad=64)
            for d in sel[: a.gve]:
                until = list(d.get("until") or []); sf = (lambda g, u=until: has_stop(g, u)) if until else None
                rg = hz.decode(d["ids"], 256, stop_fn=sf); re_ = he.decode(d["ids"], 256, stop_fn=sf)
                same = rg["gen_raw"] == re_["gen_raw"] and rg["log"] == re_["log"]
                OUT["gve"].append(dict(mode=mode, doc=d["doc_id"], identical=same)); log(f"graph vs eager {mode} doc {d['doc_id']}: identical {same}")
            del he
        del hz, ref; torch.cuda.empty_cache(); save()
OUT["meta"]["done"] = True; save(); log("DONE")
