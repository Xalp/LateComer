"""run_la.py -- LoopAhead-repro driver, mirroring run_latek.py's protocol conventions (same timing, stop strings,
warm-up, burn, audit). Runs engine-AR FIRST and LoopAhead in the SAME process on the SAME GPU, so every ratio
shares a denominator.

Modes: ar | la (full LoopAhead: scheduling + suspension + LAD + draft vocab) | la:nolad (no drafter: the
pipelined native-proposal tree, their no-LAD ablation arm).

⚠ --docs selects by POSITION in the prompt file (i0:i1), not by doc_id (positional slicing);
the output records n and meta.done.

  python run_la.py --model /tmp/ouro26 --prompts fair_gsm8k_prompts.jsonl --docs 0:120 --modes ar,la \
      --lad-ckpt .../lad_ouro26.pt --draft-vocab .../draft_vocab.json --max-new 1024 --out results/....json
"""
import argparse, hashlib, json, os, socket, statistics, sys, time, traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "decode"))
import engine as E
import late as LT
from loopahead import LACfg, LAEngine, LADRunner, LoopAheadDecoder
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "drafter"))
from lad import LAD

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--meta", default="")
ap.add_argument("--docs", default="0:20", help="POSITIONAL slice i0:i1 of the prompt file")
ap.add_argument("--modes", default="ar,la")
ap.add_argument("--lad-ckpt", default="")
ap.add_argument("--draft-vocab", default="")
ap.add_argument("--max-new", type=int, default=1024)
ap.add_argument("--until", type=int, default=1)
ap.add_argument("--graphs", type=int, default=1)
ap.add_argument("--fused", type=int, default=1)
ap.add_argument("--S", type=int, default=0, help="0 = auto from prompts + max_new")
ap.add_argument("--burn-s", type=float, default=20.0)
ap.add_argument("--out", required=True)
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--prefill-graph", type=int, default=1)
ap.add_argument("--save-gen", type=int, default=1)
ap.add_argument("--lossless-check", type=int, default=3, help="first N docs: la vs ar token identity (256 tok)")
ap.add_argument("--graph-vs-eager", type=int, default=1)
ap.add_argument("--audit", type=int, default=1)
a = ap.parse_args()
T0 = time.time()
def log(*x): print(f"[la {time.time()-T0:7.1f}s]", *x, flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT = dict(meta=dict(host=socket.gethostname(), argv=sys.argv, torch=torch.__version__, cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name() if dev == "cuda" else "cpu", done=False,
                     reimplementation="OUR reimplementation of LoopAhead (arXiv 2610.21594); assumptions in PLAN.md"),
           modes={}, summary={}, checks={}, audit={})
def save():
    tmp = a.out + ".tmp"; json.dump(OUT, open(tmp, "w")); os.replace(tmp, a.out)

meta_path = a.meta or os.path.join(os.path.dirname(a.prompts), "meta_" + os.path.basename(a.prompts).replace(".jsonl", ".json"))
PM = json.load(open(meta_path))
docs_all = [json.loads(l) for l in open(a.prompts)]
d0, d1 = (int(x) for x in a.docs.split(":"))
docs = docs_all[d0:d1]                                     # POSITIONAL
OUT["meta"]["docs"] = a.docs; OUT["meta"]["n_docs"] = len(docs); OUT["meta"]["doc_ids"] = [d["doc_id"] for d in docs]
MODES = [m for m in a.modes.split(",") if m]
DT = getattr(torch, a.dtype); OUT["meta"]["dtype"] = a.dtype

tw = time.time(); W, cfg = E.load_weights(a.model, dev, DT)
NV = min(PM["nv"], W["lm_head.weight"].shape[0])           # released model: 49152 (no <|mask|> row)
EOS = [e for e in PM["eos"] if e < NV]
S = a.S or (max(len(d["ids"]) for d in docs) + a.max_new + 96)
log(f"{len(docs)} docs (positional {a.docs}), nv {NV} eos {EOS}, modes {MODES}, max_new {a.max_new}, S {S}")
eng = LAEngine(W, cfg, dev, DT, S=S, fused=bool(a.fused), log=log); del W
OUT["meta"].update(nv=NV, eos=EOS, S=S, ops=eng.ops.chosen,
                   md5={f: hashlib.md5(open(os.path.join(HERE, f), "rb").read()).hexdigest() for f in ("loopahead.py", "run_la.py")})
if dev == "cuda": log(f"GPU mem {torch.cuda.memory_allocated()/2**30:.1f} GB")

tokzr = None
if a.until:
    from tokenizers import Tokenizer
    tokzr = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
def has_stop(ids, until):
    return any(s in tokzr.decode(ids[-64:], skip_special_tokens=False) for s in until)
def eff_len(gen, until):
    if not until or tokzr is None: return len(gen)
    for k in range(1, len(gen) + 1):
        if has_stop(gen[:k], until): return k
    return len(gen)

# ---------------------------------------------------------------- attention / numerics checks
if dev == "cuda":
    try:
        err = E.check_attention(eng, 0, "paged", Lc=137, log=log)
        OUT["checks"]["lattice_paged"] = err
        assert err < 0.02
    except Exception:
        OUT["meta"]["fatal"] = "paged attention unavailable"; log(traceback.format_exc()[-1200:]); save(); sys.exit(5)

def check_la_mixed(variant, Lc=137, seed=0):
    """one mixed-depth 16-row forward, paged (FA3) vs torch, on random pool content. Bisecting variants:
    prefix_t0      : every row pass 0, reads prefix+self only (page values < S, like the lattice)
    prefix_mixed   : rows at random passes, prefix+self only (page values offset by t*S)
    chains_t0      : 4 chains x 4 rows, all at pass 0
    chains         : chains at random per-chain passes (the real structure)
    shuffled / dup : chains with shuffled anc order / a duplicated page (diagnostic only)"""
    import random as _r
    rng = _r.Random(seed)
    torch.manual_seed(seed)
    for li in range(cfg.NL):
        eng.Kp[li].normal_(); eng.Vp[li].normal_()
    la = LoopAheadDecoder(eng, NV, EOS, cfg=LACfg(), lad=None, attn="paged", graphs=False)
    la._dummy_rows()
    i = 0
    for chain in range(4):
        t = 0 if variant in ("prefix_t0", "chains_t0") else rng.randrange(cfg.T)
        slots = [S - 2 - (chain * 4 + k) for k in range(4)]
        for k in range(4):
            if variant.startswith("prefix"):
                anc = [t * S + slots[k]]
            else:
                anc = sorted(t * S + s_ for s_ in slots[: k + 1])
                if variant == "shuffled": rng.shuffle(anc)
                if variant == "dup" and len(anc) > 1: anc[0] = anc[-1]
            la.b_node[i] = i; la.b_kind[i] = rng.randrange(2); la.b_tok[i] = rng.randrange(NV)
            la.b_t[i] = t; la.b_slot[i] = slots[k]; la.b_pos[i] = Lc + (0 if variant.startswith("prefix") else k)
            la.b_lc[i] = Lc; la.b_nanc[i] = len(anc)
            la.b_anc[i].zero_()
            for j, pg in enumerate(anc): la.b_anc[i, j] = pg
            i += 1
    torch.manual_seed(seed + 1); la.nX.normal_()
    m = la.meta
    m.pos.copy_(la.b_pos); m.slot.copy_(la.b_t * S + la.b_slot); m.seqlens.copy_(la.b_lc + la.b_nanc)
    tS = (la.b_t.to(torch.int32) * S)[:, None]
    rel = (la.col - la.b_lc[:, None]).clamp(0, la.cfg.maxanc - 1)
    m.page_table.copy_(torch.where(la.col < la.b_lc[:, None], tS + la.col, la.b_anc.gather(1, rel.long()).to(torch.int32)))
    # ONE attention call (the engine's own check style: a full 48-layer bf16 pass compounds rounding ~0.1-0.25 on
    # random inputs and is not a valid gate -- measured 2026-10-30, paged == paged_b bitwise there)
    from loopahead import attn_paged_batch
    c = eng.c; H = c.H
    qkv = torch.randn(la.R, 3 * H, device=dev, dtype=eng.dt)
    q, k, v = qkv[:, :H], qkv[:, H:2 * H], qkv[:, 2 * H:]
    PS = c.T * S
    kc = eng.Kp[0].view(PS, 1, c.NH, c.HD); vc = eng.Vp[0].view(PS, 1, c.NH, c.HD)
    kc.view(PS, H).index_copy_(0, m.slot, k); vc.view(PS, H).index_copy_(0, m.slot, v)
    ref = E.attn_torch(q, kc, vc, m, c.NH, c.HD)
    out = {}
    for pb, fn in (("paged", E.attn_paged), ("paged_b", attn_paged_batch)):
        oa = fn(q, kc, vc, m, c.NH, c.HD)
        per = (oa.float() - ref.float()).abs().amax(-1)
        out[pb] = (per.max().item(), [round(x, 4) for x in per.tolist()])
    return out, la.b_t.tolist()

LA_ATTN = "paged"
if dev == "cuda":
    best = {}
    for variant in ("prefix_t0", "prefix_mixed", "chains_t0", "chains", "shuffled", "dup"):
        res, ts_ = check_la_mixed(variant)
        for pb, (err, per) in res.items():
            OUT["checks"][f"la_mixed_{variant}_{pb}"] = err
            log(f"[check] LA mixed 16-row {pb} vs torch [{variant}]: max {err:.5f} per-row {per} t {ts_}")
            best.setdefault(pb, 0.0)
            if variant in ("prefix_t0", "prefix_mixed", "chains_t0", "chains"):
                best[pb] = max(best[pb], err)
    ok = [pb for pb in ("paged", "paged_b") if best[pb] < 0.02]
    if not ok:
        OUT["meta"]["fatal"] = f"mixed-depth attention mismatch (both forms): {best}"; save()
        if not int(os.environ.get("LA_CHECK_ONLY", "0")): sys.exit(6)
    else:
        LA_ATTN = ok[0]
        OUT["meta"]["la_attn"] = LA_ATTN
        log(f"[check] LA attention backend: {LA_ATTN} (max gated err {best})")
    if int(os.environ.get("LA_CHECK_ONLY", "0")):
        log("LA_CHECK_ONLY set: exiting after checks"); save(); sys.exit(0)
PREFILL = "auto"

def burn(sec):
    if dev != "cuda" or sec <= 0: return
    x = torch.randn(512, 2048, device=dev, dtype=torch.bfloat16); w = eng.layers[0]["gu"]; te = time.time() + sec
    while time.time() < te:
        for _ in range(20): torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()

# ---------------------------------------------------------------- decoders
LADR = None
if any(m == "la" for m in MODES):
    assert a.lad_ckpt and a.draft_vocab, "la mode needs --lad-ckpt and --draft-vocab"
    ck = torch.load(a.lad_ckpt, map_location="cpu", weights_only=False)
    lad_model = LAD()
    lad_model.load_state_dict(ck["sd"])
    dvj = json.load(open(a.draft_vocab))
    LADR = LADRunner(eng, lad_model, dvj["vocab"], EOS, max_new=a.max_new)
    OUT["meta"]["lad"] = dict(ckpt=a.lad_ckpt, upd=ck.get("upd"), dev=ck.get("dev"), draft_vocab_n=len(dvj["vocab"]))
    log(f"LAD loaded: upd {ck.get('upd')}, dev {ck.get('dev', {})}")

DEC = {}
def get_dec(mode):
    if mode in DEC: return DEC[mode]
    if mode == "ar":
        L = E.Lattice(eng, 0, nv=NV, eos=EOS, mask_id=0, attn="paged" if dev == "cuda" else "torch",
                      graphs=bool(a.graphs) and dev == "cuda")
        L.capture()
    else:
        L = LoopAheadDecoder(eng, NV, EOS, cfg=LACfg(), lad=None if mode == "la:nolad" else LADR,
                             attn=LA_ATTN if dev == "cuda" else "torch", graphs=bool(a.graphs) and dev == "cuda")
        L.capture()
    DEC[mode] = L
    return L

if a.prefill_graph and dev == "cuda":
    eng.prefill_graphs = True; tc = time.time()
    for Pb in sorted({-(-len(d["ids"]) // eng.pg_bucket) * eng.pg_bucket for d in docs}): eng._pgraph(Pb, NV)
    log(f"prefill graphs for buckets {sorted(k[0] for k in eng.pgraphs)} ({time.time()-tc:.1f}s)")

# graph vs eager + lossless checks (not timed)
if dev == "cuda" and a.graph_vs_eager and any(m.startswith("la") for m in MODES):
    mode = [m for m in MODES if m.startswith("la")][0]
    Lg = get_dec(mode)
    Le = LoopAheadDecoder(eng, NV, EOS, cfg=LACfg(), lad=Lg.lad, attn=LA_ATTN, graphs=False)
    for d in docs[:2]:
        rg = Lg.decode(d["ids"], 128, prefill_attn=PREFILL); re_ = Le.decode(d["ids"], 128, prefill_attn=PREFILL)
        same = rg["gen_raw"] == re_["gen_raw"]
        OUT["checks"].setdefault("graph_vs_eager", []).append(dict(doc=d["doc_id"], identical=same))
        log(f"graph vs eager {mode} doc {d['doc_id']}: identical {same}")
    save()

if dev == "cuda" and a.lossless_check:
    Lar = get_dec("ar")
    for mode in [m for m in MODES if m.startswith("la")]:
        Lla = get_dec(mode); nid = 0
        for d in docs[: a.lossless_check]:
            rr = Lar.decode(d["ids"], 256, prefill_attn=PREFILL); rl = Lla.decode(d["ids"], 256, prefill_attn=PREFILL)
            same = rr["gen_raw"] == rl["gen_raw"]; nid += same
            OUT["checks"].setdefault("lossless_vs_ar", []).append(dict(mode=mode, doc=d["doc_id"], identical=same,
                                                                       n=len(rr["gen_raw"])))
        log(f"lossless check {mode}: identical to AR on {nid}/{min(a.lossless_check, len(docs))} docs (256 tok; bf16 near-ties may flip)")
    save()

burn(a.burn_s)

# ---------------------------------------------------------------- timed decode
for mode in MODES:
    L = get_dec(mode)
    L.decode(docs[0]["ids"], 32, prefill_attn=PREFILL)       # warm-up
    recs = []
    for d in docs:
        until = list(d.get("until") or []) if a.until else []
        sf = (lambda g, u=until: has_stop(g, u)) if until else None
        if dev == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter(); r = L.decode(d["ids"], a.max_new, stop_fn=sf, prefill_attn=PREFILL)
        if dev == "cuda": torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        gen = r["gen"]; te = min(eff_len(gen, until), a.max_new)
        rec = dict(doc_id=d["doc_id"], prompt_tokens=len(d["ids"]), tokens=len(gen), tokens_eff=te, steps=r["steps"],
                   log=r["log"], sec=sec, prefill_ms=r["prefill_ms"], step_ms=[round(x, 4) for x in r["step_ms"]],
                   finish="eos" if len(r["gen_raw"]) > len(gen) else ("stop" if until and has_stop(gen[:te], until) else "length"))
        if "stat" in r: rec["stat"] = r["stat"]; rec["rows_log_mean"] = round(statistics.mean(r["rows_log"] or [0]), 2)
        if a.save_gen: rec["gen"] = gen
        if tokzr is not None: rec["text"] = tokzr.decode(gen[:te], skip_special_tokens=False)
        recs.append(rec)
    OUT["modes"][mode] = recs
    tok = sum(x["tokens"] for x in recs); tok_e = sum(x["tokens_eff"] for x in recs)
    sec = sum(x["sec"] for x in recs); steps = sum(x["steps"] for x in recs)
    allstep = [s for x in recs for s in x["step_ms"]]
    summ = dict(mode=mode, docs=len(recs), tokens=tok, tokens_eff=tok_e, sec=sec, steps=steps,
                tok_per_s=tok / sec, tok_eff_per_s=tok_e / sec, tok_per_step=tok / max(1, steps),
                decode_tok_per_s=sum(x["tokens_eff"] - 1 for x in recs) / max(1e-9, sec - sum(x["prefill_ms"] for x in recs) / 1000),
                ms_step_mean=statistics.mean(allstep) if allstep else None,
                ms_step_median=statistics.median(allstep) if allstep else None,
                prefill_ms_mean=statistics.mean(x["prefill_ms"] for x in recs))
    if mode.startswith("la"):
        st = [x["stat"] for x in recs]
        summ.update(lad_calls=sum(s["lad_calls"] for s in st), lad_drafted=sum(s["lad_drafted"] for s in st),
                    emit_native=sum(s["emit_native"] for s in st), emit_lad=sum(s["emit_lad"] for s in st),
                    emit_bonus=sum(s["emit_bonus"] for s in st), rows_mean=statistics.mean(x["rows_log_mean"] for x in recs))
    ar = OUT["summary"].get("ar")
    if ar and mode != "ar":
        summ["speedup_vs_ar"] = summ["tok_eff_per_s"] / ar["tok_eff_per_s"]
        summ["decode_speedup_vs_ar"] = summ["decode_tok_per_s"] / ar["decode_tok_per_s"]
    OUT["summary"][mode] = summ
    log(f"MODE {mode}: {len(recs)} docs, {tok} tok ({tok_e} eff), {steps} steps, {summ['tok_per_step']:.3f} tok/step, "
        f"{summ['tok_eff_per_s']:.1f} tok/s (decode only {summ['decode_tok_per_s']:.1f}), step med {summ['ms_step_median']:.3f} ms"
        + (f", speedup vs ar {summ['speedup_vs_ar']:.3f}x (decode {summ['decode_speedup_vs_ar']:.3f}x)" if "speedup_vs_ar" in summ else "")
        + (f", emits n/l/b {summ['emit_native']}/{summ['emit_lad']}/{summ['emit_bonus']}" if mode.startswith("la") else ""))
    save()

# ---------------------------------------------------------------- audit
if a.audit:
    tA = time.time()
    by_id = {d["doc_id"]: d for d in docs}
    for key, recs in OUT["modes"].items():
        nd_ = nt = 0; per = []
        for rec in recs:
            g = rec.get("gen", [])[: rec["tokens_eff"]]; P0 = rec["prompt_tokens"]
            if not g: per.append([rec["doc_id"], 0, 0]); continue
            ids = by_id[rec["doc_id"]]["ids"] + g
            fd, mg = LT.audit_full_depth(eng, ids, P0, len(g), NV)
            diffs = [i for i in range(len(g)) if fd[i] != g[i]]
            nd_ += len(diffs); nt += len(g)
            per.append([rec["doc_id"], len(g), len(diffs)] + ([diffs[0], round(mg[diffs[0]], 4)] if diffs else []))
        OUT["audit"][key] = dict(tokens=nt, diff=nd_, share=nd_ / max(1, nt), per_doc=per)
        log(f"AUDIT {key}: {nd_}/{nt} committed tokens differ from full-depth argmax = {nd_/max(1,nt):.5f}")
        save()
    log(f"audit done in {time.time()-tA:.0f}s")

OUT["meta"]["done"] = True
save()
log("DONE")
