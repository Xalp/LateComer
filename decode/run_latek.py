"""run_latek.py -- copy of ../late2_full/run_late.py adapted to the K-PASS LATE LATTICE (latek.py). Modes: ar | latt:B | sp:B:sets (exact
references, hybrid attention, unchanged decoders running on the LateEngine's per-loop cache views) | lk:<K>:<shape> (latek.py; paged FA3
only: one varlen call per layer over [active background groups | token slots | sets]).
Extra GPU check (--k4-check N): every lk:4:latt:B mode vs the exact dense lattice latt:B (paged) on the first N docs (256 tokens each):
identical tokens and per-step log expected (K = 4 is the dense exact lattice).
Per doc additionally (l2): c per step, real proposals per step (nprop), per-step bitmask of proposals equal to the verifying slot's argmax
(mbits), n_acc / n_gate (accepted != argmax) / n_amrej (argmax-equal proposals refused by the gate).
--audit 1: after all timed decodes, a teacher-forced full-depth T-pass forward over prompt + scored generation of every doc of every mode;
per doc n_diff = committed tokens != full-depth argmax given the decoder's own prefix (+ the full-depth top-2 margin at each difference).
Everything else (timing, stop strings, warm-up, burn, prefill graphs, AR first) as run_gate.py.
"""
import argparse, hashlib, json, os, socket, statistics, sys, time, traceback
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
import sparse as SP
import late as LT
import latek as LK

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--prompts", required=True); ap.add_argument("--meta", default="")
ap.add_argument("--docs", default="0:20"); ap.add_argument("--modes", default="ar"); ap.add_argument("--attn", default="hybrid")
ap.add_argument("--max-new", type=int, default=1024); ap.add_argument("--until", type=int, default=1)
ap.add_argument("--graphs", type=int, default=1); ap.add_argument("--fused", type=int, default=1); ap.add_argument("--S", type=int, default=2560)
ap.add_argument("--microbench", default="")  # ";"-separated modes
ap.add_argument("--mb-L", default="300,1024"); ap.add_argument("--mb-reps", type=int, default=100)
ap.add_argument("--burn-s", type=float, default=20.0); ap.add_argument("--out", required=True)
ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--prefill-graph", type=int, default=1); ap.add_argument("--save-gen", type=int, default=1)
ap.add_argument("--graph-vs-eager", type=int, default=0); ap.add_argument("--audit", type=int, default=1); ap.add_argument("--k4-check", type=int, default=3)
a = ap.parse_args()
T0 = time.time()
def log(*x): print(f"[late {time.time() - T0:7.1f}s]", *x, flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT = dict(meta=dict(host=socket.gethostname(), argv=sys.argv, torch=torch.__version__, cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name() if dev == "cuda" else "cpu"), modes={}, summary={}, microbench=[], checks=[], audit={})
def save():
    tmp = a.out + ".tmp"; json.dump(OUT, open(tmp, "w")); os.replace(tmp, a.out)

meta_path = a.meta or os.path.join(os.path.dirname(a.prompts), "meta_" + os.path.basename(a.prompts).replace(".jsonl", ".json"))
PM = json.load(open(meta_path)); NV, EOS, MASK = PM["nv"], PM["eos"], PM["mask_id"]
docs = [json.loads(l) for l in open(a.prompts)]
d0, d1 = (int(x) for x in a.docs.split(":")); docs = [d for d in docs if d0 <= d["doc_id"] < d1]
MODES = [m for m in a.modes.split(",") if m]
log(f"{len(docs)} prompts, nv {NV} eos {EOS} mask {MASK}, modes {MODES}, attn {a.attn}, max_new {a.max_new}, until {a.until}")

DT = getattr(torch, a.dtype); OUT["meta"]["dtype"] = a.dtype
tw = time.time(); W, cfg = E.load_weights(a.model, dev, DT); log(f"weights loaded in {time.time() - tw:.1f}s ({len(W)} tensors, {a.dtype})")
eng = LT.LateEngine(W, cfg, dev, DT, S=a.S, fused=bool(a.fused), log=log, M=8, Bmax=16); del W
OUT["meta"]["ops"] = eng.ops.chosen
HERE = os.path.dirname(os.path.abspath(__file__))
OUT["meta"]["md5"] = {f: hashlib.md5(open(os.path.join(HERE, f), "rb").read()).hexdigest() for f in ("engine.py", "rolled.py", "sparse.py", "late.py", "latek.py", "run_latek.py")}
log(f"code md5 {OUT['meta']['md5']}")
if dev == "cuda": log(f"GPU mem allocated {torch.cuda.memory_allocated() / 2**30:.1f} GB")

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

# ---------------------------------------------------------------- attention checks
avail = ["torch"]
if dev == "cuda" and a.attn != "torch":
    for at in ("paged", "split"):
        try:
            errs = [E.check_attention(eng, B, at, Lc=Lc, log=log) for B in (0, 2, 8) for Lc in (37, 300)]
            errs += [SP.check_attention_mode(eng, md, at, Lc=Lc, log=log) for md in ("sp:8:0+1+8", "sp:13:0+1+2+3+5+8+11") for Lc in (37, 300)]
            ok = max(errs) < 0.02
            OUT["checks"].append(dict(kind="attention", attn=at, max_err=max(errs), ok=ok))
            if ok: avail.append(at)
        except Exception:
            OUT["checks"].append(dict(kind="attention", attn=at, error=traceback.format_exc()[-1500:])); log(f"attention {at} FAILED:\n{traceback.format_exc()[-1500:]}")
    lmodes = [m for m in MODES if m.startswith(("lk:","lknb:","lkq:","lkdb:"))]
    try:
        errs = [LK.check_attention_latek(eng, md, hdr_base=h, log=log) for md in lmodes for h in (137, 60, 900)]
        ok = (max(errs) if errs else 0) < 0.02
        OUT["checks"].append(dict(kind="attention_late", modes=lmodes, max_err=max(errs) if errs else None, ok=ok))
        if not ok: log("FATAL: late attention check failed"); OUT["meta"]["fatal"] = "late attention"; save(); sys.exit(6)
    except SystemExit: raise
    except Exception:
        OUT["checks"].append(dict(kind="attention_late", error=traceback.format_exc()[-1500:])); log(traceback.format_exc()[-1500:]); save(); sys.exit(6)
    save()
log(f"attention backends available: {avail}")
PREFILL = "auto" if ("paged" in avail or dev != "cuda") else "torch"
if dev == "cuda" and a.attn != "torch" and len(avail) == 1:
    log("FATAL: no fused attention backend works on this GPU"); OUT["meta"]["fatal"] = "no attention backend"; save(); sys.exit(5)


def burn(sec):
    if dev != "cuda" or sec <= 0: return
    x = torch.randn(512, 2048, device=dev, dtype=torch.bfloat16); w = eng.layers[0]["gu"]; te = time.time() + sec
    while time.time() < te:
        for _ in range(20): torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()


@torch.no_grad()
def time_step(L, lc, reps):
    """ms per step at prefix length lc, CUDA events, median (exact lattice: T replays; late lattice: its 2 call replays)"""
    if isinstance(L, LK.LateKLattice):
        D = L.D; L.hdr.copy_(torch.tensor([lc - i * D for i in range(L.NG + 1)], device=L.hdr.device)); L.ids.fill_(MASK)
        run = L.replay
    else:
        L.lcur.fill_(lc); L.ids.fill_(MASK); run = lambda: L.run_passes(lc)
    for _ in range(5): run()
    torch.cuda.synchronize(); ev = [torch.cuda.Event(enable_timing=True) for _ in range(reps + 1)]; ev[0].record()
    for i in range(reps): run(); ev[i + 1].record()
    torch.cuda.synchronize(); ts = sorted(ev[i].elapsed_time(ev[i + 1]) for i in range(reps))
    return ts[len(ts) // 2]


DEC = {}
def get_dec(B, attn):
    key = (B, attn)
    if key not in DEC:
        tc = time.time(); L = LK.make_decoder(eng, B, NV, EOS, MASK, attn=attn, graphs=bool(a.graphs) and dev == "cuda", margins=False, switch_L=SWITCH.get(B)); L.capture()
        DEC[key] = L; log(f"decoder {B} attn={attn}: N={L.N} rows, {L.nseq} seqs, graphs {L.graphs is not None} ({time.time() - tc:.1f}s)")
    return DEC[key]


SWITCH = {}
def pick_attn(B):
    """late modes: paged; exact modes: hybrid as run_gate.py (switch to split at the smallest probed prefix length where it is faster)"""
    if B.startswith(("lk:","lknb:","lkq:","lkdb:")): return "paged" if dev == "cuda" else "torch"
    if a.attn == "hybrid" and dev == "cuda" and "split" in avail and "paged" in avail:
        if B not in SWITCH:
            Lp, Ls = get_dec(B, "paged"), get_dec(B, "split"); sw = 10 ** 9; prof = {}
            for Lc in (128, 256, 384, 512, 640, 768, 1024):
                mp, ms_ = time_step(Lp, Lc, 20), time_step(Ls, Lc, 20); prof[Lc] = (round(mp, 3), round(ms_, 3))
                if ms_ < mp and sw == 10 ** 9: sw = Lc
            SWITCH[B] = sw; OUT["checks"].append(dict(kind="hybrid_switch", B=B, switch_L=sw, profile=prof)); log(f"hybrid {B}: switch to split at L>={sw} ({prof})")
        return "hybrid"
    if a.attn not in ("auto", "hybrid"): return a.attn
    return "paged" if "paged" in avail else "torch"


if a.prefill_graph and dev == "cuda" and "paged" in avail:
    eng.prefill_graphs = True; tc = time.time()
    for Pb in sorted({-(-len(d["ids"]) // eng.pg_bucket) * eng.pg_bucket for d in docs}): eng._pgraph(Pb, NV)
    log(f"prefill graphs captured for buckets {sorted(k[0] for k in eng.pgraphs)} ({time.time() - tc:.1f}s)")
OUT["meta"]["prefill_graphs"] = eng.prefill_graphs
burn(a.burn_s)

# ---------------------------------------------------------------- microbench: ms per step vs prefix length
if a.microbench and dev == "cuda":
    for B in [x for x in a.microbench.split(";") if x]:
        try:
            L = get_dec(B, "paged")
            for Lc in [int(x) for x in a.mb_L.split(",")]:
                ms = time_step(L, Lc, a.mb_reps); npass = LK.parse_latek(B)["K"] if B.startswith(("lk:","lknb:","lkq:","lkdb:")) else cfg.T
                r = dict(mode=B, rows=L.N, rows_per_call=[cl.N for cl in L.calls] if B.startswith(("lk:","lknb:","lkq:","lkdb:")) else None, attn="paged", L=Lc, ms_step=ms, ms_pass=ms / npass); OUT["microbench"].append(r)
                log(f"microbench {B:40s} rows={L.N:3d} L={Lc:4d}: {ms:7.3f} ms/step = {ms / npass:.3f} ms/pass")
        except Exception:
            OUT["microbench"].append(dict(B=B, error=traceback.format_exc()[-1200:])); log(traceback.format_exc()[-1200:])
        save()

# ---------------------------------------------------------------- graph vs eager identity (2 docs, late modes)
if a.graph_vs_eager and dev == "cuda" and a.graphs:
    for B in [m for m in MODES if m.startswith(("lk:","lknb:","lkq:","lkdb:"))]:
        Lg = get_dec(B, "paged"); Le = LK.make_decoder(eng, B, NV, EOS, MASK, attn="paged", graphs=False)
        for d in docs[:2]:
            rg = Lg.decode(d["ids"], 128, prefill_attn=PREFILL); re_ = Le.decode(d["ids"], 128, prefill_attn=PREFILL)
            same = rg["gen_raw"] == re_["gen_raw"] and rg["log"] == re_["log"]
            OUT["checks"].append(dict(kind="graph_vs_eager", B=B, doc=d["doc_id"], identical=same))
            log(f"graph vs eager {B} doc {d['doc_id']}: identical {same}")
        del Le
    save()

# ---------------------------------------------------------------- K = 4 == exact dense lattice (bf16, same kernels)
if a.k4_check and dev == "cuda":
    for B in [m for m in MODES if m.startswith(f"lk:{eng.c.T}:latt:")]:
        Lk = get_dec(B, "paged"); Lx = get_dec(B.split(":", 2)[2], "paged"); nid = 0
        for d in docs[: a.k4_check]:
            rk = Lk.decode(d["ids"], 256, prefill_attn=PREFILL); rx = Lx.decode(d["ids"], 256, prefill_attn=PREFILL)
            same = rk["gen_raw"] == rx["gen_raw"] and rk["log"] == rx["log"]; nid += same
            OUT["checks"].append(dict(kind="k4_vs_exact", T=eng.c.T, B=B, doc=d["doc_id"], identical=same, n=len(rk["gen_raw"])))
        log(f"K={eng.c.T} vs exact {B.split(':', 2)[2]} (paged): identical {nid}/{min(a.k4_check, len(docs))} docs (256 tokens)")
    save()

# ---------------------------------------------------------------- decode
for mode in MODES:
    LK.parse_latek(mode) if mode.startswith(("lk:","lknb:","lkq:","lkdb:")) else SP.parse_mode(mode)
    at = pick_attn(mode) if dev == "cuda" else "torch"
    L = get_dec(mode, at)
    L.decode(docs[0]["ids"], 32, prefill_attn=PREFILL)                                                    # warm-up (not recorded)
    recs = []; key = f"{mode}@{at}"
    for d in docs:
        until = list(d.get("until") or []) if a.until else []
        sf = (lambda g, u=until: has_stop(g, u)) if until else None
        if dev == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter(); r = L.decode(d["ids"], a.max_new, stop_fn=sf, prefill_attn=PREFILL)
        if dev == "cuda": torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        gen = r["gen"]; te = min(eff_len(gen, until), a.max_new)
        rec = dict(doc_id=d["doc_id"], prompt_tokens=len(d["ids"]), tokens=len(gen), tokens_eff=te, steps=r["steps"], log=r["log"], sec=sec,
                   prefill_ms=r["prefill_ms"], step_ms=[round(x, 4) for x in r["step_ms"]], gpu_ms=[round(x, 4) for x in r["gpu_ms"]],
                   finish="eos" if len(r["gen_raw"]) > len(gen) else ("stop" if until and has_stop(gen[:te], until) else "length"))
        if a.save_gen: rec["gen"] = gen
        for k_ in ("nprop", "c", "mbits", "n_acc", "n_gate", "n_amrej", "p1", "n_db", "n_dbh"):
            if r.get(k_) is not None: rec[k_] = r[k_]
        if tokzr is not None: rec["text"] = tokzr.decode(gen[:te], skip_special_tokens=False)
        recs.append(rec)
    OUT["modes"][key] = recs
    tok = sum(x["tokens"] for x in recs); tok_e = sum(x["tokens_eff"] for x in recs); sec = sum(x["sec"] for x in recs); steps = sum(x["steps"] for x in recs)
    allstep = [s for x in recs for s in x["step_ms"]]; allgpu = [s for x in recs for s in x["gpu_ms"]]
    npass = LK.parse_latek(mode)["K"] if mode.startswith(("lk:","lknb:","lkq:","lkdb:")) else cfg.T
    summ = dict(mode=mode, attn=at, rows=L.N, rows_per_call=[cl.N for cl in L.calls] if mode.startswith(("lk:","lknb:","lkq:","lkdb:")) else None, docs=len(recs), tokens=tok, tokens_eff=tok_e, sec=sec, steps=steps, passes_per_step=npass,
                tok_per_s=tok / sec, tok_eff_per_s=tok_e / sec, tok_per_step=tok / max(1, steps), tok_per_pass=tok / max(1, steps) / npass,
                decode_tok_per_s=sum(x['tokens_eff'] - 1 for x in recs) / max(1e-9, sec - sum(x['prefill_ms'] for x in recs) / 1000),
                ms_step_mean=statistics.mean(allstep) if allstep else None, ms_step_median=statistics.median(allstep) if allstep else None,
                ms_gpu_median=statistics.median(allgpu) if allgpu else None, prefill_ms_mean=statistics.mean(x["prefill_ms"] for x in recs))
    summ["ms_pass_median"] = summ["ms_gpu_median"] / npass if summ["ms_gpu_median"] else None
    if mode.startswith(("lk:","lknb:","lkq:","lkdb:")):
        na = sum(x["n_acc"] for x in recs); ng = sum(x["n_gate"] for x in recs)
        summ.update(acc_props=na, gate_props=ng, gate_share=ng / max(1, na), amrej=sum(x["n_amrej"] for x in recs))
    if mode.startswith("lkdb:"):
        fires = sum(x.get("n_db", 0) for x in recs); rw = sum(x.get("n_dbh", 0) for x in recs)
        summ.update(db_fires=fires, db_rewrites=rw, db_fire_share=fires / max(1, steps))
    OUT["summary"][key] = summ
    ar = next((v for k, v in OUT["summary"].items() if k.startswith("ar@")), None)
    if ar and mode != "ar": summ["speedup_vs_ar"] = summ["tok_eff_per_s"] / ar["tok_eff_per_s"]; summ["decode_speedup_vs_ar"] = summ["decode_tok_per_s"] / ar["decode_tok_per_s"]
    log(f"MODE {key}: {len(recs)} docs, {tok} tok ({tok_e} eff), {steps} steps, {summ['tok_per_step']:.3f} tok/step ({summ['tok_per_pass']:.3f} tok/pass), "
        f"{summ['tok_eff_per_s']:.1f} tok/s (decode only {summ['decode_tok_per_s']:.1f}), step median {summ['ms_step_median']:.3f} ms (gpu part {summ['ms_gpu_median']:.3f}), "
        f"prefill {summ['prefill_ms_mean']:.1f} ms" + (f", gate-only accepts {summ['gate_props']}/{summ['acc_props']}" if "gate_share" in summ else "")
        + (f", speedup vs ar {summ['speedup_vs_ar']:.3f}x (decode only {summ['decode_speedup_vs_ar']:.3f}x)" if "speedup_vs_ar" in summ else ""))
    save()

# ---------------------------------------------------------------- audit: teacher-forced full-depth argmax on each decoder's own text
if a.audit:
    tA = time.time()
    by_id = {d["doc_id"]: d for d in docs}
    for key, recs in OUT["modes"].items():
        nd = nt = 0; per = []
        for rec in recs:
            g = rec["gen"][: rec["tokens_eff"]]; P0 = rec["prompt_tokens"]
            if not g: per.append([rec["doc_id"], 0, 0, None, []]); continue
            ids = by_id[rec["doc_id"]]["ids"] + g
            fd, mg = LT.audit_full_depth(eng, ids, P0, len(g), NV)
            diffs = [i for i in range(len(g)) if fd[i] != g[i]]
            nd += len(diffs); nt += len(g)
            per.append([rec["doc_id"], len(g), len(diffs), diffs[0] if diffs else None, [round(mg[i], 4) for i in diffs[:64]], diffs[:512]])
        OUT["audit"][key] = dict(tokens=nt, diff=nd, share=nd / max(1, nt), per_doc=per)
        log(f"AUDIT {key}: {nd}/{nt} committed tokens differ from the full-depth argmax = {nd / max(1, nt):.4f}")
        save()
    log(f"audit done in {time.time() - tA:.0f}s")
log("DONE"); OUT["meta"]["done"] = True; save()
