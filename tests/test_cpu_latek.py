"""test_cpu_latek.py -- CPU tests of the K-pass late lattice (latek.py) on tiny random Ouro (A2DOuro), fp32 and float64.

  PYTHONNOUSERSITE=1 NT=8 python test_cpu_latek.py [--n 40] [--json results/test_cpu_latek.json]

(1) K = 4 == the EXACT dense / sparse lattice (engine.Lattice latt:B, sparse.SparseLattice sp:..., engine.Lattice B=0 = full-depth AR for
    b0): tokens and per-step log; also with the (redundant) background forced on (bg=True).
(2) K = 2 == ../late2's late.py (l2:lkv:am:<shape>): tokens and per-step log.
(3) every K in {1,2,3,4} == the independent reference ref_late_k.RefLateK (HF-module forward, explicit per-row key lists, context depth
    derived from the step at which each position joined): tokens and per-step log.
(4) READ INSTRUMENTATION (every call of every decode of (3)): background groups = exactly the positions joined g steps ago, at pass
    K*(g-1)+kk+1, pads -> dummy; a background row reads positions <= its own exactly once at its own pass, and each of them has completed
    that pass; every block query sees every context position exactly once through region k-1 (lateKV rule), its window = its lattice mates
    at pass k; writes go to the row's own pass region.
(5) K/V BOOKKEEPING after every step: every context position's K/V at every pass its background has completed equals the full-depth K/V of
    the committed text (K = 4 without background: all 4 passes of every context position, i.e. the token slots' own K/V ARE full depth).
Shapes b0, latt:2, latt:4, sp:5:0.3+1.2+3.4; 3 tiny models (one with a sharp gate) x 5 prompts x 2 dtypes.
"""
import argparse, json, os, sys, time
import torch
ap = argparse.ArgumentParser(); ap.add_argument("--json", default=""); ap.add_argument("--n", type=int, default=40)
ap.add_argument("--quick", type=int, default=0); a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("NT", "8"))); T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "decode")); sys.path.insert(0, os.path.join(HERE, "..", "train"))
import engine as E
import sparse as SP
import late as L2
import latek as LK
import ref_late_k as RK
from dllm.pipelines.a2d import A2DOuroConfig, A2DOuroLMHeadModel
def P(*x): print(f"[{time.time() - T0:6.0f}s]", *x, flush=True)

V = 24; MASK, EOS, IM = V - 1, V - 2, V - 3; NV = V - 3


def tiny(seed, head_scale, gate_scale=1.0, layers=2, dtype=torch.float64):
    torch.manual_seed(seed)
    cfg = A2DOuroConfig(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=4,
                        total_ut_steps=4, max_position_embeddings=1024, use_cache=True, tie_word_embeddings=False, rope_theta=10000.0)
    cfg.model_type = "a2d-ouro"; cfg._attn_implementation = "eager"; m = A2DOuroLMHeadModel(cfg).eval()
    with torch.no_grad():
        m.lm_head.weight.mul_(head_scale); m.lm_head.weight[NV:].zero_(); m.model.early_exit_gate.weight.mul_(gate_scale)
        for n, p in m.named_parameters():
            if "layernorm" in n or n.endswith("norm.weight"): p.add_(0.3 * torch.randn_like(p))
    return m.to(dtype)


PROMPTS = [[3, 5, 7, 11, 13, 17, 19, 2, 4, 6], [2, 4, 8, 16, 1, 9, 6, 5, 6, 10, 12], [10, 11, 7, 7, 1, 3, 3, 9, 14, 15],
           [12, 0, 14, 2, 18, 19, 1, 1, 5, 7, 2, 3], [9, 9, 15, 16, 17, 3, 18, 0, 0, 4]]
MODELS = [("s0 head x6", 0, 6.0, 1.0), ("s1 head x3", 1, 3.0, 1.0), ("s2 head x10 gate x8", 2, 10.0, 8.0)]
SHAPES = ["b0", "latt:2", "latt:4", "sp:5:0.3+1.2+3.4"]
KS = [1, 2, 3, 4]
if a.quick: MODELS = MODELS[:1]; PROMPTS = PROMPTS[:2]
OUT = dict(cases=[], fails=[], vs_exact=[], vs_late2=[], vs_bg=[]); nfail = 0
stats = dict(calls=0, block_queries=0, bg_queries=0, ctx_reads=0, ctx_reads_late=0, kv_checked=0, steps=0)
cover = {}


def fail(msg):
    global nfail
    nfail += 1; OUT["fails"].append(msg); P("FAIL", msg)


def exact_mode(shape):
    return "ar" if shape == "b0" else shape


def check_reads(dec, rec, lcs, P0):
    S, SL, D, NB, K = dec.S, dec.SL, dec.D, dec.NB, dec.K
    kk, k = rec["kk"], rec["kpass"]; hist = rec["hdr"]; lc = hist[0]; s = len([x for x in lcs if x <= lc])
    assert lcs[s - 1] == lc
    def joined(q): return next(i for i in range(1, len(lcs)) if q < lcs[i])
    def depth(q): return 4 if (q < P0 or not dec.bg) else min(4, K * (s - joined(q) - 1) + kk + 1)
    pt, sl, cu, wslot, pos = rec["pt"], rec["sl"].tolist(), rec["cu"].tolist(), rec["wslot"].tolist(), rec["pos"].tolist()
    nseq = len(sl); ok = True; nA = len(rec["act"])
    def vis(sq, qi):
        Lq = cu[sq + 1] - cu[sq]; return pt[sq, : sl[sq] - Lq + qi + 1].tolist()
    exp_act = [g for g in range(1, dec.NG + 1) if K * (g - 1) + kk + 1 <= 4]
    if rec["act"] != exp_act: fail(f"active groups {rec['act']} vs {exp_act}"); ok = False
    for gi, (g, j) in enumerate(zip(rec["act"], rec["gpass"])):
        grp = [q for q in range(P0, lc) if joined(q) == s - g]
        if grp != list(range(hist[g], hist[g - 1])): fail(f"bookkeeping step {s} group {g}: {grp} vs [{hist[g]},{hist[g-1]})"); ok = False
        for qi in range(D):
            r = gi * D + qi; real = qi >= D - len(grp)
            if not real:
                if wslot[r] != dec.e.DUMMY: fail(f"pad row {r} writes {wslot[r]}"); ok = False
                continue
            p = grp[qi - (D - len(grp))]
            if j != K * (s - joined(p) - 1) + kk + 1: fail(f"bg row pos {p} at pass {j}, expected {K * (s - joined(p) - 1) + kk + 1}"); ok = False
            if pos[r] != p or wslot[r] != (j - 1) * SL + p: fail(f"bg row {r} pos {pos[r]} wslot {wslot[r]} expected {p}/{(j-1)*SL+p}"); ok = False
            v = vis(gi, qi)
            if sorted(v) != [(j - 1) * SL + q for q in range(p + 1)]: fail(f"bg row {r} (pos {p}, pass {j}) reads wrong pages"); ok = False
            for q in range(p):
                if q not in grp and depth(q) < j: fail(f"bg row pos {p} pass {j} reads pos {q} with depth {depth(q)}"); ok = False
            stats["bg_queries"] += 1
    tails = dec.tails
    slot = lambda rr: (lc + rr) if rr < D else (S + rr - D)
    G = rec["G"]
    for sq in range(nA, nseq):
        Lq = cu[sq + 1] - cu[sq]
        for qi in range(Lq):
            r = cu[sq] + qi; rb = r - G; v = vis(sq, qi)
            if wslot[r] != (k - 1) * SL + slot(rb): fail(f"block row {rb} writes {wslot[r]}"); ok = False
            ctx = [x for x in v if (x % SL) < lc and (x % SL) < S]; win = [x for x in v if x not in ctx]
            if sorted(x % SL for x in ctx) != list(range(lc)): fail(f"step {s} pass {k} block row {rb}: context positions not each exactly once"); ok = False
            for x in ctx:
                if x // SL != k - 1: fail(f"step {s} pass {k} row {rb} pos {x % SL} read region {x // SL}, expected {k - 1}"); ok = False
                stats["ctx_reads"] += 1; stats["ctx_reads_late"] += depth(x % SL) < k
            mates = tails[sq - nA][: len(tails[sq - nA]) - Lq + qi + 1]
            if sorted(win) != sorted((k - 1) * SL + slot(m_) for m_ in mates): fail(f"step {s} pass {k} row {rb}: window {win} vs mates {mates}"); ok = False
            stats["block_queries"] += 1
    stats["calls"] += 1
    return ok


def check_kv(dec, ref, known, lc, s, lcs, P0, tol):
    eng = dec.e; SL = dec.SL; fd, _ = ref.full_depth(known[:lc]); err = 0.0; n = 0
    def joined(q): return next(i for i in range(1, len(lcs)) if q < lcs[i])
    for q in range(lc):
        dq = 4 if (q < P0 or not dec.bg) else min(4, dec.K * (s - joined(q)))
        for j in range(1, dq + 1):
            for li in range(eng.c.NL):
                K_, V_ = fd[(q, j)][li]
                ek = eng.kbuf[li][(j - 1) * SL + q, 0]; ev = eng.vbuf[li][(j - 1) * SL + q, 0]
                err = max(err, (ek.double() - K_.double()).abs().max().item(), (ev.double() - V_.double()).abs().max().item()); n += 1
    stats["kv_checked"] += n
    if err > tol: fail(f"KV step {s} K={dec.K} bg={dec.bg}: max err {err:.3g} > {tol}")
    return err


for dtype in (torch.float32, torch.float64):
    dn = str(dtype).split(".")[-1]; tol = 1e-4 if dtype == torch.float32 else 1e-9
    for mname, seed, hs, gs in MODELS:
        m = tiny(seed, hs, gs, dtype=dtype)
        Wd, cfg = E.weights_from_hf(m)
        eng = LK.LateEngine(Wd, cfg, "cpu", dtype, S=512, fused=False, log=lambda *x: None, M=8, Bmax=16)
        ref = RK.RefLateK(m, NV, [EOS, IM], MASK)
        for shape in SHAPES:
            B, sets = LK.parse_shape(shape)
            exd = SP.make_decoder(eng, exact_mode(shape), NV, [EOS, IM], MASK, attn="torch", graphs=False)
            l2d = L2.LateLattice(eng, B, sets, NV, [EOS, IM], MASK, read="lkv", ver="am", attn="torch", graphs=False)
            for K in KS:
                mode = f"lk:{K}:{shape}"
                dec = LK.LateKLattice(eng, K, B, sets, NV, [EOS, IM], MASK, attn="torch", graphs=False, check=True)
                for pi, prompt in enumerate(PROMPTS):
                    dec.reads = []; lcs = [len(prompt)]; kverr = [0.0]
                    def hook(d_, s, hist, known, prompt=prompt):
                        lcs.append(hist[0])
                        if dtype == torch.float64 or s <= 6: kverr[0] = max(kverr[0], check_kv(d_, ref, known, hist[0], s, lcs, len(prompt), tol))
                    r = dec.decode(prompt, a.n, hook=hook)
                    g_ref, l_ref = ref.decode_k(prompt, a.n, K, B, sets)
                    eq = r["gen_raw"] == g_ref and r["log"] == l_ref
                    okr = all(check_reads(dec, rec, lcs, len(prompt)) for rec in dec.reads)
                    stats["steps"] += r["steps"]
                    h = cover.setdefault(mode, [0, 0, 0])
                    for npr, c in zip(r["nprop"], r["c"]):
                        if npr: h[0 if c == 0 else (2 if c >= npr else 1)] += 1
                    OUT["cases"].append(dict(dtype=dn, model=mname, mode=mode, prompt=pi, n=len(r["gen_raw"]), eq_ref=eq, reads_ok=okr, kv_err=kverr[0],
                                             log=r["log"], c=r["c"]))
                    if not eq:
                        fd_ = next((i for i in range(min(len(g_ref), len(r["gen_raw"]))) if g_ref[i] != r["gen_raw"][i]), None)
                        fail(f"(3) {dn} {mname} {mode} prompt {pi}: engine != reference (first diff {fd_}; log {r['log'][:12]} vs {l_ref[:12]})")
                    if K == 4:
                        rx = exd.decode(prompt, a.n)
                        e1 = rx["gen_raw"] == r["gen_raw"] and rx["log"] == r["log"]
                        OUT["vs_exact"].append(dict(dtype=dn, model=mname, shape=shape, prompt=pi, identical=e1))
                        if not e1: fail(f"(1) {dn} {mname} {mode} prompt {pi}: K=4 != exact {exact_mode(shape)} (log {r['log'][:12]} vs {rx['log'][:12]})")
                        db = LK.LateKLattice(eng, 4, B, sets, NV, [EOS, IM], MASK, attn="torch", graphs=False, bg=True)
                        rbg = db.decode(prompt, a.n)
                        e3 = rbg["gen_raw"] == r["gen_raw"] and rbg["log"] == r["log"]
                        OUT["vs_bg"].append(dict(dtype=dn, model=mname, shape=shape, prompt=pi, identical=e3))
                        if not e3: fail(f"(1b) {dn} {mname} {mode} prompt {pi}: K=4 bg=True != K=4")
                    if K == 2:
                        r2 = l2d.decode(prompt, a.n)
                        e2 = r2["gen_raw"] == r["gen_raw"] and r2["log"] == r["log"]
                        OUT["vs_late2"].append(dict(dtype=dn, model=mname, shape=shape, prompt=pi, identical=e2))
                        if not e2: fail(f"(2) {dn} {mname} {mode} prompt {pi}: K=2 != late.py l2:lkv:am:{shape}")
                cs_ = [c for c in OUT["cases"] if c["mode"] == mode and c["dtype"] == dn and c["model"] == mname]
                P(f"{dn} {mname} {mode}: " + ", ".join(f"p{c['prompt']} eq={c['eq_ref']} reads={c['reads_ok']} kv={c['kv_err']:.1e} acc={sum(c['c'])}/{len(c['c'])}st" for c in cs_))
OUT["coverage"] = cover; OUT["stats"] = stats; OUT["nfail"] = nfail
covok = all(h[0] > 0 and h[1] > 0 for k_, h in cover.items() if not k_.endswith("b0"))
summ = {k: f"{sum(x['identical'] for x in OUT[k])}/{len(OUT[k])}" for k in ("vs_exact", "vs_late2", "vs_bg")}
summ["vs_ref"] = f"{sum(c['eq_ref'] for c in OUT['cases'])}/{len(OUT['cases'])}"; OUT["summary"] = summ
OUT["ok"] = nfail == 0 and covok
P(f"identity: {summ}")
P(f"coverage (steps with proposals: none / partial / all accepted): {cover}")
P(f"stats {stats}")
P(f"RESULT: {'ALL OK' if OUT['ok'] else 'NOT OK'} (failures {nfail}; zero+partial acceptance in every B>0 config: {covok})")
if a.json: json.dump(OUT, open(a.json, "w"), indent=1)
