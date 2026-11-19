"""test_cpu_core.py -- CPU equivalence of the SGLang-side executors with the standalone late.py (acceptance (a), CPU part).

  PYTHONNOUSERSITE=1 NT=8 python tests/test_cpu_core.py [--n 40] [--quick 1] [--json results/test_cpu_core.json]

Tiny random Ouro (A2DOuro, T=4, 2 layers, sharp heads as late2/test_cpu_late.py), float32 and float64, attention = torch reference.
For every model x prompt x mode (l2:{lkv,old}:am:{b0, latt:2, latt:4, latt:8, sp:5:0.3+1.2+3.4} and ar) x slot mapping (identity /
shuffled base slots):
  (1) tokens (gen_raw) and per-step commit log of harness.Harness (= the server's LateStepExecutor + LateReqState + prefill) equal
      late.LateLattice.decode (ar: engine.Lattice B=0) on the same weights;
  (2) after the decode, the physical K/V the executor holds for every (layer, loop region, position < Lc_last + D) equals late.py's
      kbuf / vbuf entry for that (region, position) -- exact (both run the same op sequence);
  (3) a prompt shorter than D (2 tokens) decodes without NaN/inf in any real row (late.py cannot run it: it asserts P0 >= D).
"""
import argparse, json, os, sys, time
import torch
ap = argparse.ArgumentParser(); ap.add_argument("--json", default=""); ap.add_argument("--n", type=int, default=40)
ap.add_argument("--quick", type=int, default=0); a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("NT", "8"))); T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(REPO_ROOT, "decode")); sys.path.insert(0, os.path.join(REPO_ROOT, "train"))
import engine as E                       # late2 (read-only import)
import late as LT
from dllm.pipelines.a2d import A2DOuroConfig, A2DOuroLMHeadModel
from sglang_late.core_ops import Cfg, OuroCore
from sglang_late.harness import Harness
def P(*x): print(f"[{time.time() - T0:6.0f}s]", *x, flush=True)

V = 24; MASK, EOS, IM = V - 1, V - 2, V - 3; NV = V - 3


def tiny(seed, head_scale, gate_scale=1.0, layers=2, dtype=torch.float64):
    """copy of late2/test_cpu_late.tiny"""
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
SHAPES = ["b0", "latt:2", "latt:4", "latt:8", "sp:5:0.3+1.2+3.4"]
READS = ["lkv", "old"]
PERMS = [None, 7]
if a.quick: MODELS = MODELS[:1]; PROMPTS = PROMPTS[:2]; READS = ["lkv"]
S = 512
OUT = dict(cases=[], fails=[]); nfail = 0; stats = dict(decodes=0, steps=0, kv_checked=0)


def fail(msg):
    global nfail
    nfail += 1; OUT["fails"].append(msg); P("FAIL", msg)


for dtype in (torch.float32, torch.float64):
    dn = str(dtype).split(".")[-1]
    for mname, seed, hs, gs in MODELS:
        m = tiny(seed, hs, gs, dtype=dtype)
        Wd, ecfg = E.weights_from_hf(m)
        eng = LT.LateEngine(Wd, ecfg, "cpu", dtype, S=S, fused=False, log=lambda *x: None, M=8, Bmax=16)
        core = OuroCore(Cfg(m.config.to_dict()), "cpu", dtype, W=Wd, fused=False, log=lambda *x: None)
        modes = ["ar"] + [f"l2:{r}:am:{s}" for s in SHAPES for r in READS]
        for mode in modes:
            if mode == "ar":
                ref = E.Lattice(eng, 0, NV, [EOS, IM], MASK, attn="torch", graphs=False)
            else:
                p = LT.parse_late(mode)
                ref = LT.LateLattice(eng, p["B"], p["sets"], NV, [EOS, IM], MASK, read=p["read"], ver="am", attn="torch", graphs=False)
            for perm in PERMS:
                hz = Harness(core, mode, S, NV, [EOS, IM], MASK, size=S + 64, attn="torch", graphs=False, perm_seed=perm)
                res = []
                for pi, prompt in enumerate(PROMPTS):
                    for kb in eng.kbuf + eng.vbuf: kb.zero_()
                    for kb in hz.kv.kflat + hz.kv.vflat: kb.zero_()
                    lcs = []
                    def hook(h, s, lc, p1, p2, known): lcs.append((lc, p1))
                    r0 = ref.decode(prompt, a.n, prefill_attn="torch")
                    r1 = hz.decode(prompt, a.n, hook=hook)
                    eq = r0["gen_raw"] == r1["gen_raw"] and r0["log"] == r1["log"]
                    stats["decodes"] += 1; stats["steps"] += r1["steps"]
                    if not eq:
                        fd = next((i for i in range(min(len(r0["gen_raw"]), len(r1["gen_raw"]))) if r0["gen_raw"][i] != r1["gen_raw"][i]), None)
                        fail(f"(1) {dn} {mname} {mode} perm={perm} prompt {pi}: first token diff {fd}; log {r0['log'][:10]} vs {r1['log'][:10]}")
                    # (2) physical K/V vs late.py's combined buffers
                    kerr = 0.0; nchk = 0
                    if mode != "ar":
                        D = hz.sh.D; lc_prev = lcs[-2][0] if len(lcs) >= 2 else len(prompt)
                        hi = max(lc_prev + D, len(prompt))
                        for li in range(core.c.NL):
                            for t in range(4):
                                for q in range(hi):
                                    k1, v1 = hz.kv_at(li, t, q)
                                    k0 = eng.kbuf[li][t * eng.SL + q, 0]; v0 = eng.vbuf[li][t * eng.SL + q, 0]
                                    kerr = max(kerr, (k1 - k0).abs().max().item(), (v1 - v0).abs().max().item()); nchk += 1
                    else:
                        lc_end = len(prompt) + len(r1["gen_raw"]) - 1
                        for li in range(core.c.NL):
                            for t in range(4):
                                for q in range(lc_end):
                                    k1, v1 = hz.kv_at(li, t, q); k0 = eng.kc[t][li][q, 0]; v0 = eng.vc[t][li][q, 0]
                                    kerr = max(kerr, (k1 - k0).abs().max().item(), (v1 - v0).abs().max().item()); nchk += 1
                    stats["kv_checked"] += nchk
                    if kerr != 0.0: fail(f"(2) {dn} {mname} {mode} perm={perm} prompt {pi}: physical K/V differ from late.py, max {kerr:.3g}")
                    res.append(dict(prompt=pi, eq=eq, kv_err=kerr, n=len(r1["gen_raw"]), steps=r1["steps"], acc=sum(r1["c"])))
                    OUT["cases"].append(dict(dtype=dn, model=mname, mode=mode, perm=perm, **res[-1]))
                P(f"{dn} {mname} {mode:28s} perm={perm}: " + ", ".join(f"p{x['prompt']} eq={x['eq']} kv={x['kv_err']:.0e} n={x['n']} st={x['steps']} acc={x['acc']}" for x in res))
        # (3) short prompt (2 tokens < D) on the largest block
        for mode in ("l2:lkv:am:latt:8", "l2:lkv:am:latt:4"):
            hz = Harness(core, mode, S, NV, [EOS, IM], MASK, size=S + 64, attn="torch", graphs=False, perm_seed=3)
            r = hz.decode([5, 9], 24)
            fin = all(bool(torch.isfinite(x).all()) for x in (hz.ex.exp, hz.ex.Xb, hz.ex.HB[:-1]))
            OUT["cases"].append(dict(dtype=dn, model=mname, mode=mode, short_prompt=True, n=len(r["gen_raw"]), finite=fin))
            if not fin or len(r["gen_raw"]) == 0: fail(f"(3) {dn} {mname} {mode}: short prompt finite={fin} n={len(r['gen_raw'])}")
            P(f"{dn} {mname} {mode} short prompt [5, 9]: {len(r['gen_raw'])} tokens, {r['steps']} steps, finite={fin}")

OUT["stats"] = stats; OUT["nfail"] = nfail; OUT["ok"] = nfail == 0
P(f"stats {stats}")
P(f"RESULT: {'ALL OK' if OUT['ok'] else 'NOT OK'} (failures {nfail})")
if a.json: json.dump(OUT, open(a.json, "w"), indent=1)
