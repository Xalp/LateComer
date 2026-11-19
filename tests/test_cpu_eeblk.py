"""test_cpu_eeblk.py -- CPU identity tests for the TRUNCATED-T dense lattice (eeblk's T-generalized latek.py).

For a tiny random A2DOuro with total_ut_steps = T in {2, 3}: the K = T LateKLattice (dense, no background) must commit
token-identical sequences (and per-step logs) to the exact sparse/dense block decoder, and token-identical sequences to
the sequential AR (b0) decoder, in float64 (fp32 reported, near-ties allowed). This is the K==T corner the EE+block
runs use; the T=4 regression is the untouched test_cpu_latek.py suite on the same patched files.

  PYTHONNOUSERSITE=1 python tests/test_cpu_eeblk.py
"""
import json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "decode")); sys.path.insert(0, os.path.join(HERE, "..", "train"))
import engine as E
import sparse as SP
import latek as LK
from dllm.pipelines.a2d import A2DOuroConfig, A2DOuroLMHeadModel

T0 = time.time()
def P(*x): print(f"[{time.time() - T0:5.0f}s]", *x, flush=True)

V = 24; MASK, EOS, IM = V - 1, V - 2, V - 3; NV = V - 3
PROMPTS = [[3, 5, 7, 11, 13, 17, 19, 2, 4, 6], [2, 4, 8, 16, 1, 9, 6, 5, 6, 10, 12], [10, 11, 7, 7, 1, 3, 3, 9, 14, 15],
           [12, 0, 14, 2, 18, 19, 1, 1, 5, 7, 2, 3], [9, 9, 15, 16, 17, 3, 18, 0, 0, 4]]


def tiny(seed, head_scale, gate_scale, T, dtype):
    torch.manual_seed(seed)
    cfg = A2DOuroConfig(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=4, total_ut_steps=T, max_position_embeddings=1024, use_cache=True,
                        tie_word_embeddings=False, rope_theta=10000.0)
    cfg.model_type = "a2d-ouro"; cfg._attn_implementation = "eager"; m = A2DOuroLMHeadModel(cfg).eval()
    with torch.no_grad():
        m.lm_head.weight.mul_(head_scale); m.lm_head.weight[NV:].zero_(); m.model.early_exit_gate.weight.mul_(gate_scale)
        for n, p in m.named_parameters():
            if "layernorm" in n or n.endswith("norm.weight"): p.add_(0.3 * torch.randn_like(p))
    return m.to(dtype)


N = int(os.environ.get("N", 40)); fails = []; cases = []
for dtype in (torch.float64, torch.float32):
    dn = str(dtype).split(".")[-1]
    for T in (2, 3):
        for mname, seed, hs, gs in [("s0 head x6", 0, 6.0, 1.0), ("s2 head x10 gate x8", 2, 10.0, 8.0)]:
            m = tiny(seed, hs, gs, T, dtype)
            Wd, cfg = E.weights_from_hf(m)
            eng = LK.LateEngine(Wd, cfg, "cpu", dtype, S=512, fused=False, log=lambda *x: None, M=8, Bmax=16)
            assert eng.c.T == T
            ar = SP.make_decoder(eng, "ar", NV, [EOS, IM], MASK, attn="torch", graphs=False)
            for shape in ("latt:2", "latt:4"):
                B, sets = LK.parse_shape(shape)
                dec = LK.LateKLattice(eng, T, B, sets, NV, [EOS, IM], MASK, attn="torch", graphs=False)
                exd = SP.make_decoder(eng, shape, NV, [EOS, IM], MASK, attn="torch", graphs=False)
                def trim(g):                                   # effective sequence: cut at the first stop id, cap at N
                    for i, t in enumerate(g):
                        if t in (EOS, IM): return g[: i + 1][:N]
                    return g[:N]

                for pi, prompt in enumerate(PROMPTS):
                    r = dec.decode(prompt, N); rx = exd.decode(prompt, N); ra = ar.decode(prompt, N)
                    e_ex = r["gen_raw"] == rx["gen_raw"] and r["log"] == rx["log"]
                    e_ar = trim(r["gen_raw"]) == trim(ra["gen_raw"])
                    cases.append(dict(dtype=dn, T=T, model=mname, shape=shape, prompt=pi, n=len(r["gen_raw"]),
                                      vs_exact=e_ex, vs_ar=e_ar))
                    if not e_ex:
                        fails.append(f"{dn} T={T} {mname} lk:{T}:{shape} p{pi}: != exact block (log {r['log'][:8]} vs {rx['log'][:8]})")
                    if not e_ar and dtype == torch.float64:
                        fd = next((i for i in range(min(len(r["gen_raw"]), len(ra["gen_raw"]))) if r["gen_raw"][i] != ra["gen_raw"][i]), None)
                        fails.append(f"{dn} T={T} {mname} lk:{T}:{shape} p{pi}: != AR (first diff {fd})")
        P(f"{dn} T={T}: done ({sum(1 for c in cases if c['dtype'] == dn and c['T'] == T)} cases)")

ok = not fails
nar32 = sum(1 for c in cases if c["dtype"] == "float32" and not c["vs_ar"])
P(f"cases {len(cases)}, fails {len(fails)}, fp32 vs-AR mismatches (allowed near-ties) {nar32}")
for f in fails: P("FAIL", f)
json.dump(dict(cases=cases, fails=fails), open(os.path.join(HERE, "test_cpu_eeblk.json"), "w"), indent=1)
P("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
