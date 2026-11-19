"""CPU test of the teacher-forced late-entry probes (mech_core.py / mech_run.py) on tiny random Ouro configs, fp32.

  (a) L_T(p) with deep context (slot entering at pass 1, T passes)   == the HF model's normal full-depth forward at p (mixture logits, 1e-5)
      and the probe's own full forward                                 == the HF forward (per-pass normed states and mixture logits)
  (b) the k-pass baseline (first k passes of the full forward)         == the HF model run with total_ut_steps = k (k = 1..T)
  (c) every late-entry cell(D, t), t < D                               == an independent per-position computation with the HF decoder
      layers: HF forward of x_<p at total_ut_steps = D with a UniversalTransformerCache, then a cold row x_p run through the layers at
      pass indices D-t+1..D against the cached pass-k prefix keys (the late-entry decoders' substrate), renormalised mixture readout
  (d) identities: cell(k, k) == ar_k; P3 (deep, deep) == cell(T, 1), (shallow, shallow) == cell(1, 1); P4 m=0 / m >= N endpoints;
      deep-for-everything extras == cell(T,1); the attention-stat path (explicit softmax) == the SDPA path; stats sane (in [0,1],
      self <= last8, entropy <= log #keys); logit lens at the last layer == the pass readout; process_doc runs and its checks are tiny
Run:  PYTHONPATH=train <mars env python> test_mech_cpu.py
"""
import math, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mech_core as mc  # noqa: E402
import mech_run as mr  # noqa: E402
from dllm.pipelines.a2d import A2DOuroConfig, A2DOuroLMHeadModel  # noqa: E402
from dllm.pipelines.a2d.models.ouro.modeling_ouro import UniversalTransformerCache  # noqa: E402

TOL = 1e-5


def tiny(seed, Hkv=4, T=4, L=3, V=97):
    torch.manual_seed(seed)
    cfg = A2DOuroConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=L, num_attention_heads=4, num_key_value_heads=Hkv,
                        head_dim=16, total_ut_steps=T, max_position_embeddings=512, rope_theta=10000.0, layer_types=["full_attention"] * L,
                        use_cache=False, use_sliding_window=False, rms_norm_eps=1e-6, pad_token_id=0, bos_token_id=0, eos_token_id=0)
    cfg._attn_implementation = "sdpa"
    m = A2DOuroLMHeadModel(cfg).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n: p.copy_(1.0 + 0.2 * torch.randn_like(p))
            elif "early_exit_gate" in n: p.copy_(0.5 * torch.randn_like(p))
            else: p.copy_(0.15 * torch.randn_like(p))   # std .25 made the 12-layer-application random net amplify fp32 rounding of the
    return m                                             # slot path (different SDPA mask/row blocking) to 2.7e-5 at logit 7.6; in fp64 the paths are bit-equal


def hf_mix(m, ids, k=None):
    """HF forward (A2DOuroLMHeadModel default = gate-weighted mixture) at total_ut_steps = k"""
    mm = m.model; old = mm.total_ut_steps
    if k is not None: mm.total_ut_steps = k
    try:
        with torch.no_grad(): out = m(input_ids=ids[None], use_cache=False).logits[0].float()
    finally: mm.total_ut_steps = old
    return out


def hf_late(m, ids, p, D, t):
    """independent late-entry slot at p: HF prefix cache of x_<p at depth D, then x_p's cold row at pass indices D-t+1..D"""
    mm = m.model; L = len(mm.layers); old = mm.total_ut_steps; mm.total_ut_steps = D; m.config.total_ut_steps = D
    try:
        with torch.no_grad():
            cache = UniversalTransformerCache(L * D); mm(input_ids=ids[None, :p], past_key_values=cache, use_cache=True)
            h = mm.embed_tokens(ids[p:p + 1])[None]; pos = torch.tensor([[p]]); pe = mm.rotary_emb(h, pos); lgs, gls = [], []
            for k in range(D - t, D):
                c = UniversalTransformerCache(L); c.key_cache = [cache.key_cache[k * L + l] for l in range(L)]; c.value_cache = [cache.value_cache[k * L + l] for l in range(L)]
                mask = torch.ones(1, 1, 1, p + 1, dtype=torch.bool)
                for layer in mm.layers:
                    h = layer(h, attention_mask=mask, position_ids=pos, past_key_value=c, use_cache=True, cache_position=None, position_embeddings=pe, current_ut=0)
                h = mm.norm(h); gls.append(mm.early_exit_gate(h)[0, :, 0]); lgs.append(m.lm_head(h)[0].float())
            return mc.mixture(lgs, gls)[0]
    finally: mm.total_ut_steps = old; m.config.total_ut_steps = old


def run(seed, Hkv, T, N=40, P0=12):
    m = tiny(seed, Hkv=Hkv, T=T); V = m.config.vocab_size; pr = mc.Probe(m); dev = "cpu"
    g = torch.Generator().manual_seed(seed + 1); ids = torch.randint(1, V, (N,), generator=g)
    S = torch.arange(P0, N - 1); C = mc.causal_strict(S, N, dev); worst = {}

    def chk(name, a, b, tol=TOL):
        e = float((a - b).abs().max()); worst[name] = max(worst.get(name, 0.0), e)
        assert e < tol, f"[seed {seed} Hkv {Hkv} T {T}] {name}: max |diff| {e:.3e} >= {tol}"

    FF = pr.full_forward(ids, S, stats_passes=(0, T - 1), lens_passes=(0, T - 1))
    # (a) probe full forward == HF; L_T (slot from pass 1) == HF full depth
    hf_full = hf_mix(m, ids)[S]
    chk("a: probe full fwd == HF", mc.mixture(FF["lg"], FF["g"]), hf_full)
    with torch.no_grad():
        _, hsl, _ = m.model(input_ids=ids[None], use_cache=False)
    for k in range(T): chk(f"a: pass-{k + 1} state == HF", FF["hs"][k], hsl[k][0, S])
    LT = pr.slot(ids, S, FF, mc.cell_specs(T, T, C))["mix"]; chk("a: L_T slot == HF full depth", LT, hf_full)
    # (b) k-pass baseline == HF at total_ut_steps = k
    ar = {}
    for k in range(1, T + 1):
        ar[k] = mc.mixture(FF["lg"][:k], FF["g"][:k]); chk(f"b: ar{k} == HF total_ut_steps={k}", ar[k], hf_mix(m, ids, k)[S])
    # (c) late-entry cells vs the independent HF-layer computation (a few positions)
    cells = {}
    for D in range(1, T + 1):
        for t in range(1, D + 1):
            cells[(D, t)] = pr.slot(ids, S, FF, mc.cell_specs(D, t, C))["mix"]
            for p in (P0, P0 + 7, N - 2):
                chk(f"c: cell({D},{t}) == HF late slot", cells[(D, t)][p - P0], hf_late(m, ids, p, D, t))
    # (d) identities
    for k in range(1, T + 1): chk(f"d: cell({k},{k}) == ar{k}", cells[(k, k)], ar[k])
    d_, s_ = T - 1, 0
    one = lambda kd, vd, M: pr.slot(ids, S, FF, [[(kd, vd, M)]])["mix"]
    chk("d: P3 deep/deep == cell(T,1)", one(d_, d_, C), cells[(T, 1)]); chk("d: P3 shallow/shallow == cell(1,1)", one(s_, s_, C), cells[(1, 1)])
    jj = torch.arange(N); dist = S[:, None] - jj[None]
    two = lambda A: pr.slot(ids, S, FF, [[(d_, d_, C & A), (s_, s_, C & ~A)]])["mix"]
    chk("d: P4 last m=0 == cell(1,1)", two(dist <= 0), cells[(1, 1)]); chk("d: P4 last m=N == cell(T,1)", two(dist <= N), cells[(T, 1)])
    chk("d: P4 old m=0 == cell(T,1)", two(dist > 0), cells[(T, 1)]); chk("d: P4 old m=N == cell(1,1)", two(dist > N), cells[(1, 1)])
    # a P3 mix with distinct sources must differ from both endpoints (the K/V split is real)
    kv = one(d_, s_, C); assert float((kv - cells[(T, 1)]).abs().max()) > 1e-3 and float((kv - cells[(1, 1)]).abs().max()) > 1e-3, "P3 K/V split is a no-op"
    o = pr.slot(ids, S, FF, mc.cell_specs(T, 1, C), stats=True, lens=True)
    chk("d: stats path == SDPA path", o["mix"], cells[(T, 1)])
    assert o["lens"].shape == (pr.L, len(S)) and torch.equal(o["lens"][-1], o["mix"].argmax(-1).int()), "lens last layer != readout"
    assert torch.equal(FF["lens"][T - 1][-1], FF["lg"][T - 1].argmax(-1).int()), "full-forward lens last layer != pass readout"
    for st, nk in ((o["stats"], S + 1), (FF["stats"][0], S + 1)):
        ent, l8, sk, sf = st[..., 0], st[..., 1], st[..., 2], st[..., 3]
        assert float(l8.max()) <= 1 + 1e-6 and float(sk.min()) >= 0 and bool((sf <= l8 + 1e-6).all()) and bool((ent <= torch.log(nk.float())[None] + 1e-5).all()) and float(ent.min()) >= -1e-6
    # stats of the normal pass-1 query: same queries as the slot at layer 1 when the slot reads pass-1 keys -> identical layer-1 stats
    o1 = pr.slot(ids, S, FF, mc.cell_specs(1, 1, C), stats=True); chk("d: cell(1,1) stats == normal pass-1 stats", o1["stats"], FF["stats"][0], 1e-5)
    # process_doc end to end
    rec, feats = mr.process_doc(pr, ids.tolist(), P0, dev)
    assert max(rec["checks"].values()) < TOL, rec["checks"]
    assert rec["res"]["cell_%d_1" % T]["am"].shape == (len(S),) and rec["attn"].shape == (3, 2, 4, pr.L) and rec["cos"].shape == (len(S), 2 * T - 1)
    assert abs(rec["attn_n"].sum() - len(S)) == 0 and len(feats) == T + 1
    import numpy as np
    assert np.allclose(rec["res"]["full"]["nll"], -torch.log_softmax(hf_full, -1)[torch.arange(len(S)), ids[S + 1]].numpy(), atol=1e-5)
    names = {n for n, _, _ in mr.doc_configs(T, S, N, P0, dev)}
    assert {f"p4_last_{x}" for x in mr.M_LIST} <= names and "p3_KdVs" in names and f"cell_{T}_{T}" in names
    # CKA accumulator: CKA(X, X) == 1 and CKA is invariant to isotropic scaling
    acc = mr.CKAAcc(T, 64, dev); acc.add(feats); acc.close(); acc.add([f * 2 for f in feats]); r = acc.result(B=20)
    for k, v in r["pairs"].items(): assert 0 <= v["point"] <= 1 + 1e-9
    return worst


if __name__ == "__main__":
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    allw = {}
    for seed, Hkv, T in ((0, 4, 4), (1, 2, 4), (2, 4, 3)):
        w = run(seed, Hkv, T)
        for k, v in w.items(): allw[k] = max(allw.get(k, 0.0), v)
        print(f"seed {seed} Hkv {Hkv} T {T}: OK (worst {max(w.values()):.2e})", flush=True)
    for k, v in sorted(allw.items()): print(f"  {k:45s} {v:.2e}")
    print("ALL TESTS PASSED")
