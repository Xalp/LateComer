"""CPU test of the entry-signal probe (entry_core.py / entry_run.py) on tiny random Ouro configs, fp32 (TF32 irrelevant on CPU).

  (a) forward_lean == mech_core full_forward (per-pass normed states, gates, readouts at the rows; the kept pass's K/V) == HF forward
  (b) ORA (entry = true pass-T input) through the late slot: its pass-T logits == HF's pass-T readout (exit_at_step = T-1) and its output
      state == HF's pass-T state at every slot position; substituting it into the gate mixture gives HF's FULL-DEPTH mixture logits (1e-5)
  (c) E (entry = embedding) == mech_core cell(T, 1) (the existing ideal-late-entry probe) == test_mech_cpu.hf_late (independent HF-layer
      late slot on a UniversalTransformerCache) at several positions
  (d) batching / chunking invariance of slot_from (several entries in one call, chunk 7 vs 4096); rows never see each other
  (e) soft embeddings: one-hot predecessor -> SE == E, OSE (rescaled) == OE; SE10 == SE when the support is <= 10; OSE == lm_head^T p
  (f) ridge: lambda -> 0 equals least squares (with intercept); sufficient-statistics MSE == direct MSE; merged accumulators == one pass;
      chosen lambda minimises held-out MSE
  (g) TokMean table == direct per-token means; entries() shapes / rescale norms (||NE|| = ||OE|| = c); TM falls back to LE for unseen
  (h) eval_doc end to end (all 20 entries): ORA agrees with the pass-T readout everywhere (am, NLL), cos_out(ORA) = 1, checks tiny,
      E check vs mech cell(T, 1) tiny; fit_phase end to end on random docs runs and returns finite R2
Run:  PYTHONPATH=train <python with torch + transformers> test_entry_cpu.py
"""
import os, sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import entry_core as ec  # noqa: E402
import entry_run as er  # noqa: E402
import mech_core as mc  # noqa: E402
import test_mech_cpu as tm  # noqa: E402   (tiny(), hf_mix(), hf_late())

TOL = 1e-5


def hf_states(m, ids):
    with torch.no_grad():
        _, hsl, gl = m.model(input_ids=ids[None], use_cache=False)
    return [h[0] for h in hsl], [g[0, :, 0] for g in gl]


def hf_pass_logits(m, ids, k):
    with torch.no_grad():
        return m(input_ids=ids[None], use_cache=False, exit_at_step=k).logits[0].float()


def run(seed, Hkv, T, N=40, P0=12):
    torch.set_grad_enabled(False)
    m = tm.tiny(seed, Hkv=Hkv, T=T); V = m.config.vocab_size; pr = ec.EntryProbe(m); dev = "cpu"
    g = torch.Generator().manual_seed(seed + 1); ids = torch.randint(1, V, (N,), generator=g)
    Sx = torch.arange(P0 - 1, N - 1); S = Sx[1:]; x = ids[S]; tgt = ids[S + 1]; nq = S.shape[0]; worst = {}

    def chk(name, a, b, tol=TOL):
        e = float((a - b).abs().max()); worst[name] = max(worst.get(name, 0.0), e)
        assert e < tol, f"[seed {seed} Hkv {Hkv} T {T}] {name}: max |diff| {e:.3e} >= {tol}"

    # (a) lean forward == mech full_forward == HF
    FF = pr.forward_lean(ids, Sx, kv_pass=T - 1); FM = pr.full_forward(ids, Sx)
    for k in range(T):
        chk("a: lean hs == mech hs", FF["hs"][k], FM["hs"][k]); chk("a: lean g == mech g", FF["g"][k], FM["g"][k]); chk("a: lean lg == mech lg", FF["lg"][k], FM["lg"][k])
    for l in range(pr.L): chk("a: lean K/V == mech K/V", torch.cat([FF["K"][l], FF["V"][l]]), torch.cat([FM["K"][T - 1][l], FM["V"][T - 1][l]]))
    hsl, gl = hf_states(m, ids)
    for k in range(T): chk("a: lean hs == HF", FF["hs"][k], hsl[k][Sx])
    hf_full = tm.hf_mix(m, ids)
    chk("a: lean mixture == HF full depth", mc.mixture(FF["lg"], FF["g"]), hf_full[Sx])
    # (b) ORA
    Y = FF["hs"][T - 2][1:]
    o = pr.slot_from(Y, S, FF, tgt=tgt, keep_logits=True)
    hfT = hf_pass_logits(m, ids, T - 1)
    chk("b: ORA logits == HF pass-T readout", o["logits"], hfT[S]); chk("b: ORA state == HF pass-T state", o["h"], hsl[T - 1][S])
    gO = m.model.early_exit_gate(o["h"])[:, 0].float()
    sub = mc.mixture([lg[1:] for lg in FF["lg"][:-1]] + [o["logits"]], [gg[1:] for gg in FF["g"][:-1]] + [gO])
    chk("b: ORA substituted into the mixture == HF full-depth logits", sub, hf_full[S])
    # (c) E == mech cell(T, 1) == independent HF late slot
    E = m.model.embed_tokens(x)
    oE = pr.slot_from(E, S, FF, keep_logits=True)
    FFm = dict(K={T - 1: FF["K"]}, V={T - 1: FF["V"]}, N=N)
    cm = pr.slot(ids, S, FFm, mc.cell_specs(T, 1, mc.causal_strict(S, N, dev)))["mix"]
    chk("c: E == mech cell(T,1)", oE["logits"], cm)
    cm2 = pr.slot(ids, S, pr.full_forward(ids, S), mc.cell_specs(T, 1, mc.causal_strict(S, N, dev)))["mix"]
    chk("c: E == mech cell(T,1) on its own full_forward", oE["logits"], cm2)
    for p in (P0, P0 + 9, N - 2):
        chk("c: E == HF late slot", oE["logits"][p - P0], tm.hf_late(m, ids, p, T, 1))
    # (d) batching / chunking invariance
    Z = torch.cat([E, Y, 3 * E + 0.5]); pos = S.repeat(3)
    a1 = pr.slot_from(Z, pos, FF, keep_logits=True, chunk=7); a2 = pr.slot_from(Z, pos, FF, keep_logits=True, chunk=4096)
    chk("d: chunk 7 == chunk 4096", a1["logits"], a2["logits"]); chk("d: batched E == alone", a2["logits"][:nq], oE["logits"])
    chk("d: batched ORA == alone", a2["logits"][nq:2 * nq], o["logits"])
    # (e) soft embeddings
    f = ec.features(pr, x, FF)
    chk("e: features E == embedding", f["E"], E); chk("e: features Y == pass T-1 state", f["Y"], Y); chk("e: features H == predecessor pass-T", f["H"], hsl[T - 1][S - 1])
    P = torch.softmax(mc.mixture(FF["lg"], FF["g"])[:-1], -1)
    chk("e: SE == P @ E", f["SE"], P @ m.model.embed_tokens.weight[: pr.nv]); chk("e: OSE == lm_head^T p", f["OSE"], P @ m.lm_head.weight[: pr.nv])
    FF1 = dict(FF); big = torch.full_like(FF["lg"][T - 1], -1e4); big[torch.arange(len(Sx)), torch.cat([x, x[-1:]])] = 1e4   # row i (predecessor of slot i) one-hot on x_q
    FF1["lg"] = [big] * T
    f1 = ec.features(pr, x, FF1)
    chk("e: one-hot SE == E", f1["SE"], f1["E"]); chk("e: one-hot SE10 == E", f1["SEk"], f1["E"])
    chk("e: one-hot OSE rescaled == OE", ec.rescale(f1["OSE"], 3.0), ec.rescale(f1["O"], 3.0))
    assert bool(f1["pred_ok"].all()) and float(f1["pxq"].min()) > 0.999
    lgk = torch.full_like(FF["lg"][T - 1], -1e4); lgk[:, :5] = torch.randn(len(Sx), 5); FF2 = dict(FF); FF2["lg"] = [lgk] * T
    f2 = ec.features(pr, x, FF2); chk("e: SE10 == SE when support <= 10", f2["SEk"], f2["SE"])
    # (f) ridge
    gen = torch.Generator().manual_seed(seed + 7); n, D, d = 400, 12, 5
    X = torch.randn(n, D, generator=gen, dtype=torch.float64) * torch.linspace(0.1, 5, D, dtype=torch.float64) + 2
    Wt = torch.randn(D, d, generator=gen, dtype=torch.float64); Yr = X @ Wt + 1.5 + 0.3 * torch.randn(n, d, generator=gen, dtype=torch.float64)
    acc = ec.RidgeAcc(D, d, dev); acc.add(X[:300], Yr[:300]); va = ec.RidgeAcc(D, d, dev); va.add(X[300:], Yr[300:])
    W0, b0 = ec.ridge_solve(acc, 1e-12)
    Xa = torch.cat([X[:300], torch.ones(300, 1, dtype=torch.float64)], 1); sol = torch.linalg.lstsq(Xa, Yr[:300]).solution
    chk("f: ridge lam->0 == lstsq W", W0, sol[:-1], 1e-8); chk("f: ridge lam->0 == lstsq b", b0, sol[-1], 1e-8)
    W1, b1 = ec.ridge_solve(acc, 0.3); mse, var = ec.ridge_mse(va, W1, b1)
    dmse = float(((Yr[300:] - X[300:] @ W1 - b1) ** 2).sum(-1).mean()); dvar = float(((Yr[300:] - Yr[300:].mean(0)) ** 2).sum(-1).mean())
    assert abs(mse - dmse) < 1e-8 * max(1, dmse) and abs(var - dvar) < 1e-8 * max(1, dvar), (mse, dmse, var, dvar)
    one = ec.RidgeAcc(D, d, dev); one.add(X, Yr); mg = acc.merged(va)
    for k in ("sx", "sy", "Sxx", "Sxy", "syy"): chk(f"f: merged {k}", getattr(mg, k), getattr(one, k), 1e-8)
    r = ec.ridge_fit(acc, va); assert str(r["lam"]) == min(r["val"], key=lambda k: r["val"][k][0]) and np.isfinite(r["val_r2"])
    # (g) TokMean + entries
    tmc = ec.TokMean(pr.nv, m.config.hidden_size, dev); xs = torch.tensor([3, 5, 3, 7, 3]); Ys = torch.randn(5, m.config.hidden_size); tmc.add(xs, Ys)
    tab, seen = tmc.table(); chk("g: TokMean", tab[3], Ys[[0, 2, 4]].mean(0)); assert bool(seen[[3, 5, 7]].all()) and int(seen.sum()) == 3
    hd = m.config.hidden_size; c = 7.5
    maps = {"LE": (torch.randn(hd, hd) * .1, torch.randn(hd)), "LSE": (torch.randn(hd, hd) * .1, torch.randn(hd)),
            "LOE": (torch.randn(hd, hd) * .1, torch.randn(hd)), "LEH": (torch.randn(2 * hd, hd) * .1, torch.randn(hd))}
    tab2 = torch.randn(pr.nv, hd); seen2 = torch.zeros(pr.nv, dtype=torch.bool); seen2[x[: nq // 2]] = True
    fit = dict(c=c, maps=maps, tm=tab2, tm_seen=seen2, mu=torch.randn(hd))
    Zd = ec.entries(pr, f, x, fit)
    assert list(Zd) == [n for n, _ in ec.ENTRIES] and all(v.shape == (nq, hd) for v in Zd.values())
    for n in ("NE", "OE", "OSE", "OSE10", "NSE"): chk(f"g: ||{n}|| == c", Zd[n].norm(dim=-1), torch.full((nq,), c), 1e-4)
    chk("g: NE = rescaled self.norm(E)", Zd["NE"], ec.rescale(m.model.norm(E), c)); chk("g: EH", Zd["EH"], E + f["H"]); chk("g: ORA", Zd["ORA"], Y)
    chk("g: LEH", Zd["LEH"], torch.cat([E, f["H"]], -1) @ maps["LEH"][0] + maps["LEH"][1])
    exp_tm = torch.where(seen2[x][:, None], tab2[x], E @ maps["LE"][0] + maps["LE"][1]); chk("g: TM with LE fallback", Zd["TM"], exp_tm)
    # (h) end to end
    cosv = ec.centered_cos(m.model.embed_tokens.weight[: pr.nv], m.lm_head.weight[: pr.nv])
    rec = er.eval_doc(pr, ids.tolist(), P0, fit, cosv, chunk=13, check_cell=True)
    assert set(rec["ent"]) == set(er.NAMES) and rec["nq"] == nq
    assert (rec["ent"]["ORA"]["am"] == rec["ref"]["p4"]["am"]).all(), "ORA argmax != pass-T readout"
    assert np.allclose(rec["ent"]["ORA"]["nll"], rec["ref"]["p4"]["nll"], atol=1e-5) and np.allclose(rec["ent"]["ORA"]["cos_out"], 1, atol=1e-5)
    assert max(rec["checks"].values()) < TOL, rec["checks"]
    assert np.allclose(rec["ent"]["E"]["nll"], -torch.log_softmax(oE["logits"], -1)[torch.arange(nq), tgt].numpy(), atol=1e-5)
    assert np.allclose(rec["ref"]["full"]["nll"], -torch.log_softmax(hf_full[S], -1)[torch.arange(nq), tgt].numpy(), atol=1e-5)
    for k in range(1, T): assert (rec["ref"][f"ar{k}"]["am"] == tm.hf_mix(m, ids, k)[S].argmax(-1).numpy()).all()
    for n in er.NAMES: assert np.isfinite(rec["ent"][n]["nll"]).all() and np.isfinite(rec["ent"][n]["cos_out"]).all(), n
    # fit_phase on a few random 'docs' (val doc ids >= 100)
    docs = []
    for j in range(6):
        gj = torch.Generator().manual_seed(100 + j); L = int(torch.randint(20, 36, (1,), generator=gj))
        seq = torch.randint(1, V, (L,), generator=gj).tolist(); docs.append(dict(doc_id=95 + j, ctx=seq[:8], gen=seq[8:], rt_ok=True, finish="eos"))
    fit2, info = er.fit_phase(pr, docs, 100, log=lambda s: None)
    assert all(np.isfinite(info[k]["val_r2"]) for k in ec.MAPS) and info["scale_c"] > 0 and np.isfinite(info["TM"]["val_r2"])
    rec2 = er.eval_doc(pr, ids.tolist(), P0, fit2, cosv, chunk=4096)
    assert max(rec2["checks"].values()) < TOL
    return worst


if __name__ == "__main__":
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    allw = {}
    for seed, Hkv, T in ((0, 4, 4), (1, 2, 4), (2, 4, 3)):
        w = run(seed, Hkv, T)
        for k, v in w.items(): allw[k] = max(allw.get(k, 0.0), v)
        print(f"seed {seed} Hkv {Hkv} T {T}: OK (worst {max(w.values()):.2e})", flush=True)
    for k, v in sorted(allw.items()): print(f"  {k:60s} {v:.2e}")
    print("ALL TESTS PASSED")
