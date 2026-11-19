"""ref_late.py -- INDEPENDENT reference of the two-pass late lattice (CPU tests only). Shares no code with late.py / engine.py:
the forward is written from the HF Ouro modules (input_layernorm, q/k/v/o proj, HF rotary + apply_rotary_pos_emb, input_layernorm_2,
post_attention_layernorm(_2), mlp, model.norm, early_exit_gate, lm_head), attention is an explicit softmax over an explicit per-row key list,
and context depth is derived from "the step at whose end each position joined the context" (no group/pointer bookkeeping).

Per step s (1-based) and call kk in {0, 1} (block pass k = 3 + kk):
  full-depth K/V of the committed text = a plain causal T-pass forward over known[:Lc] recomputed from scratch (what the background must equal)
  d(q) = 4 for prompt positions, else min(4, 2*(s - joined[q] - 1) + kk + 1)
  read "old": K/V of q at pass min(k, d(q)) from the full-depth forward;  read "lkv": pass k from the full-depth forward if d(q) >= k, else the
  late K/V that q's token slot produced at pass k in the step that committed it.
Block rows, lattice mask, readout (lam3, 1 - lam3), verification and next-window rule as described in late.py's docstring.
B = 0 (sets = []) is the plain late-entry t=2 decoder with the same context-reading rule: one token slot per step.
"""
import torch
from importlib import import_module


class RefLate:
    def __init__(self, m, nv, eos, mask_id):
        self.m, self.nv, self.eos, self.mask_id = m, nv, set(eos), mask_id
        mm = m.model; self.layers = list(mm.layers); self.norm = mm.norm; self.rot = mm.rotary_emb; self.gate_l = mm.early_exit_gate
        self.E = mm.embed_tokens.weight; self.head = m.lm_head; self.T = mm.config.total_ut_steps
        mod = import_module(type(self.layers[0]).__module__); self.apply_rope = mod.apply_rotary_pos_emb
        a0 = self.layers[0].self_attn; self.NH = a0.config.num_attention_heads; self.HD = a0.head_dim

    # ---- one pass over rows; keys[i] = list of ('ext', store_key) or ('row', j); store[(pos, pass)] = [(K, V) per layer] ----
    @torch.no_grad()
    def pass_rows(self, X, pos, keys, store):
        n = X.shape[0]; NH, HD = self.NH, self.HD
        cos, sin = self.rot(X[None], torch.tensor([pos]))
        kv_out = [[None] * len(self.layers) for _ in range(n)]
        h = X[None]
        for li, L in enumerate(self.layers):
            res = h; x = L.input_layernorm(h); at = L.self_attn
            q = at.q_proj(x).view(1, n, NH, HD).transpose(1, 2); k = at.k_proj(x).view(1, n, NH, HD).transpose(1, 2)
            v = at.v_proj(x).view(1, n, NH, HD).transpose(1, 2)
            q, k = self.apply_rope(q, k, cos, sin)
            outs = []
            for i in range(n):
                kv_out[i][li] = (k[0, :, i], v[0, :, i])
                Ks, Vs = [], []
                for kind, ref in keys[i]:
                    if kind == "ext": kk_, vv_ = store[ref][li]
                    else: kk_, vv_ = k[0, :, ref], v[0, :, ref]
                    Ks.append(kk_); Vs.append(vv_)
                Kt = torch.stack(Ks, 1); Vt = torch.stack(Vs, 1)                        # [NH, nk, HD]
                sc = torch.einsum("hd,hkd->hk", q[0, :, i], Kt) * at.scaling
                outs.append(torch.einsum("hk,hkd->hd", torch.softmax(sc, -1), Vt))
            o = torch.stack(outs, 0).reshape(1, n, NH * HD)
            h = res + L.input_layernorm_2(at.o_proj(o))
            res = h; h = res + L.post_attention_layernorm_2(L.mlp(L.post_attention_layernorm(h)))
        hn = self.norm(h)[0]
        return hn, kv_out

    def readout_parts(self, hn):
        return torch.sigmoid(self.gate_l(hn))[:, 0], self.head(hn)[:, : self.nv]

    @torch.no_grad()
    def full_depth(self, ids):
        """causal T-pass forward of ids: store {(q, pass): [(K, V)]}, and the gate-mixture prediction of the last position"""
        n = len(ids); X = self.E[torch.tensor(ids)]; store = {}; pos = list(range(n))
        rem = torch.ones(1, dtype=X.dtype); exp = None
        for t in range(self.T):
            keys = [[("row", j) for j in range(i + 1)] for i in range(n)]
            X, kv = self.pass_rows(X, pos, keys, store)
            for i in range(n): store[(i, t + 1)] = kv[i]
            lam, lg = self.readout_parts(X[-1:])
            pt = lam * rem if t < self.T - 1 else rem; rem = rem * (1 - lam)
            exp = lg.float() * pt[:, None] if exp is None else exp + lg.float() * pt[:, None]
        return store, int(exp[0].argmax())

    @torch.no_grad()
    def decode(self, prompt, max_new, B, sets, read="old", ver="am", theta=None):
        P0 = len(prompt); D = B + 1
        offs = []; o = D
        for _, s in sets: offs.append(o); o += s
        NB = o
        _, first = self.full_depth(list(prompt))
        known = list(prompt) + [first]; joined = {}; late = {}; Lc = P0; props, pprop = [], []; log = []; done = first in self.eos; s = 0
        while not done and len(known) - P0 < max_new:
            s += 1
            fd, _ = self.full_depth(known[:Lc])
            # block rows: (kind, pos, input id, visible block rows)
            rows = [("tok", Lc + j, ([known[-1]] + props + [self.mask_id] * B)[j], list(range(j + 1))) for j in range(D)]
            for k_, (a, sz) in enumerate(sets):
                for i in range(1, sz + 1):
                    rows.append(("ph", Lc + a + i, self.mask_id, list(range(a + 1)) + [offs[k_] + ii for ii in range(i)]))
            assert len(rows) == NB
            X = self.E[torch.tensor([r[2] for r in rows])]; pos = [r[1] for r in rows]
            rem = torch.ones(NB, dtype=torch.float32); exp = None; kvs = []
            for kk in range(2):
                kp = 3 + kk; ctx = []
                for q in range(Lc):
                    d = 4 if q < P0 else min(4, 2 * (s - joined[q] - 1) + kk + 1)
                    if read == "old": ctx.append(("ext", ("fd", q, min(kp, d))))
                    else: ctx.append(("ext", ("fd", q, kp)) if d >= kp else ("ext", ("late", q, kp)))
                store = {("fd", q, t): fd[(q, t)] for q in range(Lc) for t in range(1, 5)}
                store.update({("late", q, t): v for (q, t), v in late.items()})
                keys = [ctx + [("row", j) for j in r[3]] for r in rows]
                X, kv = self.pass_rows(X, pos, keys, store); kvs.append(kv)
                lam, lg = self.readout_parts(X)
                if kk == 0: pt = lam.float() * rem; rem = rem * (1 - lam.float()); exp = lg.float() * pt[:, None]
                else: exp = exp + lg.float() * rem[:, None]
            pr = torch.softmax(exp, -1); top = exp.argmax(-1).tolist(); pmax = pr.max(-1).values.tolist()
            c = 0
            while c < len(props):
                am = top[c] == props[c]
                if ver == "am": ok = am
                else:
                    g = pprop[c] * float(pr[c, props[c]]) >= theta
                    ok = g if ver == "g" else (am or g)
                if not ok: break
                c += 1
            commit = props[:c] + [top[c]]
            for j in range(c + 1):                                                      # token slots 0..c join the context
                joined[Lc + j] = s; late[(Lc + j, 3)] = kvs[0][j]; late[(Lc + j, 4)] = kvs[1][j]
            if sets:
                a = max(x for x, _ in sets if x <= c); k_ = [x for x, _ in sets].index(a); off, sz = offs[k_], sets[k_][1]
                i0 = c - a; n = max(0, min(B, sz - i0)); props = [top[off + i0 + j - 1] for j in range(1, n + 1)]; pprop = pmax[off + i0 : off + i0 + n]
            Lc += c + 1
            for tk in commit:
                known.append(tk)
                if tk in self.eos: done = True; break
            log.append(len(commit))
        return known[P0:], log
