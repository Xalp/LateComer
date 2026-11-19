"""ref_latek.py -- INDEPENDENT teacher-forced reference of the K-pass late lattice (tests only; shares no code with lattice_model.py,
late.py or latek.py).

Forward written from the HF Ouro modules (input_layernorm, q/k/v/o proj, HF rotary + apply_rotary_pos_emb, input_layernorm_2,
post_attention_layernorm(_2), mlp, model.norm, early_exit_gate, lm_head) with an explicit softmax over explicitly gathered keys.
The decoder is replayed STEP BY STEP, as it runs:
  * full-depth K/V of the committed text = the causal T-pass forward of the text, computed once (causal -> prefix-independent; the
    decoder's background equals it, late.py test (3) / latek.py docstring)
  * step s at anchor c: block rows (late.py row order, dense latt:B; token slot j input x_{c+j}, placeholders <|mask|>) enter at pass
    E = T-K+1 and run K passes; at pass k = E+kk a context position q < c is read from the full-depth pass-k K/V iff d(q) >= k with
    d(q) = T (prompt) else min(T, K*(s - joined(q) - 1) + kk + 1)  [the background's progress counted in calls], else from the pass-k
    K/V that q's own token slot produced in the step that committed it (lateKV)
  * after the step, token slots 0..l-1 join the context (l = next anchor - c): their per-pass K/V are stored for later steps
  * readout per row: own gate mixture over passes E..T restarted at entry (lam*rem before T, rem at T), fp32
returns per step: (anchor, l, logits [NB, nv] fp32 in late.py's block row order).
negctl=True: the "simpler approximation" -- every context position read from the full-depth K/V (no lateKV) -- for the negative control.
"""
import torch
from importlib import import_module


class RefLateK:
    def __init__(self, m, nv, mask_id):
        self.m, self.nv, self.mask_id = m, nv, mask_id
        mm = m.model; self.layers = list(mm.layers); self.norm = mm.norm; self.rot = mm.rotary_emb; self.gate_l = mm.early_exit_gate
        self.Emb = mm.embed_tokens.weight; self.head = m.lm_head; self.T = int(mm.config.total_ut_steps)
        self.apply_rope = import_module(type(self.layers[0]).__module__).apply_rotary_pos_emb
        a0 = self.layers[0].self_attn; self.NH = a0.config.num_attention_heads; self.HD = a0.head_dim

    @torch.no_grad()
    def pass_rows(self, X, pos, ext, allow):
        """one pass over rows X [m,H] at positions pos; ext(li) -> (K [c,NH,HD], V) external keys of layer li (already roped) or None;
        allow [m, c + m] bool: row i attends external key j (j < c) / row j-c of this call. returns normed state, [(k, v)] per layer"""
        m = X.shape[0]; NH, HD = self.NH, self.HD
        cos, sin = self.rot(X[None], torch.tensor([pos], device=X.device))
        h = X[None]; kvs = []
        for li, L in enumerate(self.layers):
            res = h; x = L.input_layernorm(h); at = L.self_attn
            q = at.q_proj(x).view(1, m, NH, HD).transpose(1, 2); k = at.k_proj(x).view(1, m, NH, HD).transpose(1, 2)
            v = at.v_proj(x).view(1, m, NH, HD).transpose(1, 2)
            q, k = self.apply_rope(q, k, cos, sin)
            q, k, v = q[0].transpose(0, 1), k[0].transpose(0, 1), v[0].transpose(0, 1)          # [m, NH, HD]
            kvs.append((k, v))
            if ext is not None:
                Ke, Ve = ext(li); Kall = torch.cat([Ke, k], 0); Vall = torch.cat([Ve, v], 0)
            else: Kall, Vall = k, v
            sc = torch.einsum("ihd,jhd->hij", q, Kall) * at.scaling
            sc = sc.masked_fill(~allow[None], float("-inf"))
            o = torch.einsum("hij,jhd->ihd", torch.softmax(sc, -1), Vall).reshape(1, m, NH * HD)
            h = res + L.input_layernorm_2(at.o_proj(o))
            res = h; h = res + L.post_attention_layernorm_2(L.mlp(L.post_attention_layernorm(h)))
        return self.norm(h)[0], kvs

    def readout(self, hs):
        """gate mixture over the given per-pass states (fp32 accumulation; = the decoders)"""
        exp = None; rem = None; npass = len(hs)
        for t, hn in enumerate(hs):
            lam = torch.sigmoid(self.gate_l(hn))[:, 0].float(); lg = self.head(hn)[:, : self.nv].float()
            r = torch.ones_like(lam) if rem is None else rem
            pt = lam * r if t < npass - 1 else r; rem = r * (1 - lam)
            exp = lg * pt[:, None] if exp is None else exp + lg * pt[:, None]
        return exp

    @torch.no_grad()
    def full_depth(self, ids):
        """causal T-pass forward: fd[t][li] = (K [n,NH,HD], V) at pass t (1-based), and the T-pass readout of every position [n, nv]"""
        n = len(ids); X = self.Emb[torch.tensor(ids, device=self.Emb.device)]; fd = {}; hs = []
        causal = torch.tril(torch.ones(n, n, dtype=torch.bool, device=X.device))
        for t in range(1, self.T + 1):
            X, kv = self.pass_rows(X, list(range(n)), None, causal); fd[t] = kv; hs.append(X)
        return fd, self.readout(hs)

    @torch.no_grad()
    def tf_decode(self, text, a, anchors, K, B, negctl=False):
        """teacher-forced K-pass late lattice on `text` (prompt = text[:a]); anchors[0] == a; runs every step whose token slots exist
        (anchor + B < len(text)). returns [(c, l, logits [NB, nv])]"""
        T = self.T; E = T - K + 1; D = B + 1; n = len(text); dev = self.Emb.device
        assert anchors[0] == a
        fd, _ = self.full_depth(list(text))
        rows = [("tok", j, 0, j) for j in range(D)] + [("ph", aa, i, aa + i) for aa in range(D) for i in range(1, B + 1)]
        NB = len(rows)
        wm = torch.zeros(NB, NB, dtype=torch.bool, device=dev)
        for r, (kind, ja, ii, _) in enumerate(rows):
            for r2, (k2, j2, i2, _) in enumerate(rows):
                if kind == "tok": wm[r, r2] = k2 == "tok" and j2 <= ja
                else: wm[r, r2] = (k2 == "tok" and j2 <= ja) or (k2 == "ph" and j2 == ja and i2 <= ii)
        joined = {}
        Z = lambda: torch.zeros(n, self.NH, self.HD, dtype=self.Emb.dtype, device=dev)
        late = {t: [(Z(), Z()) for _ in self.layers] for t in range(E, T + 1)}                # late[t][li] = (K [n,..], V) at joined q
        has_late = torch.zeros(n, dtype=torch.bool, device=dev)
        out = []
        for s, c in enumerate(anchors, start=1):
            if c + B >= n: break
            l = (anchors[s] - c) if s < len(anchors) else 1
            ids = [text[c + j] if kind == "tok" else self.mask_id for (kind, j, _, _) in rows]
            pos = [c + rp for (_, _, _, rp) in rows]
            X = self.Emb[torch.tensor(ids, device=dev)]; hs = []; step_kv = {}
            allow = torch.cat([torch.ones(NB, c, dtype=torch.bool, device=dev), wm], 1)
            for kk in range(K):
                kp = E + kk
                src = []
                for q in range(c):
                    d = T if q < a else min(T, K * (s - joined[q] - 1) + kk + 1)
                    src.append(True if (negctl or d >= kp) else False)
                srct = torch.tensor(src, dtype=torch.bool, device=dev)[:, None, None]
                assert bool((srct[:, 0, 0] | has_late[:c]).all()), "lateKV read of a position without a token slot"
                def ext(li, kp=kp, srct=srct):
                    Kf, Vf = fd[kp][li]; Kl, Vl = late[kp][li]
                    return torch.where(srct, Kf[:c], Kl[:c]), torch.where(srct, Vf[:c], Vl[:c])
                X, kv = self.pass_rows(X, pos, ext, allow); hs.append(X); step_kv[kp] = kv
            out.append((c, l, self.readout(hs)))
            for j in range(min(l, D)):                                                       # token slots 0..l-1 join the context
                joined[c + j] = s; has_late[c + j] = True
                for kp in range(E, T + 1):
                    for li in range(len(self.layers)):
                        late[kp][li][0][c + j] = step_kv[kp][li][0][j]; late[kp][li][1][c + j] = step_kv[kp][li][1][j]
        return out
