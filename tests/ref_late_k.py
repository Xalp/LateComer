"""ref_late_k.py -- INDEPENDENT reference of the K-pass late lattice (CPU tests only). Generalizes ../late2/ref_late.py (imported for its
HF-module forward `pass_rows`, `full_depth`, `readout_parts`; it shares no code with latek.py / late.py / engine.py).

Per step s (1-based) and call kk in 0..K-1 (block pass k = 5 - K + kk):
  full-depth K/V of the committed text = a plain causal T-pass forward over known[:Lc] recomputed from scratch
  d(q) = 4 for prompt positions, else min(4, K*(s - joined[q] - 1) + kk + 1)    (the formula is used for K = 4 too: there d(q) >= k always
         holds for the previous step's tokens, so a K = 4 block always reads full depth, which is what the background-free engine claims)
  lateKV: K/V of q at pass k from the full-depth forward if d(q) >= k, else the K/V that q's token slot produced at pass k in the step that
          committed it (always computed for fixed K; asserted).
Block rows enter at pass 5-K (token / <|mask|> embedding), lattice mask, readout = gate mixture over passes 5-K..4 renormalised, argmax
verification, sparse.py next-window rule.
"""
import torch
from ref_late import RefLate


class RefLateK(RefLate):
    @torch.no_grad()
    def decode_k(self, prompt, max_new, K, B, sets):
        P0 = len(prompt); D = B + 1; Ep = 5 - K
        offs = []; o = D
        for _, s in sets: offs.append(o); o += s
        NB = o
        _, first = self.full_depth(list(prompt))
        known = list(prompt) + [first]; joined = {}; late = {}; Lc = P0; props = []; log = []; done = first in self.eos; s = 0
        while not done and len(known) - P0 < max_new:
            s += 1
            fd, _ = self.full_depth(known[:Lc])
            rows = [("tok", Lc + j, ([known[-1]] + props + [self.mask_id] * B)[j], list(range(j + 1))) for j in range(D)]
            for k_, (a, sz) in enumerate(sets):
                for i in range(1, sz + 1):
                    rows.append(("ph", Lc + a + i, self.mask_id, list(range(a + 1)) + [offs[k_] + ii for ii in range(i)]))
            assert len(rows) == NB
            X = self.E[torch.tensor([r[2] for r in rows])]; pos = [r[1] for r in rows]
            rem = torch.ones(NB, dtype=torch.float32); exp = None; kvs = {}
            store = {("fd", q, t): fd[(q, t)] for q in range(Lc) for t in range(1, 5)}
            store.update({("late", q, t): v for (q, t), v in late.items()})
            for kk in range(K):
                kp = Ep + kk; ctx = []
                for q in range(Lc):
                    d = 4 if q < P0 else min(4, K * (s - joined[q] - 1) + kk + 1)
                    if d >= kp: ctx.append(("ext", ("fd", q, kp)))
                    else:
                        assert ("late", q, kp) in store, ("late K/V missing", q, kp); ctx.append(("ext", ("late", q, kp)))
                keys = [ctx + [("row", j) for j in r[3]] for r in rows]
                X, kv = self.pass_rows(X, pos, keys, store); kvs[kp] = kv
                lam, lg = self.readout_parts(X)
                if kp < 4: pt = lam.float() * rem; rem = rem * (1 - lam.float())
                else: pt = rem
                exp = lg.float() * pt[:, None] if exp is None else exp + lg.float() * pt[:, None]
            top = exp.argmax(-1).tolist()
            c = 0
            while c < len(props) and top[c] == props[c]: c += 1
            commit = props[:c] + [top[c]]
            for j in range(c + 1):
                joined[Lc + j] = s
                for kp in kvs: late[(Lc + j, kp)] = kvs[kp][j]
            if sets:
                a = max(x for x, _ in sets if x <= c); k_ = [x for x, _ in sets].index(a); off, sz = offs[k_], sets[k_][1]
                i0 = c - a; n = max(0, min(B, sz - i0)); props = [top[off + i0 + j - 1] for j in range(1, n + 1)]
            Lc += c + 1
            for tk in commit:
                known.append(tk)
                if tk in self.eos: done = True; break
            log.append(len(commit))
        return known[P0:], log
