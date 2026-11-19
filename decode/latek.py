"""latek.py -- K-PASS LATE LATTICE, K in {1, 2, 3, 4} (batch 1, greedy readouts, argmax verification). Generalizes ../late2/late.py
(K = 2, read lkv, ver am) to K passes per lattice step. engine.py / rolled.py / sparse.py / late.py here are byte copies of ../late2.

STEP = K forward calls (call kk = 0..K-1), each ONE 48-layer pass over every row of the call. Block pass of call kk: k = E + kk, E = 5 - K
  (1-indexed passes; KV region index = pass - 1). K = 4: passes 1-4 (= the dense exact lattice); K = 2: passes 3-4 (= late.py); K = 1: pass 4.
BLOCK (as late.py / sparse.SparseLattice / the dense lattice): D = B+1 token slots (slot 0 = last committed token, slots 1..B = proposals)
  at positions Lc+j, plus placeholder sets (a, s) after some token slots, placeholder (a, i) at Lc+a+i. Every block row ENTERS AT PASS E
  (token slots with their token's embedding, placeholders with the <|mask|> embedding) and runs passes E..4; the hidden state carries
  across the K calls. Visibility inside the block = the lattice mask at the same pass.
READOUT of a block row = its own gate mixture over passes E..4, renormalised: weight lam_k * rem at pass k < 4, rem (the remainder) at pass
  4, rem = 1 at the entry pass (K = 4: Ouro's standard mixture; K = 2: lam3 / 1 - lam3 as late.py; K = 1: the pass-4 logits alone). fp32.
VERIFICATION after the last call (pass 4): proposal i (token slot i) accepted iff it equals the argmax of token slot i-1; c = leading run of
  accepted proposals; commit proposals 1..c + token slot c's argmax; token slots 0..c join the context (Lc += c+1); next proposals = the
  deepest placeholder set a* <= c (sparse.py rule). First step after prefill: no proposals.
BACKGROUND (K < 4): a position committed at step s0 gets an exact full-depth computation that starts at call 0 of step s0+1 and advances ONE
  pass per call, batched with the block rows: at call kk of step s it does pass j = K*(s - s0 - 1) + kk + 1 (while j <= 4), input = token
  embedding at j = 1, else its state after pass j-1 (per-position store Hs). Group g = the positions committed g steps ago (g = 1..NG,
  NG = ceil(4/K)); group g is ACTIVE in call kk iff K*(g-1) + kk + 1 <= 4 (K = 1: 4 groups at passes 1..4 every call; K = 2: as late.py,
  group A passes 1-2, group B passes 3-4; K = 3: group 1 passes 1-3, group 2 pass 4 in call 0 only). Each active group is padded to D rows
  at the front (pads write a dummy KV slot / Hs row). A background row at pass j reads every position <= its own at pass j (all have done
  pass j by then) -> the background IS the exact full-depth computation of the committed text.
  K = 4: NO background. A token slot that enters at pass 1 with its token embedding and reads exact context at every pass computes exact
  full-depth K/V, so the K/V its committing step wrote are already the background's (CPU-checked: every context K/V at every pass equals
  the full-depth K/V of the committed text). bg=True forces the redundant background (CPU test only; equal decode).
d(p) = passes the background of context position p has completed, INCLUDING the current call (its K/V of the current call are written layer
  by layer before the attention of that layer, so rows of the same call read them): prompt positions and K = 4: 4; otherwise
  d(p) = min(4, K*(s - s0 - 1) + kk + 1) at call kk of step s for p committed at step s0.
LATEKV READING RULE (generalizes late.py's "lkv"): a block row at pass k reads context position p
  - from the background's pass-k K/V if d(p) >= k;
  - else (a newest committed position whose background has not reached pass k) from the K/V that p's TOKEN SLOT produced at pass k in the
    lattice step that committed it, if that step computed pass k;
  - else from p's deepest completed background pass d(p).
  Because K is fixed per decoder, the committing step computed every block pass E..4 and the third case never occurs. All of these K/V sit
  at the same page: region k-1, slot p (the token slot at position p wrote its pass-k K/V there in its committing step; the background
  overwrites it with the exact pass-k K/V when it reaches pass k; nothing else writes region k-1 slot p once p < Lc). So the page table is
  simply "region k-1 for every context position", and K and V always come from the same page.
Rows per call kk: D * (#active groups) + NB.  Mode string: lk:<K>:<shape>, shape = b0 | latt:B | sp:B:a.s+... (sparse.py syntax).
"""
import os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
import sparse as SP
import late as L2

LateEngine = L2.LateEngine
parse_shape = L2.parse_shape


def parse_latek(mode):
    """'lk:K:shape' (background catch-up for K < T), 'lknb:K:shape' (NO background: the mechanism-ablation arm) or
    'lkq:<theta>:K:shape' (LOSSY acceptance: a proposal is also accepted when its probability under the verifying
    slot's readout is >= theta -- the threshold-curve baseline) -> dict(K, B, sets, shape, bg, theta)"""
    assert mode.startswith(("lk:", "lknb:", "lkq:", "lkdb:")), mode
    theta = None
    if mode.startswith("lkq:"):
        pre, th, K, shape = mode.split(":", 3); theta = float(th)
    else:
        pre, K, shape = mode.split(":", 2)
    K = int(K); assert 1 <= K <= 4, mode
    B, sets = parse_shape(shape)
    return dict(K=K, B=B, sets=sets, shape=shape, bg=None if pre != "lknb" else False, theta=theta, db=pre == "lkdb")


def n_groups(K, bg, T=4):
    return -(-T // K) if bg else 0


def active_groups(K, NG, kk, T=4):
    return [g for g in range(1, NG + 1) if K * (g - 1) + kk + 1 <= T]


def rows_of(mode, T=4):
    """rows per call (list over the K calls)"""
    p = parse_latek(mode); D = p["B"] + 1; NB = D + sum(s for _, s in p["sets"]); NG = n_groups(p["K"], p["K"] < T, T)
    return [D * len(active_groups(p["K"], NG, kk, T)) + NB for kk in range(p["K"])]


class _Call:
    pass


class LateKLattice:
    def __init__(self, eng, K, B, sets, nv, eos, mask_id, attn="paged", graphs=None, check=False, bg=None, theta=None, db=False):
        self.T = eng.c.T                                     # eeblk: generalized from the T=4-only original (SNAPSHOT_MD5 has the base)
        assert 1 <= K <= self.T, (K, self.T)
        self.e, self.K, self.B, self.nv, self.eos, self.mask_id, self.check = eng, K, B, nv, set(eos), mask_id, check
        self.theta = theta
        self.db = bool(db)                                   # deep bonus: the newest token's catch-up starts THIS step (extra bg row)
        if self.db:
            assert K < eng.c.T and B >= 1, "lkdb needs K < T and proposals"
            self.db_theta = float(os.environ.get("LK_DB_THETA", "2") or 2)   # fire when slot-0 top1p < this; 2 = always
        self.Ep = self.T + 1 - K                                                          # entry pass
        self.bg = (K < self.T) if bg is None else bool(bg)
        # bg=False at K < T is the no-catch-up ABLATION arm (lknb:): committed tokens keep their provisional
        # entry-pass K/V forever; only an explicit request may cross this line.
        assert self.bg or K == self.T or bg is False, "K < T needs the background (token-slot K/V are not full depth)"
        self.NG = n_groups(K, self.bg, self.T)
        c = eng.c; dev = eng.dev; S = eng.S; SL = eng.SL; self.S, self.SL = S, SL
        self.attn = attn if attn in ("paged", "torch") else "paged"
        self.graphs_on = str(dev).startswith("cuda") if graphs is None else graphs
        D = B + 1; self.D = D
        assert (B == 0 and not sets) or (B >= 1 and sets and sets[0][0] == 0), (B, sets)
        self.sets = list(sets); self.set_a = [a for a, _ in sets]; self.set_s = [s for _, s in sets]
        offs = []; o = D
        for _, s in sets: offs.append(o); o += s
        self.set_off = offs; NB = o; self.NB = NB
        assert NB - D <= eng.M * eng.Bmax, ("placeholder area too small", NB - D, eng.M * eng.Bmax)
        self.astar = [max(a for a in self.set_a if a <= cc) for cc in range(B + 1)] if sets else [None]
        self.kidx = {a: k for k, a in enumerate(self.set_a)}
        # block structure (block-local row r: token slots 0..D-1, placeholders D..NB-1) -- identical to late.py
        relpos = list(range(D)) + [a + i for a, s in sets for i in range(1, s + 1)]
        tails = [list(range(D))] + [list(range(a + 1)) + [offs[k] + i for i in range(s)] for k, (a, s) in enumerate(sets)]
        cub = [0, D] + [offs[k] + s for k, (a, s) in enumerate(sets)]
        self.nblk = len(tails); TM = max(len(x) for x in tails)
        tp = torch.zeros(self.nblk, TM, dtype=torch.int64)
        for sq, tl_ in enumerate(tails): tp[sq, : len(tl_)] = torch.tensor(tl_)
        wm = torch.zeros(NB, NB, dtype=torch.int8)
        for j in range(D): wm[j, : j + 1] = 1
        for k, (a, s) in enumerate(sets):
            for i in range(1, s + 1):
                r = offs[k] + i - 1; wm[r, : a + 1] = 1; wm[r, offs[k] : r + 1] = 1
        for sq, tl_ in enumerate(tails):
            for qi, r in enumerate(range(cub[sq], cub[sq + 1])):
                vis = set(tl_[: len(tl_) - (cub[sq + 1] - cub[sq]) + qi + 1]); assert vis == set(wm[r].nonzero().flatten().tolist()), (B, sets, sq, r)
        assert len(relpos) == NB and cub[-1] == NB
        self.wm = wm; self.tails = tails
        self.relpos = torch.tensor(relpos, device=dev); self.tailpad = tp.to(dev); self.lens = torch.tensor([len(x) for x in tails], device=dev)
        rr = torch.arange(NB, device=dev); self.is_tok = rr < D; self.ph_slot = S + rr - D
        self.col = torch.arange(S, device=dev)[None]; self.ar = torch.arange(D, device=dev); self.rr = rr
        # per-call structure
        self.calls = []
        for kk in range(K):
            cl = _Call(); cl.kk = kk; cl.kp = self.Ep + kk
            cl.act = active_groups(K, self.NG, kk, self.T); cl.gpass = [K * (g - 1) + kk + 1 for g in cl.act]
            nA = len(cl.act); cl.DB = 1 if self.db else 0
            cl.G = nA * D + cl.DB; cl.N = cl.G + NB; cl.nseq = nA + cl.DB + self.nblk
            cu = [D * i for i in range(nA + 1)] + ([nA * D + 1] if cl.DB else []) + [cl.G + x for x in cub[1:]]
            m = E.Meta(); m.cu_q = torch.tensor(cu, device=dev, dtype=torch.int32); m.max_q = max(cu[i + 1] - cu[i] for i in range(cl.nseq))
            m.pos = torch.zeros(cl.N, dtype=torch.int64, device=dev); m.wslot = torch.zeros(cl.N, dtype=torch.int64, device=dev)
            m.seqlens = torch.zeros(cl.nseq, dtype=torch.int32, device=dev); m.page_table = torch.zeros(cl.nseq, S, dtype=torch.int32, device=dev)
            cl.meta = m; cl.hidx = torch.zeros(max(cl.G, 1), dtype=torch.int64, device=dev)
            self.calls.append(cl)
        self.N_max = max(cl.N for cl in self.calls)
        # static buffers
        self.GI = D if self.NG else 0                                                     # ids layout: [group-1 ids (D) | block ids (NB)]
        self.hdr = torch.zeros(self.NG + 1, dtype=torch.int64, device=dev)                 # [Lc, Lc 1 step ago, ..., NG steps ago]
        self.ids = torch.full((self.GI + NB,), mask_id, dtype=torch.int64, device=dev)
        self.Xb = torch.zeros(NB, c.H, device=dev, dtype=eng.dt)
        self.Hs = torch.zeros(S + 1, c.H, device=dev, dtype=eng.dt); self.HSD = S
        self.rem = torch.ones(NB, device=dev); self.exp = torch.zeros(NB, nv, device=dev); self.top = torch.zeros(NB, dtype=torch.int64, device=dev)
        self.pacc = torch.zeros(max(1, D - 1), device=dev)
        self.db_rem = torch.ones(1, device=dev); self.db_exp = torch.zeros(1, nv, device=dev)
        if self.db:
            assert NB - D < eng.M * eng.Bmax, "no spare region slot for the deep-bonus scratch"
            self.DBSLOT = S + eng.M * eng.Bmax - 1
            # fixed-shape finish mini-calls (graph-capturable): rows = [pad to D | last group | db row].
            # Pads sit FIRST so the real rows end the query segment and the kernel's causal mask is exact.
            Rf = D + 1
            self.db_ms = []
            for _ in range(self.T - K):
                mf = E.Meta(); mf.cu_q = torch.tensor([0, Rf], device=dev, dtype=torch.int32); mf.max_q = Rf
                mf.pos = torch.zeros(Rf, dtype=torch.int64, device=dev); mf.wslot = torch.zeros(Rf, dtype=torch.int64, device=dev)
                mf.seqlens = torch.zeros(1, dtype=torch.int32, device=dev); mf.page_table = torch.zeros(1, S, dtype=torch.int32, device=dev)
                self.db_ms.append(mf)
            self.dbX = torch.zeros(Rf, c.H, device=dev, dtype=eng.dt)
            self.db_deep = torch.zeros(1, dtype=torch.int64, device=dev)
            self.db_graph = None
            if str(dev).startswith("cuda"): self.h_db_deep = torch.zeros(1, dtype=torch.int64).pin_memory()
        self.db_h = torch.zeros(1, c.H, device=dev, dtype=eng.dt); self.db_c0p = torch.zeros(1, device=dev)
        self.dump_conf = os.environ.get("LK_DUMP_CONF") == "1"
        self.p1 = torch.zeros(NB, device=dev)
        if str(dev).startswith("cuda"):
            self.h_top = torch.zeros(NB, dtype=torch.int64).pin_memory(); self.h_ids = torch.zeros(self.GI + NB, dtype=torch.int64).pin_memory()
            self.h_pacc = torch.zeros(max(1, D - 1)).pin_memory()
            self.h_dbp = torch.zeros(1).pin_memory()
            self.h_p1 = torch.zeros(NB).pin_memory()
            self.h_hdr = torch.zeros(self.NG + 1, dtype=torch.int64).pin_memory(); self.ev = torch.cuda.Event()
        self.graphs = None
        self.reads = []

    @property
    def N(self):
        return self.N_max

    @property
    def nseq(self):
        return max(cl.nseq for cl in self.calls)

    # ---- device side of one call (captured) ----
    def _prep(self, kk):
        e, D, S, SL = self.e, self.D, self.S, self.SL
        cl = self.calls[kk]; m = cl.meta; rb = cl.kp - 1
        lc = self.hdr[0]; col = self.col; DUM = e.DUMMY
        pos_l, w_l, h_l, pt_l, sl_l = [], [], [], [], []
        for g, pj in zip(cl.act, cl.gpass):
            hi, lo = self.hdr[g - 1], self.hdr[g]; reg = pj - 1
            posg = hi - D + self.ar; real = self.ar >= D - (hi - lo)
            pos_l.append(posg); w_l.append(torch.where(real, reg * SL + posg, torch.full_like(posg, DUM)))
            h_l.append(torch.where(real, posg, torch.full_like(posg, self.HSD)))
            pt_l.append(torch.where(col < hi, reg * SL + col, torch.zeros_like(col))); sl_l.append(hi.view(1))
        if getattr(cl, "DB", 0):
            regd = cl.kk                                                                 # newest token runs pass kk+1 this call
            pos_l.append(lc.view(1)); w_l.append(torch.full((1,), 0, dtype=torch.int64, device=self.hdr.device) + regd * SL + self.DBSLOT)
            h_l.append(torch.full((1,), self.HSD, dtype=torch.int64, device=self.hdr.device))
            ptd = torch.where(col < lc, regd * SL + col, torch.zeros_like(col))
            ptd = torch.where(col == lc, torch.full_like(col, regd * SL + self.DBSLOT), ptd)
            pt_l.append(ptd); sl_l.append((lc + 1).view(1))
        wslot_loc = torch.where(self.is_tok, lc + self.rr, self.ph_slot)
        pos_l.append(lc + self.relpos); w_l.append(rb * SL + wslot_loc)
        rel = col - lc
        tail = self.tailpad.gather(1, rel.clamp(0, self.tailpad.shape[1] - 1).expand(self.nblk, -1))
        ptb = torch.where(rel < 0, rb * SL + col, rb * SL + wslot_loc[tail])                 # lateKV rule: region k-1 for every context position
        ptb = torch.where(rel >= self.lens[:, None], torch.zeros_like(ptb), ptb)
        pt_l.append(ptb); sl_l.append(lc + self.lens)
        m.pos.copy_(torch.cat(pos_l)); m.wslot.copy_(torch.cat(w_l))
        if cl.G: cl.hidx.copy_(torch.cat(h_l))
        m.page_table.copy_(torch.cat(pt_l).to(torch.int32)); m.seqlens.copy_(torch.cat(sl_l).to(torch.int32))

    def _pass(self, kk):
        e, D = self.e, self.D
        cl = self.calls[kk]; self._prep(kk)
        xs = []
        for gi, pj in enumerate(cl.gpass):
            if pj == 1: xs.append(F.embedding(self.ids[: self.GI], e.embed))                 # group 1 enters pass 1 with its token embedding
            else: xs.append(self.Hs.index_select(0, cl.hidx[gi * D : (gi + 1) * D]))
        if getattr(cl, "DB", 0):
            xs.append(F.embedding(self.ids[self.GI : self.GI + 1], e.embed) if kk == 0 else self.db_h)   # newest token's early catch-up
        xs.append(F.embedding(self.ids[self.GI :], e.embed) if kk == 0 else self.Xb)      # block rows enter at pass E
        X = torch.cat(xs) if len(xs) > 1 else xs[0]
        hn = e.late_pass(X, cl.meta, self.attn)
        if cl.G: self.Hs.index_copy_(0, cl.hidx, hn[: cl.G])
        if getattr(cl, "DB", 0):
            hd = hn[cl.G - 1 : cl.G]; self.db_h.copy_(hd)
            if kk == 0: self.db_rem.fill_(1.0); self.db_exp.zero_()
            lamd = e.gate(hd)
            self.db_exp.add_(F.linear(hd, e.lm_head)[:, : self.nv].float() * (lamd * self.db_rem)[:, None])
            self.db_rem.mul_(1 - lamd)
        hb = hn[cl.G :]
        if kk < self.K - 1: self.Xb.copy_(hb)
        lam = e.gate(hb)
        lg = F.linear(hb, e.lm_head)[:, : self.nv]
        if kk == 0: self.rem.fill_(1.0)
        if cl.kp < self.T:
            pt = lam * self.rem; self.rem.mul_(1 - lam)
        else:
            pt = self.rem
        if kk == 0: torch.mul(lg.float(), pt[:, None], out=self.exp)
        else: self.exp.add_(lg.float() * pt[:, None])
        if cl.kp == self.T:
            self.top.copy_(self.exp.argmax(-1))
            if self.dump_conf:
                self.p1.copy_(torch.softmax(self.exp, -1).max(-1).values)
            if self.db:
                self.db_c0p.copy_(torch.softmax(self.exp[0], -1).max().view(1))
            if self.theta is not None and self.D > 1:
                pr = torch.softmax(self.exp[: self.D - 1], -1)
                tok = self.ids[self.GI + 1 : self.GI + self.D].clamp(max=self.nv - 1)   # filler slots hold mask_id; their pacc is never read (c < np_)
                self.pacc.copy_(pr.gather(1, tok[:, None])[:, 0])

    def warm_hdr(self):
        return [(self.NG + 3 - i) * self.D for i in range(self.NG + 1)]

    def capture(self):
        if not self.graphs_on: return
        self.hdr.copy_(torch.tensor(self.warm_hdr(), device=self.hdr.device))
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                for kk in range(self.K): self._pass(kk)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        pool = torch.cuda.graph_pool_handle(); gs = []
        for kk in range(self.K):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool): self._pass(kk)
            gs.append(g)
        torch.cuda.synchronize(); self.graphs = gs
        if self.db:
            wh = self.warm_hdr()
            self._db_prep(wh[0], wh[1])
            s2 = torch.cuda.Stream(); s2.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s2):
                for _ in range(2): self._db_chain()
            torch.cuda.current_stream().wait_stream(s2); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool): self._db_chain()
            torch.cuda.synchronize(); self.db_graph = g

    def replay(self):
        for g in self.graphs: g.replay()

    def step(self, hist, ids_host):
        assert all(h >= self.D for h in hist[: max(1, self.NG)]), (hist, self.D)
        if self.graphs is not None:
            for i, h in enumerate(hist): self.h_hdr[i] = h
            self.h_ids.copy_(torch.as_tensor(ids_host, dtype=torch.int64))
            self.hdr.copy_(self.h_hdr, non_blocking=True); self.ids.copy_(self.h_ids, non_blocking=True)
            self.replay()
            self.h_top.copy_(self.top, non_blocking=True)
            if self.theta is not None: self.h_pacc.copy_(self.pacc, non_blocking=True)
            if self.dump_conf: self.h_p1.copy_(self.p1, non_blocking=True)
            if self.db: self.h_dbp.copy_(self.db_c0p, non_blocking=True)
            self.ev.record(); self.ev.synchronize()
            return self.h_top.tolist(), (self.h_pacc.tolist() if self.theta is not None else None), (self.h_dbp.item() if self.db else None)
        self.hdr.copy_(torch.tensor(hist, device=self.hdr.device)); self.ids.copy_(torch.as_tensor(ids_host, dtype=torch.int64, device=self.ids.device))
        for kk in range(self.K):
            self._pass(kk)
            if self.check:
                cl = self.calls[kk]; m = cl.meta
                self.reads.append(dict(kk=kk, kpass=cl.kp, hdr=list(hist), act=list(cl.act), gpass=list(cl.gpass), G=cl.G,
                                       pt=m.page_table.cpu().clone(), sl=m.seqlens.cpu().clone(), cu=m.cu_q.cpu().clone(),
                                       pos=m.pos.cpu().clone(), wslot=m.wslot.cpu().clone()))
        return self.top.tolist(), (self.pacc.tolist() if self.theta is not None else None), (float(self.db_c0p.item()) if self.db else None)


    def _db_prep(self, lc, lo):
        """fill the fixed-shape finish batch: rows = [D-n pads | group positions lo..hi-1 | db row], one Meta per finish pass.
        Pads write to the dummy slot and their outputs are never read; real rows end the query segment, so the paged
        kernel's causal mask (last Lq positions of the sequence) is exact for them."""
        e, D, S, SL = self.e, self.D, self.S, self.SL; dev = self.hdr.device
        hi = int(lc); lo = int(lo); n = hi - lo; p0 = D - n
        assert 0 <= n <= D, (lo, hi, D)
        gi = torch.arange(lo, hi, device=dev)
        if p0: self.dbX[:p0].zero_()
        if n: self.dbX[p0:D].copy_(self.Hs.index_select(0, gi))
        self.dbX[D : D + 1].copy_(self.db_h)
        pos = torch.zeros(D + 1, dtype=torch.int64, device=dev)
        pos[p0:D] = gi; pos[D] = hi
        col = self.col; DUM = e.DUMMY
        for i, j in enumerate(range(self.K + 1, self.T + 1)):
            reg = j - 1; m = self.db_ms[i]
            m.pos.copy_(pos)
            ws = reg * SL + pos
            if p0: ws[:p0] = DUM
            ws[D] = reg * SL + self.DBSLOT
            m.wslot.copy_(ws)
            m.seqlens.fill_(hi + 1)
            pt = torch.where(col < hi, reg * SL + col, torch.zeros_like(col))
            pt = torch.where(col == hi, torch.full_like(col, reg * SL + self.DBSLOT), pt)
            m.page_table.copy_(pt.to(torch.int32))

    def _db_chain(self):
        """finish passes K+1..T on the fixed-shape batch (graph-capturable; same attention path as the main calls).
        Reads follow the lateKV invariant: pass j reads region j-1 of positions <= own; the group's rows are advanced in
        the same mini-call so the newest token's deep reads hit real entries. Next step's normal schedule recomputes the
        same values (deterministic, harmless)."""
        e = self.e; X = self.dbX
        for i, j in enumerate(range(self.K + 1, self.T + 1)):
            hn = e.late_pass(X, self.db_ms[i], self.attn)
            hd = hn[self.D : self.D + 1]
            lamd = e.gate(hd)
            w = (lamd * self.db_rem) if j < self.T else self.db_rem
            self.db_exp.add_(F.linear(hd, e.lm_head)[:, : self.nv].float() * w[:, None])
            if j < self.T: self.db_rem.mul_(1 - lamd)
            X = hn
        self.db_deep.copy_(self.db_exp.argmax(-1))

    @torch.no_grad()
    def _db_finish(self, lc, lo):
        """finish passes K+1..T for [last committed group | newest token] so the newest token's full-depth readout exists."""
        self._db_prep(lc, lo)
        if self.db_graph is not None:
            self.db_graph.replay()
            self.h_db_deep.copy_(self.db_deep, non_blocking=True)
            self.ev.record(); self.ev.synchronize()
            return int(self.h_db_deep.item())
        self._db_chain()
        return int(self.db_deep.item())

    def next_props(self, top, c):
        if not self.sets: return []
        a = self.astar[c]; k = self.kidx[a]; off, s = self.set_off[k], self.set_s[k]
        i0 = c - a; n = max(0, min(self.B, s - i0))
        return [top[off + i0 + j - 1] for j in range(1, n + 1)]

    @torch.no_grad()
    def decode(self, prompt_ids, max_new, stop_fn=None, prefill_attn="auto", hook=None):
        """hook(dec, step_index, hist, known) after every step (CPU tests)"""
        e, B, D, NB = self.e, self.B, self.D, self.NB; P0 = len(prompt_ids)
        assert P0 >= D, ("prompt shorter than the background group padding", P0, D)
        assert P0 + max_new + NB + B + 8 <= e.S, ("cache too small", P0, max_new, NB, e.S)
        cuda = str(e.dev).startswith("cuda")
        t0 = time.perf_counter()
        first, m0 = e.prefill(list(prompt_ids), self.nv, attn="torch" if self.attn == "torch" else prefill_attn)
        if cuda: torch.cuda.synchronize()
        t_pref = time.perf_counter() - t0
        known = list(prompt_ids) + [first]; hist = [P0] * (self.NG + 1); props = []; done = first in self.eos
        log, nprop, cs, mbits, step_ms, gpu_ms = [], [], [], [], [], []
        n_acc = 0; filler = [self.mask_id] * B; p1s = []; n_db = 0; n_dbh = 0
        while not done and len(known) - P0 < max_new:
            ts = time.perf_counter(); lc = hist[0]
            ga = [known[p] if p >= hist[1] else self.mask_id for p in range(lc - D, lc)] if self.NG else []
            ids = ga + [known[-1]] + props + filler[len(props):] + [self.mask_id] * (NB - D)
            top, pacc, dbp = self.step(hist, ids)
            if self.dump_conf: p1s.append([round(float(x), 4) for x in (self.h_p1 if self.graphs is not None else self.p1).tolist()])
            tg = time.perf_counter()
            np_ = len(props); mb = 0
            for i in range(np_):
                if top[i] == props[i]: mb |= 1 << i
            c = 0
            th = self.theta
            while c < np_ and (top[c] == props[c] or (th is not None and pacc[c] >= th)): c += 1
            n_acc += c
            n_db_fire = 0; n_db_hit = 0
            if self.db and np_ > 0 and c == 0 and (dbp is None or dbp < self.db_theta):
                n_db_fire = 1
                deep = self._db_finish(lc, hist[1] if self.NG else lc)
                if deep != top[0]: n_db_hit = 1
                commit = [deep]
            else:
                commit = props[:c] + [top[c]]
            props = self.next_props(top, c)
            hist = [lc + c + 1] + hist[: self.NG]                                          # token slots 0..c join the context
            for tk in commit:
                known.append(tk)
                if tk in self.eos: done = True; break
            n_db += n_db_fire; n_dbh += n_db_hit
            log.append(len(commit)); nprop.append(np_); cs.append(c); mbits.append(mb)
            if hook is not None: hook(self, len(log), list(hist), known)
            if stop_fn is not None and not done and stop_fn(known[P0:]): done = True
            te = time.perf_counter(); step_ms.append(1000 * (te - ts)); gpu_ms.append(1000 * (tg - ts))
        gen = known[P0:]
        return dict(gen=[t for t in gen if t not in self.eos], gen_raw=gen, log=log, nprop=nprop, c=cs, mbits=mbits, steps=len(log),
                    step_ms=step_ms, gpu_ms=gpu_ms, prefill_ms=1000 * t_pref, n_acc=n_acc, n_gate=0, n_amrej=0, margins=None, p1=p1s if self.dump_conf else None, n_db=n_db if self.db else None, n_dbh=n_dbh if self.db else None)


def make_decoder(eng, mode, nv, eos, mask_id, **kw):
    if mode.startswith(("lk:", "lknb:", "lkq:", "lkdb:")):
        p = parse_latek(mode)
        for k_ in ("margins", "switch_L", "dump"): kw.pop(k_, None)
        at = kw.pop("attn", "paged"); at = "torch" if at == "torch" else "paged"
        if p["bg"] is not None: kw["bg"] = p["bg"]
        if p["theta"] is not None: kw["theta"] = p["theta"]
        if p.get("db"): kw["db"] = True
        return LateKLattice(eng, p["K"], p["B"], p["sets"], nv, eos, mask_id, attn=at, **kw)
    kw.pop("check", None); kw.pop("bg", None)
    return SP.make_decoder(eng, mode, nv, eos, mask_id, **kw)


@torch.no_grad()
def check_attention_latek(eng, mode, seed=0, hdr_base=900, log=print):
    """one layer of every call with random buffers: paged FA3 vs the torch reference (max abs error over all rows)"""
    torch.manual_seed(seed); c = eng.c; H = c.H
    eng.kbuf[0].normal_(); eng.vbuf[0].normal_()
    p = parse_latek(mode)
    L = LateKLattice(eng, p["K"], p["B"], p["sets"], 8, [0], 1, attn="paged", graphs=False)
    hist = [hdr_base - i * (L.D // 2 + 1) for i in range(L.NG + 1)]
    errs = []
    for kk in range(L.K):
        L.hdr.copy_(torch.tensor(hist, device=L.hdr.device)); L._prep(kk); m = L.calls[kk].meta; N = L.calls[kk].N
        qkv = torch.randn(N, 3 * H, device=eng.dev, dtype=eng.dt); q, k, v = qkv[:, :H], qkv[:, H : 2 * H], qkv[:, 2 * H :]
        eng.kbuf[0].view(-1, H).index_copy_(0, m.wslot, k); eng.vbuf[0].view(-1, H).index_copy_(0, m.wslot, v)
        ref = E.attn_torch(q, eng.kbuf[0], eng.vbuf[0], m, c.NH, c.HD)
        out = E.attn_paged(q, eng.kbuf[0], eng.vbuf[0], m, c.NH, c.HD)
        errs.append((out.float() - ref.float()).abs().max().item())
    log(f"[check] latek attention {mode} rows/call={[cl.N for cl in L.calls]} hdr={hist}: max abs err paged vs torch {max(errs):.5f}")
    return max(errs)
