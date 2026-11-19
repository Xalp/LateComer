"""late_step.py -- the decode executors, scheduler-agnostic (no ScheduleBatch here).

LateStepExecutor = decode/late.py LateLattice (_prep / _pass / capture / step) ported to
  * R requests per call (rows of request b at offset b*N, its attention sequences at b*nseq), and
  * SGLang's KV addressing: position p of the request in batch row b lives at base slot req_to_token[rpi_b, p]; loop t of base slot s
    at t*STRIDE + s of the flat per-layer buffer (kv_pool.LoopStridedMHAPool). Placeholder rows write the per-row scratch area of
    regions 2/3 (t*STRIDE + SCR0 + b*PH + i); pad rows write the region's dummy slot t*STRIDE + 0.
  The hidden store late.py indexes by position is a per-request ring HB[rpi*RING + pos % RING] (live positions [p2, lc) span <= 2D).
  Float math (embedding, 48-layer pass, gate, lm_head, fp32 mixture lam3 / 1 - lam3, argmax) is the same op sequence as late.py, so at
  R = 1 with the same page-table width W the result is bit-identical to late.py on the same GPU.
ARStepExecutor = engine.Lattice with B = 0 (plain full-depth AR: one token slot, T graphs of a 1-row pass), same addressing.
LateReqState = the per-request host state of late.LateLattice.decode (lc, p1, p2, proposals) with its ids builder and verify/commit.
"""
import torch
import torch.nn.functional as F
from .core_ops import Meta


class LateReqState:
    """lc = positions in the context (KV committed), p1 / p2 = lc one / two steps ago, props = proposals for the next step"""
    __slots__ = ("lc", "p1", "p2", "props", "steps")

    def __init__(self, P0):
        self.lc = self.p1 = self.p2 = P0; self.props = []; self.steps = 0

    def build_ids(self, tok, last, shape, mask_id):
        """late.LateLattice.decode: ids = group A (tokens committed by the previous step, front-padded with mask) + group B (D masks) +
        token slots (last committed token, proposals, mask filler) + placeholders (masks). tok(p) = committed token at position p."""
        D, NB, B = shape.D, shape.NB, shape.B; lc, p1 = self.lc, self.p1
        ga = [tok(p) if p >= p1 else mask_id for p in range(lc - D, lc)]
        pr = self.props
        return ga + [mask_id] * D + [last] + pr + [mask_id] * (B - len(pr)) + [mask_id] * (NB - D)

    def advance(self, top, shape):
        """argmax verification (late.py ver 'am'): c = leading run of proposals equal to the verifying slot's argmax (capped at the real
        proposals: <|mask|> is inside the argmax range); commit proposals[:c] + top[c]; token slots 0..c join the context."""
        pr = self.props; npr = len(pr); c = 0
        while c < npr and top[c] == pr[c]: c += 1
        commit = pr[:c] + [top[c]]
        self.props = shape.next_props(top, c)
        self.p2, self.p1 = self.p1, self.lc; self.lc += c + 1; self.steps += 1
        return commit, c


def _pinned(shape, dtype, cuda):
    t = torch.zeros(shape, dtype=dtype)
    return t.pin_memory() if cuda else t


class LateStepExecutor:
    """one lattice step (2 forward calls = block passes 3 and 4) for R requests; see module docstring"""

    def __init__(self, core, shape, kv, rtt, W, nv, mask_id, R=1, read="lkv", attn="paged", graphs=None, log=print, hb_rows=None, margins=False):
        c = core.c; dev = core.dev
        assert c.T == 4, "two-pass late lattice is defined for T=4 (block passes 3-4)"
        assert read in ("lkv", "old")
        self.core, self.sh, self.kv, self.rtt, self.W, self.nv, self.mask_id, self.R = core, shape, kv, rtt, W, nv, mask_id, R
        self.read, self.attn, self.log, self.dev = read, attn, log, dev
        self.cuda = str(dev).startswith("cuda")
        self.graphs_on = self.cuda if graphs is None else (graphs and self.cuda)
        sh = shape; D, NB, N, G, PH, nseq = sh.D, sh.NB, sh.N, sh.G, sh.PH, sh.nseq
        assert W <= rtt.shape[1], (W, rtt.shape)
        assert kv.SCR >= R * PH, ("placeholder scratch too small", kv.SCR, R, PH)
        core.ops.ensure_rope(W + 64)
        self.RING = 4 * D
        nrows = hb_rows if hb_rows is not None else rtt.shape[0]
        self.HB = torch.zeros(nrows * self.RING + 1, c.H, device=dev, dtype=core.dt); self.HBD = nrows * self.RING
        # constants
        i64 = dict(dtype=torch.int64, device=dev)
        self.ar = torch.arange(D, **i64); self.arNB = torch.arange(NB, **i64)
        self.relpos = torch.tensor(sh.relpos, **i64); self.tailpad = sh.tailpad.to(dev); self.lens = torch.tensor(sh.lens, **i64)
        self.is_tok = self.arNB < D; self.ph_off = (self.arNB - D).clamp(min=0)
        self.bidx = torch.arange(R, **i64)
        self.col = torch.arange(W, **i64)[None]                                                    # [1, W]
        cu = [b * N + x for b in range(R) for x in sh.cu[:-1]] + [R * N]
        self.cu_q = torch.tensor(cu, dtype=torch.int32, device=dev)
        rr = torch.arange(R * N, **i64); loc = rr % N
        self.rowsAB = rr[loc < G].contiguous(); self.rowsB = rr[(loc >= D) & (loc < G)].contiguous(); self.rowsK = rr[loc >= G].contiguous()
        self.ab_isB = ((self.rowsAB % N) >= D)
        # static per-step inputs / outputs
        self.hdr = torch.zeros(R, 3, **i64); self.rpi = torch.zeros(R, **i64)
        self.ids = torch.full((R * N,), mask_id, **i64)
        self.X = torch.zeros(R * N, c.H, device=dev, dtype=core.dt)
        self.Xb = torch.zeros(R * NB, c.H, device=dev, dtype=core.dt)
        self.rem = torch.ones(R * NB, device=dev); self.exp = torch.zeros(R * NB, nv, device=dev); self.top = torch.zeros(R * NB, **i64)
        self.margins = margins; self.gap = torch.zeros(R * NB, device=dev); self.last_gap = None     # diagnostic: top-2 gap of every block row
        m = Meta(); m.cu_q = self.cu_q; m.max_q = sh.max_q
        m.pos = torch.zeros(R * N, **i64); m.wslot = torch.zeros(R * N, **i64)
        m.seqlens = torch.zeros(R * nseq, dtype=torch.int32, device=dev); m.page_table = torch.zeros(R * nseq, W, dtype=torch.int32, device=dev)
        self.meta = m
        self.hidx = torch.zeros(R * G, **i64)
        self.h_hdr = _pinned((R, 3), torch.int64, self.cuda); self.h_rpi = _pinned((R,), torch.int64, self.cuda)
        self.h_ids = _pinned((R * N,), torch.int64, self.cuda); self.h_top = _pinned((R * NB,), torch.int64, self.cuda)
        self.h_gap = _pinned((R * NB,), torch.float32, self.cuda)
        if self.cuda: self.ev = torch.cuda.Event()
        self.graphs = None
        self.reads = None                                                                       # check mode (tests): list of per-call metas

    # ---- device side of one call (captured; kk = 0 -> block pass 3, kk = 1 -> block pass 4) ----
    def _prep(self, kk):
        sh, kv, m, W = self.sh, self.kv, self.meta, self.W
        D, NB, R = sh.D, sh.NB, self.R; ST = kv.STRIDE
        kp = 3 + kk; ra = kp - 3; rb = kp - 1
        lc, p1, p2 = self.hdr[:, 0:1], self.hdr[:, 1:2], self.hdr[:, 2:3]                       # [R, 1]
        base = self.rtt[:, :W].index_select(0, self.rpi).long()                                 # [R, W] base slots by position
        posA = lc - D + self.ar[None]; realA = self.ar[None] >= D - (lc - p1)                   # front-padded groups (bottom-right causal)
        posB = p1 - D + self.ar[None]; realB = self.ar[None] >= D - (p1 - p2)
        bA = base.gather(1, posA.clamp(min=0)); bB = base.gather(1, posB.clamp(min=0))
        wA = torch.where(realA, ra * ST + bA, torch.full_like(bA, ra * ST))                      # pads -> the region's dummy slot
        wB = torch.where(realB, rb * ST + bB, torch.full_like(bB, rb * ST))
        tokp = (lc + self.arNB[None]).clamp(max=W - 1)
        bK = base.gather(1, tokp)                                                               # token slot j at position lc + j
        wK = torch.where(self.is_tok[None], rb * ST + bK, rb * ST + kv.SCR0 + self.bidx[:, None] * sh.PH + self.ph_off[None])   # [R, NB]
        posK = lc + self.relpos[None]
        m.pos.copy_(torch.cat([posA.clamp(min=0), posB.clamp(min=0), posK], 1).reshape(-1))
        m.wslot.copy_(torch.cat([wA, wB, wK], 1).reshape(-1))
        hrA = self.rpi[:, None] * self.RING + posA.clamp(min=0) % self.RING
        hrB = self.rpi[:, None] * self.RING + posB.clamp(min=0) % self.RING
        self.hidx.copy_(torch.cat([torch.where(realA, hrA, torch.full_like(hrA, self.HBD)), torch.where(realB, hrB, torch.full_like(hrB, self.HBD))], 1).reshape(-1))
        col = self.col
        ptA = torch.where(col < lc, ra * ST + base, torch.zeros_like(base))
        ptB = torch.where(col < p1, rb * ST + base, torch.zeros_like(base))
        rel = col - lc                                                                          # [R, W]
        nb, TM = self.tailpad.shape
        tail = self.tailpad[None].expand(R, nb, TM).gather(2, rel.clamp(0, TM - 1)[:, None, :].expand(R, nb, W))     # [R, nblk, W]
        twin = wK.gather(1, tail.reshape(R, -1)).reshape(R, nb, W)
        if self.read == "old": creg = torch.where((col >= p1) & (col < lc), ra * ST + base, rb * ST + base)
        else: creg = rb * ST + base
        ptb = torch.where(rel[:, None, :] < 0, creg[:, None, :], twin)
        ptb = torch.where(rel[:, None, :] >= self.lens[None, :, None], torch.zeros_like(ptb), ptb)
        m.page_table.copy_(torch.cat([ptA[:, None], ptB[:, None], ptb], 1).reshape(R * sh.nseq, W).to(torch.int32))
        m.seqlens.copy_(torch.cat([lc, p1, lc + self.lens[None]], 1).reshape(-1).to(torch.int32))

    def _pass(self, kk):
        core, sh = self.core, self.sh; D, G, NB = sh.D, sh.G, sh.NB
        self._prep(kk)
        X = self.X
        if kk == 0:
            X.copy_(F.embedding(self.ids, core.embed))                                          # group A + block rows enter with their embedding
            X.index_copy_(0, self.rowsB, self.HB.index_select(0, self.hidx.view(self.R, G)[:, D:].reshape(-1)))   # group B: state after pass 2
        else:
            X.index_copy_(0, self.rowsAB, self.HB.index_select(0, self.hidx))
            X.index_copy_(0, self.rowsK, self.Xb)
        hn = core.pass_rows(X, self.meta, self.kv, self.attn)
        if self.R == 1:
            self.HB.index_copy_(0, self.hidx, hn[:G]); hb = hn[G:]
        else:
            self.HB.index_copy_(0, self.hidx, hn.index_select(0, self.rowsAB)); hb = hn.index_select(0, self.rowsK)
        if kk == 0: self.Xb.copy_(hb)
        lam = core.gate(hb)
        lg = F.linear(hb, core.lm_head)[:, : self.nv]
        if kk == 0:
            self.rem.fill_(1.0); pt = lam * self.rem; self.rem.mul_(1 - lam)
            torch.mul(lg.float(), pt[:, None], out=self.exp)
        else:
            self.exp.add_(lg.float() * self.rem[:, None])
            self.top.copy_(self.exp.argmax(-1))
            if self.margins:
                v2 = self.exp.topk(2, -1).values; self.gap.copy_(v2[:, 0] - v2[:, 1])

    def _set_dummy_inputs(self):
        D = self.sh.D
        self.hdr.copy_(torch.tensor([[4 * D + 8, 3 * D + 4, 2 * D]] * self.R, device=self.dev)); self.ids.fill_(self.mask_id)

    def capture(self):
        """late.LateLattice.capture: 2 warm-up iterations of both calls on a side stream (valid header, rpi as set), then one graph per
        call in ONE PRIVATE pool that is never freed (cuda13-image: shared pool + del -> allocator assert)"""
        if not self.graphs_on: return
        self._set_dummy_inputs()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                for kk in range(2): self._pass(kk)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        self.pool = torch.cuda.graph_pool_handle(); gs = []
        for kk in range(2):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool): self._pass(kk)
            gs.append(g)
        torch.cuda.synchronize(); self.graphs = gs

    def step(self, hdrs, rpis, ids_rows):
        """hdrs = [(lc, p1, p2)] * R, rpis = [req_pool_idx] * R, ids_rows = [N ids] * R -> [NB block argmaxes] * R (one host sync)"""
        R, NB = self.R, self.sh.NB
        assert len(hdrs) == R, ("bucket padding is M2", len(hdrs), R)
        for b, (lc, p1, p2) in enumerate(hdrs):
            assert p2 <= p1 <= lc and lc + self.sh.D + self.sh.TM < self.W, (lc, p1, p2, self.W)
        self.h_hdr.copy_(torch.as_tensor(hdrs, dtype=torch.int64)); self.h_rpi.copy_(torch.as_tensor(rpis, dtype=torch.int64))
        self.h_ids.copy_(torch.as_tensor([x for r in ids_rows for x in r], dtype=torch.int64))
        if self.graphs is not None:
            self.hdr.copy_(self.h_hdr, non_blocking=True); self.rpi.copy_(self.h_rpi, non_blocking=True); self.ids.copy_(self.h_ids, non_blocking=True)
            self.graphs[0].replay(); self.graphs[1].replay()
            self.h_top.copy_(self.top, non_blocking=True)
            if self.margins: self.h_gap.copy_(self.gap, non_blocking=True)
            self.ev.record(); self.ev.synchronize()
            top = self.h_top.tolist()
            if self.margins: self.last_gap = self.h_gap.tolist()
        else:
            self.hdr.copy_(self.h_hdr); self.rpi.copy_(self.h_rpi); self.ids.copy_(self.h_ids)
            for kk in range(2):
                self._pass(kk)
                if self.reads is not None:
                    m = self.meta
                    self.reads.append(dict(kpass=3 + kk, hdr=list(map(tuple, hdrs)), pt=m.page_table.cpu().clone(), sl=m.seqlens.cpu().clone(),
                                           pos=m.pos.cpu().clone(), wslot=m.wslot.cpu().clone()))
            top = self.top.tolist()
            if self.margins: self.last_gap = self.gap.tolist()
        return [top[b * NB : (b + 1) * NB] for b in range(R)]


class ARStepExecutor:
    """plain full-depth AR (engine.Lattice with B = 0, paged): one row per request at position lc reading the context + itself at every
    pass; T passes, one graph per pass; readout = Ouro's gate mixture over the T passes"""

    def __init__(self, core, kv, rtt, W, nv, mask_id, R=1, attn="paged", graphs=None, log=print):
        c = core.c; dev = core.dev
        self.core, self.kv, self.rtt, self.W, self.nv, self.mask_id, self.R, self.attn, self.dev = core, kv, rtt, W, nv, mask_id, R, attn, dev
        self.cuda = str(dev).startswith("cuda"); self.graphs_on = self.cuda if graphs is None else (graphs and self.cuda); self.T = c.T
        core.ops.ensure_rope(W + 64)
        i64 = dict(dtype=torch.int64, device=dev)
        self.col = torch.arange(W, **i64)[None]
        self.lcur = torch.zeros(R, **i64); self.rpi = torch.zeros(R, **i64); self.ids = torch.full((R,), mask_id, **i64)
        self.X = torch.zeros(R, c.H, device=dev, dtype=core.dt)
        self.rem = torch.ones(R, device=dev); self.exp = torch.zeros(R, nv, device=dev); self.top = torch.zeros(R, **i64)
        m = Meta(); m.cu_q = torch.arange(R + 1, dtype=torch.int32, device=dev); m.max_q = 1
        m.pos = torch.zeros(R, **i64); m.wslot = torch.zeros(R, **i64)
        m.seqlens = torch.zeros(R, dtype=torch.int32, device=dev); m.page_table = torch.zeros(R, W, dtype=torch.int32, device=dev)
        self.meta = m
        self.h_lc = _pinned((R,), torch.int64, self.cuda); self.h_rpi = _pinned((R,), torch.int64, self.cuda)
        self.h_ids = _pinned((R,), torch.int64, self.cuda); self.h_top = _pinned((R,), torch.int64, self.cuda)
        if self.cuda: self.ev = torch.cuda.Event()
        self.graphs = None

    def _prep(self, t):
        m, ST = self.meta, self.kv.STRIDE; lc = self.lcur
        base = self.rtt[:, : self.W].index_select(0, self.rpi).long()
        m.pos.copy_(lc); m.wslot.copy_(t * ST + base.gather(1, lc[:, None])[:, 0])
        m.page_table.copy_(torch.where(self.col <= lc[:, None], t * ST + base, torch.zeros_like(base)).to(torch.int32))
        m.seqlens.copy_((lc + 1).to(torch.int32))

    def _pass(self, t):
        core = self.core
        self._prep(t)
        if t == 0:
            self.X.copy_(F.embedding(self.ids, core.embed)); self.rem.fill_(1.0)
        hn = core.pass_rows(self.X, self.meta, self.kv, self.attn)
        self.X.copy_(hn)
        lam = core.gate(hn)
        lg = F.linear(hn, core.lm_head)[:, : self.nv]
        if t < self.T - 1:
            pt = lam * self.rem; self.rem.mul_(1 - lam)
        else:
            pt = self.rem
        if t == 0: torch.mul(lg.float(), pt[:, None], out=self.exp)
        else: self.exp.add_(lg.float() * pt[:, None])
        if t == self.T - 1: self.top.copy_(self.exp.argmax(-1))

    def capture(self):
        if not self.graphs_on: return
        self.lcur.fill_(8); self.ids.fill_(self.mask_id)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                for t in range(self.T): self._pass(t)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        self.pool = torch.cuda.graph_pool_handle(); gs = []
        for t in range(self.T):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool): self._pass(t)
            gs.append(g)
        torch.cuda.synchronize(); self.graphs = gs

    def step(self, lcs, rpis, last_ids):
        R = self.R; assert len(lcs) == R
        for lc in lcs: assert lc + 1 < self.W, (lc, self.W)
        self.h_lc.copy_(torch.as_tensor(lcs, dtype=torch.int64)); self.h_rpi.copy_(torch.as_tensor(rpis, dtype=torch.int64))
        self.h_ids.copy_(torch.as_tensor(last_ids, dtype=torch.int64))
        if self.graphs is not None:
            self.lcur.copy_(self.h_lc, non_blocking=True); self.rpi.copy_(self.h_rpi, non_blocking=True); self.ids.copy_(self.h_ids, non_blocking=True)
            for g in self.graphs: g.replay()
            self.h_top.copy_(self.top, non_blocking=True); self.ev.record(); self.ev.synchronize()
            return self.h_top.tolist()
        self.lcur.copy_(self.h_lc); self.rpi.copy_(self.h_rpi); self.ids.copy_(self.h_ids)
        for t in range(self.T): self._pass(t)
        return self.top.tolist()
