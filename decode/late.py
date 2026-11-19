"""late.py -- TWO-PASS LATE LATTICE on the decode substrate (batch 1, greedy readouts, LOSSY by design).

User's design (10-27): "one pass to propose, one pass to verify; always read the oldest/available context, which may differ per position".
engine.py / rolled.py / sparse.py here are byte copies of ../engine.py, ../rolled/rolled.py, ../sparse/sparse.py.

A lattice STEP = two forward calls, each ONE 48-layer pass over every row of the call:
  call A = pass 3 of the block, call B = pass 4 of the block (1-indexed passes; KV region index = pass - 1).
BLOCK (same row structure as sparse.SparseLattice / the dense lattice): D = B+1 token slots (slot 0 = last committed token, slots 1..B =
  proposals) at positions Lc+j, plus placeholder sets (a, s) after some token slots, placeholder (a, i) at Lc+a+i. Every block row ENTERS AT
  PASS 3: token slots with their token's embedding, placeholders with the <|mask|> embedding; call B continues from the call-A state.
  Visibility inside the block = the lattice mask at the same pass (token slot j: slots 0..j; placeholder (a, i): slots 0..a + (a, 1..i)).
CONTEXT (positions < Lc, committed): position p has a background full-depth computation that started in the call after p was committed
  and advances one pass per call, batched with the block rows. d(p) = passes it has completed INCLUDING the current call (its K/V of the
  current call are written layer by layer before the attention of that layer, so a row of the same call can read them; no waiting).
  Prompt and old positions: d = 4. Tokens committed by the previous step: d = 1 in call A, 2 in call B; two steps ago: 3 / 4.
  read = "old" (oldest-available, the user's rule): a block row at pass k reads position p's K and V at pass min(k, d(p))
         -> every context position at pass k, except the previous step's tokens, which are read at pass k-2.
  read = "lkv": positions with d(p) < k are read from the K/V their TOKEN SLOT produced (late entry, passes 3-4) in the lattice step that
         committed them; positions with d(p) >= k from the background.  -> every context position at "pass k".
  K and V of a position always come from the same page (= the same pass): paged attention addresses both through one page index.
BACKGROUND rows of a call: group A = the c+1 positions committed by the previous step (pass 1 in call A, pass 2 in call B, input = token
  embedding at pass 1), group B = the positions committed two steps ago (pass 3 / pass 4). A background row at pass j reads every position
  <= its own at pass j (all of them have done pass j by then) -> the background IS the exact full-depth computation of the committed text.
  Hidden states between passes live in a per-position store Hs. Each group is padded to D rows (pads write a dummy KV slot / Hs row).
READOUT of a block row = its own gate mixture over passes 3-4, renormalised (gate weight lam3 at pass 3, the remainder 1 - lam3 at pass 4;
  = the vendor late rows' readout, rem starts at 1 at the entry pass), fp32, argmax over nv.
VERIFICATION after call B (proposal i sits in token slot i, verified by token slot i-1):
  ver "am"     : accept iff proposal == argmax of token slot i-1
  ver "g<th>"  : accept iff p_i * q_i >= th   (p_i = the proposer's max probability = softmax of the placeholder row that proposed it, read
                 in the previous step; q_i = token slot i-1's probability of the proposal)   -- the literal one-gate rule
  ver "go<th>" : accept iff argmax OR p_i * q_i >= th  (extra arm)
  c = leading run of accepted proposals (capped at the real proposals); commit proposals 1..c + token slot c's argmax (the correction);
  token slots 0..c join the context (Lc += c+1); the next proposals come from the deepest placeholder set a* <= c (sparse.py rule; tail
  padded with <|mask|>, acceptance capped by count). First step after prefill: no proposals (c = 0), as the exact engines.
No committed token is ever a full-depth prediction (the prefill's first token excepted): every token is a 2-pass late-entry readout.

Mode strings: l2:<read>:<ver>:<shape>   read in {old, lkv}; ver in {am, g<th>, go<th>}; shape = b0 (B=0: one token slot, no proposals)
  | latt:B (dense = sparse with a full set after every slot) | sp:B:a.s+... (sparse.py syntax).
"""
import os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
import rolled as R
import sparse as SP


# ------------------------------------------------------------------------------------------------------------- engine
class LateEngine(R.RolledEngine):
    """OuroEngine whose per-loop KV caches are views into per-layer combined buffers [T*SL] (rolled.RolledEngine): region t = pass t+1;
    slots [0, S) of a region = positions (prefix / context / block token slots), [S, S + M*Bmax) = block placeholders, S + M*Bmax = dummy.
    The exact decoders (AR, sparse) run unchanged on the same object through the kc/vc views."""

    def late_pass(self, X, meta, attn):
        c, ops = self.c, self.ops; H, NH, HD = c.H, c.NH, c.HD
        res = X.clone()
        h = ops.rms(res, self.layers[0]["n1"])
        for li, Lw in enumerate(self.layers):
            qkv = F.linear(h, Lw["qkv"])
            q, k, v = qkv[:, :H], qkv[:, H : 2 * H], qkv[:, 2 * H :]
            q, k = ops.rope(q, k, meta.pos)
            kb, vb = self.kbuf[li], self.vbuf[li]
            kb.view(-1, H).index_copy_(0, meta.wslot, k); vb.view(-1, H).index_copy_(0, meta.wslot, v)   # every row's K/V before attention
            o = E.attn_paged(q, kb, vb, meta, NH, HD) if attn == "paged" else E.attn_torch(q, kb, vb, meta, NH, HD)
            ao = ops.rms(F.linear(o, Lw["o"]), Lw["n2"])
            h, res = ops.add_rms(ao, res, Lw["n3"])
            m = ops.rms(F.linear(ops.silu_mul(F.linear(h, Lw["gu"])), Lw["down"]), Lw["n4"])
            if li + 1 < c.NL: h, res = ops.add_rms(m, res, self.layers[li + 1]["n1"])
            else: res = res + m
        return ops.rms(res, self.norm)


# ------------------------------------------------------------------------------------------------------------- modes
def parse_shape(shape):
    if shape == "b0": return 0, []
    if shape.startswith("latt:"):
        B = int(shape.split(":")[1]); return B, [(a, B) for a in range(B + 1)]
    kind, B, sets = SP.parse_mode(shape); assert kind == "sparse", shape
    return B, sets


def parse_late(mode):
    """'l2:read:ver:shape' -> dict(read, ver, theta, B, sets)"""
    assert mode.startswith("l2:"), mode
    _, read, ver, shape = mode.split(":", 3)
    assert read in ("old", "lkv"), mode
    if ver == "am": kind, th = "am", None
    elif ver.startswith("go"): kind, th = "go", float(ver[2:])
    elif ver.startswith("g"): kind, th = "g", float(ver[1:])
    else: raise ValueError(mode)
    B, sets = parse_shape(shape)
    return dict(read=read, ver=kind, theta=th, B=B, sets=sets, shape=shape)


def rows_of(mode):
    if mode.startswith("l2:"):
        p = parse_late(mode); D = p["B"] + 1; return 2 * D + D + sum(s for _, s in p["sets"])
    return SP.rows_of(mode)


# ------------------------------------------------------------------------------------------------------------- decoder
class LateLattice:
    def __init__(self, eng, B, sets, nv, eos, mask_id, read="old", ver="am", theta=None, attn="paged", graphs=None, check=False):
        assert eng.c.T == 4, "two-pass late lattice is defined for T=4 (block passes 3-4)"
        self.e, self.B, self.nv, self.eos, self.mask_id = eng, B, nv, set(eos), mask_id
        self.read, self.ver, self.theta, self.check = read, ver, theta, check
        assert read in ("old", "lkv") and ver in ("am", "g", "go")
        c = eng.c; dev = eng.dev; self.T = c.T; S = eng.S; SL = eng.SL; self.S, self.SL = S, SL
        self.attn = attn if attn in ("paged", "torch") else "paged"
        self.graphs_on = str(dev).startswith("cuda") if graphs is None else graphs
        D = B + 1; self.D = D
        assert (B == 0 and not sets) or (B >= 1 and sets and sets[0][0] == 0), (B, sets)
        self.sets = list(sets); self.set_a = [a for a, _ in sets]; self.set_s = [s for _, s in sets]
        offs = []; o = D
        for _, s in sets: offs.append(o); o += s
        self.set_off = offs; NB = o; self.NB = NB; self.G = 2 * D; N = self.G + NB; self.N = N
        assert NB - D <= eng.M * eng.Bmax, ("placeholder area too small", NB - D, eng.M * eng.Bmax)
        self.astar = [max(a for a in self.set_a if a <= cc) for cc in range(B + 1)] if sets else [None]
        self.kidx = {a: k for k, a in enumerate(self.set_a)}
        # block structure (block-local row index r: token slots 0..D-1, placeholders D..NB-1)
        relpos = list(range(D)) + [a + i for a, s in sets for i in range(1, s + 1)]
        tails = [list(range(D))] + [list(range(a + 1)) + [offs[k] + i for i in range(s)] for k, (a, s) in enumerate(sets)]
        cu = [0, D, 2 * D] + [2 * D + D] + [2 * D + offs[k] + s for k, (a, s) in enumerate(sets)]
        self.nblk = len(tails); self.nseq = 2 + self.nblk; TM = max(len(x) for x in tails)
        tp = torch.zeros(self.nblk, TM, dtype=torch.int64)
        for sq, tl_ in enumerate(tails): tp[sq, : len(tl_)] = torch.tensor(tl_)
        # consistency with the lattice mask (as sparse.py)
        wm = torch.zeros(NB, NB, dtype=torch.int8)
        for j in range(D): wm[j, : j + 1] = 1
        for k, (a, s) in enumerate(sets):
            for i in range(1, s + 1):
                r = offs[k] + i - 1; wm[r, : a + 1] = 1; wm[r, offs[k] : r + 1] = 1
        cub = [0, D] + [offs[k] + s for k, (a, s) in enumerate(sets)]
        for sq, tl_ in enumerate(tails):
            for qi, r in enumerate(range(cub[sq], cub[sq + 1])):
                vis = set(tl_[: len(tl_) - (cub[sq + 1] - cub[sq]) + qi + 1]); assert vis == set(wm[r].nonzero().flatten().tolist()), (B, sets, sq, r)
        assert len(relpos) == NB and cu[-1] == N, (len(relpos), NB, cu, N)
        self.wm = wm
        self.relpos = torch.tensor(relpos, device=dev); self.tailpad = tp.to(dev); self.lens = torch.tensor([len(x) for x in tails], device=dev)
        self.cu_q = torch.tensor(cu, device=dev, dtype=torch.int32); self.max_q = max(cu[i + 1] - cu[i] for i in range(self.nseq))
        rr = torch.arange(NB, device=dev); self.is_tok = rr < D; self.ph_slot = S + rr - D      # placeholder window slot (region-relative)
        self.col = torch.arange(S, device=dev)[None]; self.ar = torch.arange(D, device=dev)
        # static buffers
        self.hdr = torch.zeros(3, dtype=torch.int64, device=dev)                                # [Lc, P1, P2]
        self.ids = torch.full((N,), mask_id, dtype=torch.int64, device=dev)
        self.Xb = torch.zeros(NB, c.H, device=dev, dtype=eng.dt)
        self.Hs = torch.zeros(S + 1, c.H, device=dev, dtype=eng.dt); self.HSD = S                # background states per position (+ dummy)
        self.rem = torch.ones(NB, device=dev); self.exp = torch.zeros(NB, nv, device=dev); self.top = torch.zeros(NB, dtype=torch.int64, device=dev)
        self.pq = torch.zeros(NB + max(D - 1, 0), device=dev)                                  # [0:NB) max prob per block row, [NB:) q of slots 0..D-2
        m = E.Meta(); m.cu_q = self.cu_q; m.max_q = self.max_q
        m.pos = torch.zeros(N, dtype=torch.int64, device=dev); m.wslot = torch.zeros(N, dtype=torch.int64, device=dev)
        m.seqlens = torch.zeros(self.nseq, dtype=torch.int32, device=dev); m.page_table = torch.zeros(self.nseq, S, dtype=torch.int32, device=dev)
        self.hidx = torch.zeros(2 * D, dtype=torch.int64, device=dev)
        self.meta = m
        if str(dev).startswith("cuda"):
            self.h_top = torch.zeros(NB, dtype=torch.int64).pin_memory(); self.h_ids = torch.zeros(N, dtype=torch.int64).pin_memory()
            self.h_hdr = torch.zeros(3, dtype=torch.int64).pin_memory(); self.h_pq = torch.zeros(self.pq.shape[0]).pin_memory()
            self.ev = torch.cuda.Event()
        self.graphs = None
        self.reads = []                                                                          # check mode: per call (kpass, page table, seqlens, hdr)

    # ---- device side of one call (captured; kk = 0 -> block pass 3, kk = 1 -> block pass 4) ----
    def _prep(self, kk):
        e, m, D, S, SL = self.e, self.meta, self.D, self.S, self.SL
        kp = 3 + kk; rb = kp - 1; ra = kp - 3; rB = kp - 1                 # regions: block (pass k), group A (pass k-2), group B (pass k)
        lc, p1, p2 = self.hdr[0], self.hdr[1], self.hdr[2]
        col = self.col
        # rows: group A (previous step's tokens), group B (two steps ago), padded to D rows each at the FRONT (bottom-right causal)
        posA = lc - D + self.ar; realA = self.ar >= D - (lc - p1)
        posB = p1 - D + self.ar; realB = self.ar >= D - (p1 - p2)
        DUM = e.DUMMY
        wA = torch.where(realA, ra * SL + posA, torch.full_like(posA, DUM)); wB = torch.where(realB, rB * SL + posB, torch.full_like(posB, DUM))
        wslot_loc = torch.where(self.is_tok, lc + torch.arange(self.NB, device=lc.device), self.ph_slot)      # block window slot within a region
        m.pos.copy_(torch.cat([posA, posB, lc + self.relpos]))
        m.wslot.copy_(torch.cat([wA, wB, rb * SL + wslot_loc]))
        self.hidx.copy_(torch.cat([torch.where(realA, posA, torch.full_like(posA, self.HSD)), torch.where(realB, posB, torch.full_like(posB, self.HSD))]))
        # page tables
        ptA = torch.where(col < lc, ra * SL + col, torch.zeros_like(col))
        ptB = torch.where(col < p1, rB * SL + col, torch.zeros_like(col))
        rel = col - lc
        tail = self.tailpad.gather(1, rel.clamp(0, self.tailpad.shape[1] - 1).expand(self.nblk, -1))
        if self.read == "old": creg = torch.where((col >= p1) & (col < lc), torch.full_like(col, ra), torch.full_like(col, rb))
        else: creg = torch.full_like(col, rb)
        ptb = torch.where(rel < 0, creg * SL + col, rb * SL + wslot_loc[tail])
        ptb = torch.where(rel >= self.lens[:, None], torch.zeros_like(ptb), ptb)
        m.page_table.copy_(torch.cat([ptA, ptB, ptb]).to(torch.int32))
        m.seqlens.copy_(torch.cat([lc.view(1), p1.view(1), lc + self.lens]).to(torch.int32))

    def _pass(self, kk):
        e, D, G = self.e, self.D, self.G
        self._prep(kk)
        if kk == 0:
            xa = F.embedding(self.ids[:D], e.embed)                                   # group A enters pass 1 with its token embedding
            xb = self.Hs.index_select(0, self.hidx[D:])                               # group B: state after pass 2
            xk = F.embedding(self.ids[G:], e.embed)                                    # block rows enter at pass 3
        else:
            xa = self.Hs.index_select(0, self.hidx[:D]); xb = self.Hs.index_select(0, self.hidx[D:]); xk = self.Xb
        X = torch.cat([xa, xb, xk])
        hn = self.e.late_pass(X, self.meta, self.attn)
        self.Hs.index_copy_(0, self.hidx, hn[:G])
        hb = hn[G:]
        if kk == 0: self.Xb.copy_(hb)
        lam = e.gate(hb)
        lg = F.linear(hb, e.lm_head)[:, : self.nv]
        if kk == 0:
            self.rem.fill_(1.0); pt = lam * self.rem; self.rem.mul_(1 - lam)
            torch.mul(lg.float(), pt[:, None], out=self.exp)
        else:
            self.exp.add_(lg.float() * self.rem[:, None])
            self.top.copy_(self.exp.argmax(-1))
            lse = torch.logsumexp(self.exp, -1)
            self.pq[: self.NB].copy_(torch.exp(self.exp.gather(1, self.top[:, None])[:, 0] - lse))
            if D > 1:
                nid = self.ids[G + 1 : G + D].clamp(max=self.nv - 1)
                self.pq[self.NB :].copy_(torch.exp(self.exp[: D - 1].gather(1, nid[:, None])[:, 0] - lse[: D - 1]))

    def capture(self):
        if not self.graphs_on: return
        # warm-up / capture with a valid header (writes land in context-area slots that every decode rewrites before reading)
        self.hdr.copy_(torch.tensor([4 * self.D + 8, 3 * self.D + 4, 2 * self.D], device=self.hdr.device))
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                for kk in range(2): self._pass(kk)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        pool = torch.cuda.graph_pool_handle(); gs = []
        for kk in range(2):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool): self._pass(kk)
            gs.append(g)
        torch.cuda.synchronize(); self.graphs = gs

    def step(self, lc, p1, p2, ids_host):
        assert p2 >= self.D and p2 <= p1 <= lc, (lc, p1, p2, self.D)
        if self.graphs is not None:
            self.h_hdr[0] = lc; self.h_hdr[1] = p1; self.h_hdr[2] = p2; self.h_ids.copy_(torch.as_tensor(ids_host, dtype=torch.int64))
            self.hdr.copy_(self.h_hdr, non_blocking=True); self.ids.copy_(self.h_ids, non_blocking=True)
            self.graphs[0].replay(); self.graphs[1].replay()
            self.h_top.copy_(self.top, non_blocking=True); self.h_pq.copy_(self.pq, non_blocking=True)
            self.ev.record(); self.ev.synchronize()
            return self.h_top.tolist(), self.h_pq.tolist()
        self.hdr.copy_(torch.tensor([lc, p1, p2], device=self.hdr.device)); self.ids.copy_(torch.as_tensor(ids_host, dtype=torch.int64, device=self.ids.device))
        for kk in range(2):
            self._pass(kk)
            if self.check:
                self.reads.append(dict(kpass=3 + kk, hdr=(lc, p1, p2), pt=self.meta.page_table.cpu().clone(), sl=self.meta.seqlens.cpu().clone(),
                                       cu=self.meta.cu_q.cpu().clone(), pos=self.meta.pos.cpu().clone(), wslot=self.meta.wslot.cpu().clone()))
        return self.top.tolist(), self.pq.tolist()

    def next_props(self, top, c):
        if not self.sets: return [], (0, 0)
        a = self.astar[c]; k = self.kidx[a]; off, s = self.set_off[k], self.set_s[k]
        i0 = c - a
        n = max(0, min(self.B, s - i0))
        return [top[off + i0 + j - 1] for j in range(1, n + 1)], (off + i0, n)

    @torch.no_grad()
    def decode(self, prompt_ids, max_new, stop_fn=None, prefill_attn="auto", hook=None):
        """hook(dec, step_index, lc, p1, p2, joined) is called after every step (CPU tests)"""
        e, B, D, NB, G = self.e, self.B, self.D, self.NB, self.G; P0 = len(prompt_ids)
        assert P0 >= D, ("prompt shorter than the background group padding", P0, D)
        assert P0 + max_new + NB + B + 8 <= e.S, ("cache too small", P0, max_new, NB, e.S)
        cuda = str(e.dev).startswith("cuda")
        t0 = time.perf_counter()
        first, m0 = e.prefill(list(prompt_ids), self.nv, attn="torch" if self.attn == "torch" else prefill_attn)
        if cuda: torch.cuda.synchronize()
        t_pref = time.perf_counter() - t0
        known = list(prompt_ids) + [first]; lc = p1 = p2 = P0; props, pprop = [], []; done = first in self.eos
        log, nprop, cs, mbits, step_ms, gpu_ms = [], [], [], [], [], []
        n_acc = n_gate = n_amrej = 0
        filler = [self.mask_id] * B
        th = self.theta
        while not done and len(known) - P0 < max_new:
            ts = time.perf_counter()
            ga = [known[p] if p >= p1 else self.mask_id for p in range(lc - D, lc)]           # group A ids (pass-1 embeddings); pads -> mask
            ids = ga + [self.mask_id] * D + [known[-1]] + props + filler[len(props):] + [self.mask_id] * (NB - D)
            top, pq = self.step(lc, p1, p2, ids)
            tg = time.perf_counter()
            top = top[:NB]
            np_ = len(props); mb = 0
            for i in range(np_):
                if top[i] == props[i]: mb |= 1 << i
            c = 0
            while c < np_:
                am = top[c] == props[c]
                if self.ver == "am": ok = am
                else:
                    g = pprop[c] * pq[NB + c] >= th
                    ok = g if self.ver == "g" else (am or g)
                if not ok:
                    if am: n_amrej += 1
                    break
                if not am: n_gate += 1
                c += 1
            n_acc += c
            commit = props[:c] + [top[c]]
            nxt, (r0, n) = self.next_props(top, c)
            p2, p1 = p1, lc; lc += c + 1                                                           # token slots 0..c join the context
            props = nxt; pprop = pq[r0 : r0 + n]
            for tk in commit:
                known.append(tk)
                if tk in self.eos: done = True; break
            log.append(len(commit)); nprop.append(np_); cs.append(c); mbits.append(mb)
            if hook is not None: hook(self, len(log), lc, p1, p2, known)
            if stop_fn is not None and not done and stop_fn(known[P0:]): done = True
            te = time.perf_counter(); step_ms.append(1000 * (te - ts)); gpu_ms.append(1000 * (tg - ts))
        gen = known[P0:]
        return dict(gen=[t for t in gen if t not in self.eos], gen_raw=gen, log=log, nprop=nprop, c=cs, mbits=mbits, steps=len(log),
                    step_ms=step_ms, gpu_ms=gpu_ms, prefill_ms=1000 * t_pref, n_acc=n_acc, n_gate=n_gate, n_amrej=n_amrej, margins=None)


def make_decoder(eng, mode, nv, eos, mask_id, **kw):
    if mode.startswith("l2:"):
        p = parse_late(mode)
        kw.pop("margins", None); kw.pop("switch_L", None); kw.pop("dump", None)
        at = kw.pop("attn", "paged"); at = "torch" if at == "torch" else "paged"
        return LateLattice(eng, p["B"], p["sets"], nv, eos, mask_id, read=p["read"], ver=p["ver"], theta=p["theta"], attn=at, **kw)
    kw.pop("check", None)
    return SP.make_decoder(eng, mode, nv, eos, mask_id, **kw)


# ------------------------------------------------------------------------------------------------------------- checks
@torch.no_grad()
def check_attention_late(eng, mode, seed=0, hdr=(137, 131, 126), log=print):
    """one layer of one call (both kk) with random buffers: paged FA3 vs the torch reference (max abs error over all rows)"""
    torch.manual_seed(seed); c = eng.c; H = c.H
    eng.kbuf[0].normal_(); eng.vbuf[0].normal_()
    p = parse_late(mode)
    L = LateLattice(eng, p["B"], p["sets"], 8, [0], 1, read=p["read"], ver=p["ver"], theta=p["theta"], attn="paged", graphs=False)
    errs = []
    for kk in range(2):
        L.hdr.copy_(torch.tensor(list(hdr), device=L.hdr.device)); L._prep(kk); m = L.meta
        qkv = torch.randn(L.N, 3 * H, device=eng.dev, dtype=eng.dt); q, k, v = qkv[:, :H], qkv[:, H : 2 * H], qkv[:, 2 * H :]
        eng.kbuf[0].view(-1, H).index_copy_(0, m.wslot, k); eng.vbuf[0].view(-1, H).index_copy_(0, m.wslot, v)
        ref = E.attn_torch(q, eng.kbuf[0], eng.vbuf[0], m, c.NH, c.HD)
        out = E.attn_paged(q, eng.kbuf[0], eng.vbuf[0], m, c.NH, c.HD)
        errs.append((out.float() - ref.float()).abs().max().item())
    log(f"[check] late attention {mode} N={L.N} hdr={hdr}: max abs err paged vs torch {max(errs):.5f}")
    return max(errs)


@torch.no_grad()
def audit_full_depth(eng, ids, P0, n, nv):
    """teacher-forced full-depth T-pass forward of ids (prompt + generated); returns the gate-mixture argmax predicting positions
    P0 .. P0+n-1 (i.e. the readouts of positions P0-1 .. P0+n-2) and their top-2 margins. Writes the main-area caches (harmless: every
    decode starts with a prefill)."""
    dev = eng.dev; P = len(ids); assert P <= eng.S
    meta = E.Meta(); meta.pos = torch.arange(P, device=dev); meta.slot = meta.pos
    meta.page_table = torch.arange(P, device=dev, dtype=torch.int32)[None]; meta.seqlens = torch.tensor([P], device=dev, dtype=torch.int32)
    meta.cu_q = torch.tensor([0, P], device=dev, dtype=torch.int32); meta.max_q = P
    attn = "paged" if str(dev).startswith("cuda") else "torch"
    X = F.embedding(torch.tensor(ids, device=dev), eng.embed); sel = torch.arange(P0 - 1, P0 - 1 + n, device=dev)
    rem = torch.ones(n, device=dev); exp = None
    for t in range(eng.c.T):
        X = eng.loop_pass(X, t, meta, attn)
        hn = X.index_select(0, sel); lam = eng.gate(hn)
        pt = lam * rem if t < eng.c.T - 1 else rem; rem = rem * (1 - lam)
        lg = F.linear(hn, eng.lm_head)[:, :nv].float()
        exp = lg * pt[:, None] if exp is None else exp + lg * pt[:, None]
    tv = exp.topk(2, -1)
    return tv.indices[:, 0].tolist(), (tv.values[:, 0] - tv.values[:, 1]).tolist()
