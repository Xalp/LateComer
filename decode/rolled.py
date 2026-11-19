"""rolled.py -- ROLLED MARS lattice on the decode substrate (batch 1, greedy, exact).

Semantics = FlowWave flow:B:e:kv with kv = T (= Wavefront wave:B:e) of
MARS/integration/probes/lmeval_ouro/vendor_grid/ouro_stream.py, re-implemented with one forward call = ONE 48-layer pass over every
in-flight row (each row at its own pass index t), padded to a row bucket, one CUDA graph per bucket, one host sync per call.

State = a run of COLUMNS at consecutive positions p = Lc .. newest (Lc = positions in the prefix caches). Column p = token slot p
(the token at position p: confirmed, or a proposal) + its B placeholders (p, i), i = 1..B, at positions p + i. All B+1 rows of a
column enter together (pass 0 = embedding input) and advance one pass per call until pass T; then they stay (inactive) until the
column is folded or flushed. Visibility (exactly the reference's): token slot p sees token slots <= p; placeholder (p, i) sees token
slots <= p and placeholders (p, 1..i). A row at pass t reads the pass-t K/V of what it sees; every row it sees entered no later
than it did, so that K/V exists (written this call or earlier) -> every row's computation is the synchronous T-pass one.
Readout = Ouro's gate mixture over the T passes (fp32), argmax over nv; the raw per-pass argmax is the proposal source before pass T.
Host step after every call (Wavefront.decode, kv = T): walk the token slots in position order; a confirmed slot that finished pass T
verifies the next proposal (match -> confirmed; mismatch -> flush every column after it, and enter a block after it); a confirmed,
finished newest slot enters a block. Block after slot R = R's prediction (exact if R finished, else its current raw argmax) + B
proposals from R's placeholders (mixture if finished, else current raw argmax), each as a new cold column. Speculation: when the
newest slot and its placeholders have done e passes (and it is not a finished confirmed slot), a block is entered after it.
Fold: the front column is confirmed + finished and its successor confirmed -> Lc += 1.

KV layout (no copies, no fold traffic): per layer ONE buffer [T*SL, 1, NH, HD]; loop-t region = slots [t*SL, (t+1)*SL).
  token slot at position p  -> slot p of the region (so a folded token's K/V already sits in the prefix, contiguous)
  placeholder (p, i)        -> slot S + (p mod M)*B + i-1 ;  slot S + M*Bmax = dummy (pad rows)
  rows' hidden state / gate-mixture accumulators live in per-row stores indexed by (p mod M)*(B+1) + i (dummy = last).
The dense engine (AR / synchronous lattice) runs on the same engine object: its per-loop caches eng.kc[t][li] are views of the first S
slots of each region.
Attention ("split"): FA3 varlen over the prefix, one sequence per pass index t (page table row t = region t), non-causal, LSE; then a
Triton kernel attends each row's window = token slots [Lc, p] of its region + its own placeholder run, merged via the LSE.
"split4" = fallback: 4 non-varlen FA3 calls (all rows x each region) + per-row select. "torch" = explicit masked reference (CPU tests).
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E


class RolledEngine(E.OuroEngine):
    """OuroEngine whose per-loop KV caches are views into per-layer combined buffers with a placeholder region per loop"""

    def __init__(self, W, cfg, device, dtype=torch.bfloat16, S=2560, fused=True, log=print, M=64, Bmax=16):
        self.c, self.dev, self.dt, self.S, self.log = cfg, device, dtype, S, log
        c = cfg
        g = lambda k: W[k].to(device=device, dtype=dtype).contiguous()
        self.layers = []
        for i in range(c.NL):
            p = f"model.layers.{i}."
            self.layers.append(dict(
                qkv=torch.cat([g(p + "self_attn.q_proj.weight"), g(p + "self_attn.k_proj.weight"), g(p + "self_attn.v_proj.weight")], 0).contiguous(),
                o=g(p + "self_attn.o_proj.weight"),
                gu=torch.cat([g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")], 0).contiguous(),
                down=g(p + "mlp.down_proj.weight"),
                n1=g(p + "input_layernorm.weight"), n2=g(p + "input_layernorm_2.weight"),
                n3=g(p + "post_attention_layernorm.weight"), n4=g(p + "post_attention_layernorm_2.weight")))
        self.embed = g("model.embed_tokens.weight"); self.norm = g("model.norm.weight")
        self.gw = g("model.early_exit_gate.weight"); self.gb = g("model.early_exit_gate.bias")
        self.lm_head = g("lm_head.weight")
        self.ops = E.Ops(c, device, dtype, fused=fused, log=log)
        self.ops.ensure_rope(S + 64)
        self.prefill_graphs = False; self.pg_bucket = 64; self.pgraphs = {}
        self.M, self.Bmax = M, Bmax; self.SL = S + M * Bmax + 1; self.DUMMY = S + M * Bmax
        self.kbuf = [torch.zeros(c.T * self.SL, 1, c.NH, c.HD, device=device, dtype=dtype) for _ in range(c.NL)]
        self.vbuf = [torch.zeros(c.T * self.SL, 1, c.NH, c.HD, device=device, dtype=dtype) for _ in range(c.NL)]
        self.kc = [[self.kbuf[li][t * self.SL : t * self.SL + S] for li in range(c.NL)] for t in range(c.T)]
        self.vc = [[self.vbuf[li][t * self.SL : t * self.SL + S] for li in range(c.NL)] for t in range(c.T)]

    def rolled_pass(self, X, meta, attn):
        """one 48-layer pass of rows X [N, H], each row at its own pass index; K/V written at meta.wslot of the combined buffers"""
        c, ops = self.c, self.ops; H, NH, HD = c.H, c.NH, c.HD
        res = X.clone()
        h = ops.rms(res, self.layers[0]["n1"])
        for li, Lw in enumerate(self.layers):
            qkv = F.linear(h, Lw["qkv"])
            q, k, v = qkv[:, :H], qkv[:, H : 2 * H], qkv[:, 2 * H :]
            q, k = ops.rope(q, k, meta.pos)
            kb, vb = self.kbuf[li], self.vbuf[li]
            kb.view(-1, H).index_copy_(0, meta.wslot, k); vb.view(-1, H).index_copy_(0, meta.wslot, v)
            if attn == "split": o = attn_rolled_split(q, kb, vb, meta, NH, HD, self)
            elif attn == "split4": o = attn_rolled_split4(q, kb, vb, meta, NH, HD, self)
            elif attn == "fa3only": o = attn_rolled_fa3only(q, kb, vb, meta, NH, HD, self)      # TIMING DIAGNOSTIC ONLY (not exact)
            elif attn == "none": o = q.contiguous()                                              # TIMING DIAGNOSTIC ONLY (not exact)
            else: o = attn_rolled_torch(q, kb, vb, meta, NH, HD)
            ao = ops.rms(F.linear(o, Lw["o"]), Lw["n2"])
            h, res = ops.add_rms(ao, res, Lw["n3"])
            m = ops.rms(F.linear(ops.silu_mul(F.linear(h, Lw["gu"])), Lw["down"]), Lw["n4"])
            if li + 1 < c.NL: h, res = ops.add_rms(m, res, self.layers[li + 1]["n1"])
            else: res = res + m
        return ops.rms(res, self.norm)


# ------------------------------------------------------------------------------------------------------------- attention
def attn_rolled_torch(q, kb, vb, meta, NH, HD):
    """reference: row r sees combined slots [lo_r, aend_r) U [b0_r, b0_r + bl_r) (lo_r = its region start)"""
    N = q.shape[0]; acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    K = kb.view(-1, NH, HD).to(acc); V = vb.view(-1, NH, HD).to(acc); NK = K.shape[0]
    col = torch.arange(NK, device=q.device)[None]
    vis = ((col >= meta.lo[:, None]) & (col < meta.aend[:, None])) | ((col >= meta.b0[:, None]) & (col < (meta.b0 + meta.bl)[:, None]))
    sc = torch.einsum("qhd,khd->hqk", q.view(N, NH, HD).to(acc), K) * HD ** -0.5
    sc = sc.masked_fill(~vis[None], float("-inf"))
    p = torch.softmax(sc, -1)
    p = torch.nan_to_num(p, nan=0.0)                                                   # pad rows with no visible key (CPU only)
    return torch.einsum("hqk,khd->qhd", p, V).to(q.dtype).reshape(N, NH * HD)


_WROWS = None
def get_win_rows_kernel():
    """Triton: per row r, attend combined slots [a0_r, a0_r + al_r) U [b0_r, b0_r + bl_r) and merge with the prefix partial (o1, lse)"""
    global _WROWS
    if _WROWS is not None: return _WROWS
    import triton, triton.language as tl

    @triton.jit
    def _win_rows(Q, KB, VB, O1, LSE, A0, AL, B0, BL, OUT, s_q, s_o1, s_lh, s_lr, s_out, scale,
                  H: tl.constexpr, HD: tl.constexpr, BN: tl.constexpr):
        r = tl.program_id(0); h = tl.program_id(1)
        rd = tl.arange(0, HD)
        q = tl.load(Q + r * s_q + h * HD + rd).to(tl.float32)
        m_i = tl.load(LSE + h * s_lh + r * s_lr).to(tl.float32)
        l_i = m_i * 0.0 + 1.0
        acc = tl.load(O1 + r * s_o1 + h * HD + rd).to(tl.float32)
        a0 = tl.load(A0 + r).to(tl.int64); al = tl.load(AL + r).to(tl.int64)
        b0 = tl.load(B0 + r).to(tl.int64); bl = tl.load(BL + r).to(tl.int64)
        tot = al + bl
        for n0 in range(0, tot, BN):
            j = n0 + tl.arange(0, BN); valid = j < tot
            idx = tl.where(j < al, a0 + j, b0 + j - al)
            idx = tl.where(valid, idx, 0)
            k = tl.load(KB + idx[:, None] * H + h * HD + rd[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            s = tl.sum(k * q[None, :], 1) * scale
            s = tl.where(valid, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 0))
            alpha = tl.exp(m_i - m_new); p = tl.exp(s - m_new)
            l_i = l_i * alpha + tl.sum(p, 0)
            v = tl.load(VB + idx[:, None] * H + h * HD + rd[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, 0)
            m_i = m_new
        tl.store(OUT + r * s_out + h * HD + rd, (acc / l_i).to(OUT.dtype.element_ty))

    def run(q, kb2, vb2, o1, lse, a0, al, b0, bl, out, scale, NH, HD):
        N = q.shape[0]
        _win_rows[(N, NH)](q, kb2, vb2, o1, lse, a0, al, b0, bl, out, q.stride(0), o1.stride(0), lse.stride(0), lse.stride(1),
                           out.stride(0), scale, H=NH * HD, HD=HD, BN=16, num_warps=4)
        return out
    _WROWS = run
    return run


def _lse_nh_n(lse, NH, N):
    """FA3 LSE -> [NH, N] view (varlen returns [NH, total_q]; batched [1, NH, N])"""
    if lse.dim() == 3: lse = lse.view(NH, N) if lse.shape[0] == 1 else lse.reshape(NH, N)
    assert lse.shape == (NH, N), lse.shape
    return lse


def attn_rolled_split(q, kb, vb, meta, NH, HD, eng):
    FA = E.get_fa3(); N = q.shape[0]; H = NH * HD
    r = FA(q.contiguous().view(N, NH, HD), kb, vb, cache_seqlens=meta.lcT, page_table=meta.ptab, cu_seqlens_q=meta.cu,
           max_seqlen_q=meta.max_q, causal=False, return_softmax_lse=True)
    o1, lse = r[0], _lse_nh_n(r[1], NH, N)
    out = torch.empty(N, H, device=q.device, dtype=q.dtype)
    get_win_rows_kernel()(q, kb.view(-1, H), vb.view(-1, H), o1.reshape(N, H), lse, meta.a0, meta.al, meta.b0, meta.bl, out, HD ** -0.5, NH, HD)
    return out


def attn_rolled_fa3only(q, kb, vb, meta, NH, HD, eng):
    """timing diagnostic: the prefix FA3 call of `split` without the window kernel (wrong output)"""
    FA = E.get_fa3(); N = q.shape[0]
    r = FA(q.contiguous().view(N, NH, HD), kb, vb, cache_seqlens=meta.lcT, page_table=meta.ptab, cu_seqlens_q=meta.cu,
           max_seqlen_q=meta.max_q, causal=False, return_softmax_lse=True)
    return r[0].reshape(N, NH * HD)


def attn_rolled_split4(q, kb, vb, meta, NH, HD, eng):
    FA = E.get_fa3(); N = q.shape[0]; H = NH * HD; S, SL = eng.S, eng.SL; T = eng.c.T
    os_, ls_ = [], []
    for t in range(T):
        r = FA(q.contiguous().view(1, N, NH, HD), kb[t * SL : t * SL + S].view(1, S, NH, HD), vb[t * SL : t * SL + S].view(1, S, NH, HD),
               cache_seqlens=meta.lc1, causal=False, return_softmax_lse=True)
        os_.append(r[0].reshape(N, H)); ls_.append(_lse_nh_n(r[1], NH, N))
    O = torch.stack(os_); L = torch.stack(ls_)
    o1 = O.gather(0, meta.loop[None, :, None].expand(1, N, H))[0]
    lse = L.gather(0, meta.loop[None, None, :].expand(1, NH, N))[0].contiguous()
    out = torch.empty(N, H, device=q.device, dtype=q.dtype)
    get_win_rows_kernel()(q, kb.view(-1, H), vb.view(-1, H), o1.contiguous(), lse, meta.a0, meta.al, meta.b0, meta.bl, out, HD ** -0.5, NH, HD)
    return out


# ------------------------------------------------------------------------------------------------------------- decoder
class Col:
    __slots__ = ("p", "tok", "conf", "loop", "last", "ro", "blk", "mg")

    def __init__(self, p, tok, conf, blk):
        self.p, self.tok, self.conf, self.blk = p, tok, conf, blk
        self.loop, self.last, self.ro, self.mg = 0, None, None, None


class Rolled:
    """rolled lattice decoder, see module docstring. B proposals per block, proposals enter after e passes, verify at pass T."""
    NCOL = 8   # int64 meta columns per row: tok, sidx, loop, pos, wslot, aend, b0, bl

    def __init__(self, eng, B, nv, eos, mask_id, e=2, attn="split", graphs=None, margins=False, bucket=16, max_rows=1500, check=False):
        assert 1 <= B <= eng.Bmax and 1 <= e <= eng.c.T
        self.e_, self.B, self.nv, self.eos, self.mask_id, self.e, self.margins = eng, B, nv, set(eos), mask_id, e, margins
        self.T = T = eng.c.T; dev = eng.dev; self.attn = attn; self.max_rows = max_rows; self.check = check
        self.graphs_on = str(dev).startswith("cuda") if graphs is None else graphs
        self.cuda = str(dev).startswith("cuda")
        D = B + 1; self.D = D
        nmax = -(-self.T // e) * D * D + D                                     # <= ceil(T/e) blocks in flight (+ slack)
        self.bucket = bucket; self.buckets = list(range(bucket, -(-nmax // bucket) * bucket + 1, bucket)); self.Nmax = self.buckets[-1]
        M = eng.M; self.M = M; self.NS = M * D + 1; self.SDUM = M * D
        self.mt = torch.zeros(self.Nmax, self.NCOL, dtype=torch.int64, device=dev)
        self.hdr = torch.zeros(T + 2, dtype=torch.int64, device=dev)            # [Lc, cu_0 .. cu_T]
        self.out = torch.zeros(4, self.Nmax, dtype=torch.int64, device=dev)
        self.outm = torch.zeros(2, self.Nmax, device=dev)
        self.Hs = torch.zeros(self.NS, eng.c.H, device=dev, dtype=eng.dt)
        self.Ex = torch.zeros(self.NS, nv, device=dev, dtype=torch.float32); self.Rem = torch.ones(self.NS, device=dev, dtype=torch.float32)
        self.ptab = (torch.arange(T, device=dev)[:, None] * eng.SL + torch.arange(eng.S, device=dev)[None]).to(torch.int32).contiguous()
        if self.cuda:
            self.h_mt = torch.zeros(self.Nmax, self.NCOL, dtype=torch.int64).pin_memory(); self.h_hdr = torch.zeros(T + 2, dtype=torch.int64).pin_memory()
            self.h_out = torch.zeros(2, self.Nmax, dtype=torch.int64).pin_memory(); self.h_m = torch.zeros(2, self.Nmax).pin_memory()
            self.ev = torch.cuda.Event()
        self.O = np.arange(D)
        self.gr = {}
        self._fill_pads(0, self.Nmax)

    def _fill_pads(self, n, Nb):
        e = self.e_; pad = torch.tensor([self.mask_id, self.SDUM, self.T - 1, 0, e.DUMMY, 0, 0, 0], dtype=torch.int64)
        self.mt[n:Nb] = pad.to(self.mt.device)

    # ---- device side of one call (captured per bucket) ----
    def _meta(self, Nb):
        e = self.e_; T = self.T; mt = self.mt[:Nb]
        tok, sidx, loop, pos, wslot, aend, b0, bl = mt.unbind(1)
        lc = self.hdr[0]
        tok, sidx, loop, pos, wslot, aend, b0, bl = (x.contiguous() for x in (tok, sidx, loop, pos, wslot, aend, b0, bl))   # columns of mt are strided
        m = E.Meta(); m.tok = tok; m.sidx = sidx; m.pos = pos; m.wslot = wslot; m.loop = loop
        m.lo = loop * e.SL; m.aend = aend; m.b0 = b0; m.bl = bl
        m.a0 = m.lo + lc; m.al = (aend - m.lo - lc).clamp(min=0)
        m.cu = self.hdr[1 : T + 2].to(torch.int32); m.lcT = lc.expand(T).to(torch.int32).contiguous(); m.lc1 = lc.view(1).to(torch.int32)
        m.ptab = self.ptab; m.max_q = Nb
        return m

    def _pass(self, Nb):
        e = self.e_; T = self.T; m = self._meta(Nb); tok, sidx, loop = m.tok, m.sidx, m.loop
        X = torch.where((loop == 0)[:, None], F.embedding(tok, e.embed), self.Hs.index_select(0, sidx))
        hn = e.rolled_pass(X, m, self.attn)
        self.Hs.index_copy_(0, sidx, hn)
        lam = e.gate(hn)
        rem_in = torch.where(loop == 0, torch.ones_like(self.Rem[:1]), self.Rem.index_select(0, sidx))
        pt = torch.where(loop < T - 1, lam * rem_in, rem_in)
        self.Rem.index_copy_(0, sidx, (rem_in * (1 - lam)).to(self.Rem.dtype))
        lg = F.linear(hn, e.lm_head)[:, : self.nv].float()
        prev = torch.where((loop == 0)[:, None], torch.zeros_like(self.Ex[:1]), self.Ex.index_select(0, sidx))
        ex = (prev + lg * pt[:, None]).float()
        self.Ex.index_copy_(0, sidx, ex)
        self.out[0, :Nb].copy_(lg.argmax(-1)); self.out[1, :Nb].copy_(ex.argmax(-1))
        if self.margins: self.outm[:, :Nb].copy_(ex.topk(2, -1).values.t())

    def capture(self):
        if not self.graphs_on: return
        pool = torch.cuda.graph_pool_handle()
        for Nb in self.buckets:
            self._fill_pads(0, Nb); self.hdr.zero_(); self.hdr[self.T + 1] = Nb; self.hdr[0] = 1
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2): self._pass(Nb)
            torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool): self._pass(Nb)
            torch.cuda.synchronize(); self.gr[Nb] = g

    # ---- host: build the call's rows ----
    def _rows(self, act, lc):
        """act: active columns (loop < T). Returns (meta array [n, NCOL], cu list, column order)"""
        e = self.e_; B, D, T, SL, S, M = self.B, self.D, self.T, e.SL, e.S, self.M
        order = sorted(range(len(act)), key=lambda k: act[k].loop)              # stable: position order within a pass index
        cols = [act[k] for k in order]
        P = np.array([c.p for c in cols], dtype=np.int64); Lp = np.array([c.loop for c in cols], dtype=np.int64)
        TK = np.array([c.tok for c in cols], dtype=np.int64); O = self.O[None]
        ci = P % M; base = Lp * SL
        a = np.empty((len(cols), D, self.NCOL), dtype=np.int64)
        a[:, :, 0] = self.mask_id; a[:, 0, 0] = TK
        a[:, :, 1] = ci[:, None] * D + O
        a[:, :, 2] = Lp[:, None]
        a[:, :, 3] = P[:, None] + O
        a[:, :, 4] = base[:, None] + np.where(O == 0, P[:, None], S + ci[:, None] * B + O - 1)
        a[:, :, 5] = (base + P + 1)[:, None]
        a[:, :, 6] = (base + S + ci * B)[:, None]
        a[:, :, 7] = O
        cnt = np.bincount(Lp, minlength=T) * D
        return a.reshape(-1, self.NCOL), np.concatenate([[0], np.cumsum(cnt)]), cols

    def call(self, act, lc):
        rows, cu, cols = self._rows(act, lc); n = rows.shape[0]
        Nb = next(b for b in self.buckets if b >= n)
        cu = cu.copy(); cu[-1] = Nb                                             # pad rows sit in the last pass group
        if self.check: self._check_rows(cols, lc)
        if self.graphs_on:
            self.h_mt[:n].copy_(torch.from_numpy(rows))
            if Nb > n: self.h_mt[n:Nb] = torch.tensor([self.mask_id, self.SDUM, self.T - 1, 0, self.e_.DUMMY, 0, 0, 0])
            self.h_hdr[0] = lc; self.h_hdr[1:].copy_(torch.from_numpy(cu))
            self.mt[:Nb].copy_(self.h_mt[:Nb], non_blocking=True); self.hdr.copy_(self.h_hdr, non_blocking=True)
            self.gr[Nb].replay()
            self.h_out[:, :Nb].copy_(self.out[:2, :Nb], non_blocking=True)
            if self.margins: self.h_m[:, :Nb].copy_(self.outm[:, :Nb], non_blocking=True)
            self.ev.record(); self.ev.synchronize()
            o = self.h_out[:, :n].numpy().copy(); mg = (self.h_m[0, :n] - self.h_m[1, :n]).numpy().copy() if self.margins else None
        else:
            self.mt[:n].copy_(torch.from_numpy(rows)); self._fill_pads(n, Nb)
            self.hdr[0] = lc; self.hdr[1:].copy_(torch.from_numpy(cu))
            self._pass(Nb)
            o = self.out[:2, :n].cpu().numpy().copy(); mg = (self.outm[0, :n] - self.outm[1, :n]).cpu().numpy().copy() if self.margins else None
        return o, mg, cols, n, Nb

    def _check_rows(self, cols, lc):
        """debug: every row a query sees has done at least as many passes (the reference's exactness condition)"""
        bypos = {c.p: c for c in self.cols}
        for c in cols:
            for q in range(lc, c.p + 1): assert bypos[q].loop >= c.loop, (q, bypos[q].loop, c.p, c.loop)

    # ---- host: the Wavefront schedule (kv = T) ----
    def _new(self, p, tok, conf, blk):
        c = Col(p, tok, conf, blk); self.cols.append(c); return c

    def _enter_block(self, R):
        T = self.T; done_R = R.loop == T
        first = int(R.ro[0]) if done_R else int(R.last[0]); conf = R.conf and done_R
        self.nblk += 1; b = self.nblk; nd = 0 if conf else 1
        self._new(R.p + 1, first, conf, b)
        if conf: self.known.append(first)
        for i in range(1, self.B + 1):
            t = int(R.ro[i]) if done_R else int(R.last[i]); self._new(R.p + 1 + i, t, False, b); nd += 1
        self.blocks[b] = [nd, 0]

    def _flush_after(self, pos):
        keep = [c for c in self.cols if c.p < pos]; self.flushed += (len(self.cols) - len(keep)) * self.D; self.cols = keep

    @torch.no_grad()
    def decode(self, prompt_ids, max_new, stop_fn=None, prefill_attn="auto"):
        e, B, T = self.e_, self.B, self.T; P0 = len(prompt_ids)
        assert P0 + max_new + 4 * self.D + 16 <= e.S, ("cache too small", P0, max_new, e.S)
        t0 = time.perf_counter()
        first, m0 = e.prefill(list(prompt_ids), self.nv, attn="torch" if self.attn == "torch" else prefill_attn)
        if self.cuda: torch.cuda.synchronize()
        t_pref = time.perf_counter() - t0
        self.known = list(prompt_ids) + [first]; lc = P0; self.cols = []; self.flushed = 0; self.nblk = 0; self.blocks = {}
        self._new(P0, first, True, 0)
        done = first in self.eos
        log, call_ms, rows_l, nb_l = [], [], [], []; self.mgd = {P0 - 1: m0}
        while not done:
            ts = time.perf_counter()
            act = [c for c in self.cols if c.loop < T]; assert act
            o, mg, cols, n, Nb = self.call(act, lc)
            k = 0
            for c in cols:                                                     # one row block of D per column, in call order
                c.last = o[0, k : k + self.D]; c.loop += 1
                if c.loop == T:
                    c.ro = o[1, k : k + self.D]
                    if mg is not None: self.mgd[c.p] = float(mg[k])
                k += self.D
            n0 = len(self.known)
            cs = self.cols; j = 0
            while j < len(cs):                                                 # verification walk
                R = cs[j]
                if not (R.conf and R.loop == T): break
                nxt = cs[j + 1] if j + 1 < len(cs) else None
                if nxt is None: self._enter_block(R); break
                if nxt.conf: j += 1; continue
                if int(R.ro[0]) == nxt.tok:
                    nxt.conf = True; self.known.append(nxt.tok); self.blocks[nxt.blk][1] += 1
                else:
                    self._flush_after(nxt.p); self._enter_block(R); break
                if self.known[-1] in self.eos: break
                j += 1
            if any(t in self.eos for t in self.known[n0:]): done = True
            log.append(len(self.known) - n0)
            if stop_fn is not None and not done and len(self.known) > n0 and stop_fn(self.known[P0:]): done = True
            if not done and len(self.known) - P0 < max_new:
                newest = self.cols[-1]                                         # speculation
                if newest.loop >= self.e and not (newest.conf and newest.loop == T) and len(self.cols) * self.D < self.max_rows:
                    self._enter_block(newest)
                cs = self.cols                                                 # fold
                while len(cs) >= 2 and cs[0].conf and cs[0].loop >= T and cs[1].conf and cs[0].p == lc:
                    cs.pop(0); lc += 1
                assert self.cols[-1].p - lc < self.M, ("column ring too small", self.cols[-1].p - lc, self.M)
            else: done = True
            te = time.perf_counter(); call_ms.append(1000 * (te - ts)); rows_l.append(n); nb_l.append(Nb)
        gen = self.known[P0:]
        kk = next((i for i, t in enumerate(gen) if t in self.eos), len(gen))
        hist = [0, 0, 0]                                                       # blocks with drafts: none / some / all confirmed
        for nd, na in self.blocks.values():
            if nd: hist[0 if na == 0 else (2 if na >= nd else 1)] += 1
        margins = [self.mgd.get(P0 + i - 1) for i in range(len(gen))] if self.margins else None   # token at P0+i: readout of slot P0+i-1
        return dict(gen=gen[:kk], gen_raw=gen, log=log, calls=len(log), call_ms=call_ms, rows=rows_l, padded=nb_l, flushed=self.flushed,
                    prefill_ms=1000 * t_pref, block_hist=hist, nblocks=self.nblk, margins=margins)


# ------------------------------------------------------------------------------------------------------------- checks / bench
def synth_cols(dec, Lc, layout):
    """synthetic column run starting at Lc: layout = list of (n_columns, pass index) in position order"""
    cols = []; p = Lc
    for n, lp in layout:
        for _ in range(n): c = Col(p, 5 + p % 7, True, 0); c.loop = lp; cols.append(c); p += 1
    return cols


def load_rows(dec, cols, Lc):
    """write the call's meta for the active columns into the decoder's device buffers; returns (n rows, bucket)"""
    act = [c for c in cols if c.loop < dec.T]
    rows, cu, _ = dec._rows(act, Lc); n = rows.shape[0]; Nb = next(b for b in dec.buckets if b >= n); cu = cu.copy(); cu[-1] = Nb
    dec.mt[:n].copy_(torch.from_numpy(rows).to(dec.mt.device)); dec._fill_pads(n, Nb)
    dec.hdr[0] = Lc; dec.hdr[1:].copy_(torch.from_numpy(cu).to(dec.hdr.device))
    return n, Nb


@torch.no_grad()
def check_rolled_attention(eng, B, attn, Lc=137, seed=0, log=print):
    """one layer of one call with random caches/rows: rolled attention `attn` vs the torch reference (max abs error over real rows).
    Layout: 2 finished columns, B+1 columns at pass 2, B+1 at pass 0 (pass groups 1 and 3 empty, pad rows in group 3)."""
    torch.manual_seed(seed); c = eng.c; H = c.H
    eng.kbuf[0].normal_(); eng.vbuf[0].normal_()
    dec = Rolled(eng, B, nv=8, eos=[0], mask_id=1, e=2, attn=attn, graphs=False)
    D = B + 1; n, Nb = load_rows(dec, synth_cols(dec, Lc, [(2, c.T), (D, 2), (D, 0)]), Lc)
    m = dec._meta(Nb)
    qkv = torch.randn(Nb, 3 * H, device=eng.dev, dtype=eng.dt); q, k, v = qkv[:, :H], qkv[:, H : 2 * H], qkv[:, 2 * H :]
    eng.kbuf[0].view(-1, H).index_copy_(0, m.wslot, k); eng.vbuf[0].view(-1, H).index_copy_(0, m.wslot, v)
    ref = attn_rolled_torch(q, eng.kbuf[0], eng.vbuf[0], m, c.NH, c.HD)
    out = (attn_rolled_split if attn == "split" else attn_rolled_split4)(q, eng.kbuf[0], eng.vbuf[0], m, c.NH, c.HD, eng)
    err = (out[:n].float() - ref[:n].float()).abs().max().item()
    log(f"[check] rolled attention {attn} B={B} rows={n} (bucket {Nb}) Lc={Lc}: max abs err vs torch {err:.5f}")
    return err


@torch.no_grad()
def bench_call(dec, Lc, layout, reps=50):
    """GPU ms of one rolled call (graph replay of the bucket) for a synthetic layout at prefix length Lc (CUDA events, median)"""
    n, Nb = load_rows(dec, synth_cols(dec, Lc, layout), Lc)
    g = dec.gr[Nb]
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); ev = [torch.cuda.Event(enable_timing=True) for _ in range(reps + 1)]; ev[0].record()
    for i in range(reps): g.replay(); ev[i + 1].record()
    torch.cuda.synchronize(); ts = sorted(ev[i].elapsed_time(ev[i + 1]) for i in range(reps))
    return n, Nb, ts[len(ts) // 2]
