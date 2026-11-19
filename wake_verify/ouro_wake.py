"""ouro_wake.py -- Optimistic Late-Entry Block Decoding with Wake Verification on Ouro (T = 4).

Key Principle (Wake Verification):
  In K-pass late-entry decoding (1 <= K < 4), newly committed token groups advance the frontier
  optimistically using only passes (5 - K .. 4), while `NG = ceil(4 / K)` in-flight background
  groups execute exact causal passes `1 -> 4` across subsequent steps.
  When the oldest active group `g = NG` completes pass 4, its accumulated stick-breaking mixture
  over passes `1..4` IS mathematically identical to sequential full-depth (T = 4) decoding, so
  evaluating `lm_head` on those already-computed background rows inside the same single GEMM yields
  an exact full-depth verification check in the wake at zero extra forward calls:
    - `tau = 0.0` (Strict Exact Wake Verification): rolls back whenever `y_wake != y_committed`,
      plus drains the `<= NG` in-flight wake groups at EOS/max_new, guaranteeing 100% bit-for-bit
      equivalence to full-depth T = 4 greedy decoding in exact arithmetic.
    - `tau > 0.0` (Margin-Gated Wake Verification): rolls back iff `y_wake != y_committed` AND
      `z_wake(y_wake) - z_wake(y_committed) > tau`, filtering benign near-ties while catching
      high-margin reasoning/arithmetic deviations.
    - `tau = inf`: zero rollbacks (recovers standard lossy late-entry block decoding).

Mode syntax:
  - `wk:<K>:<shape>:<tau>`        (e.g. `wk:2:latt:8:0`, `wk:2:latt:8:0.25`, `wk:3:latt:8:0`)
  - `wks:<K>:<shape>:<tau>`       (with zero-parameter suffix-ngram draft lookup + MARS fallback)
  - `sft_ar:<K>`                  (sequential K-pass early exit, for controls)
  - Plus all standard `ar`, `latt:B`, `lk:K:shape`, and `lks:K:shape` modes.
"""

import math
import os
import sys
import time
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEK_DIR = os.path.join(REPO_ROOT, "decode")
if LATEK_DIR not in sys.path:
    sys.path.insert(0, LATEK_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import engine as E
import sparse as SP
import late as L2
import latek as LK
from wake_verify.suffix_draft import suffix_propose


def parse_wake_mode(mode: str):
    """Parse `wk[s]:<K>:<shape>:<tau>` or `lks:<K>:<shape>`."""
    if mode.startswith("lks:"):
        _, K_s, shape = mode.split(":", 2)
        K = int(K_s)
        B, sets = L2.parse_shape(shape)
        return dict(kind="wake", K=K, B=B, sets=sets, shape=shape, tau=float("inf"), suffix=True)
    assert mode.startswith("wk:") or mode.startswith("wks:"), f"Invalid wake mode: {mode}"
    has_sfx = mode.startswith("wks:")
    rest = mode.split(":", 1)[1]
    K_s, tail = rest.split(":", 1)
    K = int(K_s)
    shape, tau_s = tail.rsplit(":", 1)
    tau = float("inf") if tau_s in ("inf", "none") else float(tau_s)
    B, sets = L2.parse_shape(shape)
    return dict(kind="wake", K=K, B=B, sets=sets, shape=shape, tau=tau, suffix=has_sfx)


class WakeKLattice:
    """CUDA-graphed K-pass Late-Entry Lattice with Wake Verification and optional Suffix Lookup."""

    def __init__(self, eng, K: int, B: int, sets, nv: int, eos, mask_id: int, tau: float = 0.0,
                 suffix: bool = False, attn: str = "paged", graphs=None):
        assert eng.c.T == 4, "Ouro WakeKLattice requires T=4"
        assert 1 <= K <= 4
        self.e = eng
        self.K = int(K)
        self.B = int(B)
        self.nv = int(nv)
        self.eos = set(int(x) for x in eos)
        self.mask_id = int(mask_id)
        self.tau = float(tau)
        self.suffix = bool(suffix)
        self.wake_on = (self.K < 4) and (not math.isinf(self.tau))

        self.Ep = 5 - self.K
        self.bg = self.K < 4
        self.NG = LK.n_groups(self.K, self.bg)

        c = eng.c
        dev = eng.dev
        S, SL = eng.S, eng.SL
        self.S, self.SL = S, SL
        self.attn = attn if attn in ("paged", "torch") else "paged"
        self.graphs_on = str(dev).startswith("cuda") if graphs is None else bool(graphs)

        D = self.B + 1
        self.D = D
        assert (self.B == 0 and not sets) or (self.B >= 1 and sets and sets[0][0] == 0), (self.B, sets)
        self.sets = list(sets)
        self.set_a = [a for a, _ in sets]
        self.set_s = [s for _, s in sets]
        offs = []
        o = D
        for _, s in sets:
            offs.append(o)
            o += s
        self.set_off = offs
        NB = o
        self.NB = NB
        assert NB - D <= eng.M * eng.Bmax, ("placeholder area too small", NB - D, eng.M * eng.Bmax)
        self.astar = [max(a for a in self.set_a if a <= cc) for cc in range(self.B + 1)] if sets else [None]
        self.kidx = {a: k for k, a in enumerate(self.set_a)}

        relpos = list(range(D)) + [a + i for a, s in sets for i in range(1, s + 1)]
        tails = [list(range(D))] + [list(range(a + 1)) + [offs[k] + i for i in range(s)] for k, (a, s) in enumerate(sets)]
        cub = [0, D] + [offs[k] + s for k, (a, s) in enumerate(sets)]
        self.nblk = len(tails)
        TM = max(len(x) for x in tails)
        tp = torch.zeros(self.nblk, TM, dtype=torch.int64)
        for sq, tl_ in enumerate(tails):
            tp[sq, : len(tl_)] = torch.tensor(tl_)

        self.tails = tails
        self.relpos = torch.tensor(relpos, device=dev)
        self.tailpad = tp.to(dev)
        self.lens = torch.tensor([len(x) for x in tails], device=dev)
        rr = torch.arange(NB, device=dev)
        self.is_tok = rr < D
        self.ph_slot = S + rr - D
        self.col = torch.arange(S, device=dev)[None]
        self.ar = torch.arange(D, device=dev)
        self.rr = rr

        # Per-call plans
        self.calls = []
        for kk in range(self.K):
            cl = LK._Call()
            cl.kk = kk
            cl.kp = self.Ep + kk
            cl.act = LK.active_groups(self.K, self.NG, kk)
            cl.gpass = [self.K * (g - 1) + kk + 1 for g in cl.act]
            nA = len(cl.act)
            cl.G = nA * D
            cl.N = cl.G + NB
            cl.nseq = nA + self.nblk
            cu = [D * i for i in range(nA + 1)] + [cl.G + x for x in cub[1:]]
            m = E.Meta()
            m.cu_q = torch.tensor(cu, device=dev, dtype=torch.int32)
            m.max_q = max(cu[i + 1] - cu[i] for i in range(cl.nseq))
            m.pos = torch.zeros(cl.N, dtype=torch.int64, device=dev)
            m.wslot = torch.zeros(cl.N, dtype=torch.int64, device=dev)
            m.seqlens = torch.zeros(cl.nseq, dtype=torch.int32, device=dev)
            m.page_table = torch.zeros(cl.nseq, S, dtype=torch.int32, device=dev)
            cl.meta = m
            cl.hidx = torch.zeros(max(cl.G, 1), dtype=torch.int64, device=dev)
            cl.w_ridx = torch.zeros(max(cl.G, 1), dtype=torch.int64, device=dev)
            self.calls.append(cl)
        self.N_max = max(cl.N for cl in self.calls)

        # Static device buffers
        self.GI = D if self.NG else 0
        self.hdr = torch.zeros(self.NG + 1, dtype=torch.int64, device=dev)
        self.ids = torch.full((self.GI + NB,), self.mask_id, dtype=torch.int64, device=dev)
        self.comm_next = torch.zeros(D, dtype=torch.int64, device=dev)

        self.Xb = torch.zeros(NB, c.H, device=dev, dtype=eng.dt)
        self.Hs = torch.zeros(S + 1, c.H, device=dev, dtype=eng.dt)
        self.HSD = S

        # Ring buffer for stick-breaking background logit accumulation across passes 1..4
        # Active in-flight positions span `< NG * D`, so `R_W >= 2 * (NG + 2) * D` is collision-free;
        # pad rows use distinct dummy indices `R_W .. R_W + cl.G - 1` so index_copy_ never has duplicate keys.
        self.R_W = max(64, 2 * (self.NG + 2) * D)
        max_G = max((cl.G for cl in self.calls), default=D)
        w_tot = self.R_W + max(max_G, D) + 4
        acc_dt = torch.float64 if eng.dt == torch.float64 else torch.float32
        self.acc_dt = acc_dt
        self.w_rem = torch.ones(w_tot, device=dev, dtype=acc_dt)
        self.w_exp = torch.zeros(w_tot, self.nv, device=dev, dtype=acc_dt)
        self.wake_top = torch.zeros(D, dtype=torch.int64, device=dev)
        self.wake_margin = torch.zeros(D, dtype=torch.float32, device=dev)

        self.rem = torch.ones(NB, device=dev, dtype=acc_dt)
        self.exp = torch.zeros(NB, self.nv, device=dev, dtype=acc_dt)
        self.top = torch.zeros(NB, dtype=torch.int64, device=dev)

        if str(dev).startswith("cuda"):
            self.h_top = torch.zeros(NB, dtype=torch.int64).pin_memory()
            self.h_wtop = torch.zeros(D, dtype=torch.int64).pin_memory()
            self.h_wmarg = torch.zeros(D, dtype=torch.float32).pin_memory()
            self.h_ids = torch.zeros(self.GI + NB, dtype=torch.int64).pin_memory()
            self.h_cnext = torch.zeros(D, dtype=torch.int64).pin_memory()
            self.h_hdr = torch.zeros(self.NG + 1, dtype=torch.int64).pin_memory()
            self.ev = torch.cuda.Event()
        self.graphs = None
        self._graph_pool = None

    @property
    def N(self):
        return self.N_max

    @property
    def nseq(self):
        return max(cl.nseq for cl in self.calls)

    def _prep(self, kk: int):
        e, D, S, SL = self.e, self.D, self.S, self.SL
        cl = self.calls[kk]
        m = cl.meta
        rb = cl.kp - 1
        lc = self.hdr[0]
        col = self.col
        DUM = e.DUMMY
        pos_l, w_l, h_l, wr_l, pt_l, sl_l = [], [], [], [], [], []
        for gi, (g, pj) in enumerate(zip(cl.act, cl.gpass)):
            hi, lo = self.hdr[g - 1], self.hdr[g]
            reg = pj - 1
            posg_raw = hi - D + self.ar
            real = self.ar >= D - (hi - lo)
            posg = posg_raw.clamp(min=0)
            pos_l.append(posg)
            w_l.append(torch.where(real, reg * SL + posg, torch.full_like(posg, DUM)))
            h_l.append(torch.where(real, posg, torch.full_like(posg, self.HSD)))
            wr_l.append(torch.where(real, torch.remainder(posg, self.R_W), self.R_W + gi * D + self.ar))
            pt_l.append(torch.where(col < hi, reg * SL + col, torch.zeros_like(col)))
            sl_l.append(hi.view(1))

        wslot_loc = torch.where(self.is_tok, lc + self.rr, self.ph_slot)
        pos_l.append(lc + self.relpos)
        w_l.append(rb * SL + wslot_loc)
        rel = col - lc
        tail = self.tailpad.gather(1, rel.clamp(0, self.tailpad.shape[1] - 1).expand(self.nblk, -1))
        ptb = torch.where(rel < 0, rb * SL + col, rb * SL + wslot_loc[tail])
        ptb = torch.where(rel >= self.lens[:, None], torch.zeros_like(ptb), ptb)
        pt_l.append(ptb)
        sl_l.append(lc + self.lens)

        m.pos.copy_(torch.cat(pos_l))
        m.wslot.copy_(torch.cat(w_l))
        if cl.G:
            cl.hidx.copy_(torch.cat(h_l))
            cl.w_ridx.copy_(torch.cat(wr_l))
        m.page_table.copy_(torch.cat(pt_l).to(torch.int32))
        m.seqlens.copy_(torch.cat(sl_l).to(torch.int32))

    def _pass(self, kk: int):
        e, D = self.e, self.D
        cl = self.calls[kk]
        self._prep(kk)
        xs = []
        for gi, pj in enumerate(cl.gpass):
            if pj == 1:
                xs.append(F.embedding(self.ids[: self.GI], e.embed))
            else:
                xs.append(self.Hs.index_select(0, cl.hidx[gi * D : (gi + 1) * D]))
        xs.append(F.embedding(self.ids[self.GI :], e.embed) if kk == 0 else self.Xb)
        X = torch.cat(xs) if len(xs) > 1 else xs[0]
        hn = e.late_pass(X, cl.meta, self.attn)
        if cl.G:
            self.Hs.index_copy_(0, cl.hidx, hn[: cl.G])

        if self.wake_on and cl.G:
            # Unified gate + lm_head over `[background rows | block rows]` in a single GEMM
            lam_all = e.gate(hn).to(self.acc_dt)
            lg_all = F.linear(hn, e.lm_head)[:, : self.nv].to(self.acc_dt)
            for gi, pj in enumerate(cl.gpass):
                ridx = cl.w_ridx[gi * D : (gi + 1) * D]
                lam_g = lam_all[gi * D : (gi + 1) * D]
                lg_g = lg_all[gi * D : (gi + 1) * D]
                if pj == 1:
                    self.w_rem.index_copy_(0, ridx, 1.0 - lam_g)
                    self.w_exp.index_copy_(0, ridx, lg_g * lam_g[:, None])
                elif pj < 4:
                    rem_prev = self.w_rem.index_select(0, ridx)
                    pt_g = lam_g * rem_prev
                    self.w_rem.index_copy_(0, ridx, rem_prev * (1.0 - lam_g))
                    self.w_exp.index_copy_(0, ridx, self.w_exp.index_select(0, ridx) + lg_g * pt_g[:, None])
                else:
                    rem_prev = self.w_rem.index_select(0, ridx)
                    exp_fin = self.w_exp.index_select(0, ridx) + lg_g * rem_prev[:, None]
                    self.wake_top.copy_(exp_fin.argmax(-1))
                    w_max = exp_fin.max(-1).values
                    w_comm = exp_fin.gather(1, self.comm_next[:, None]).squeeze(1)
                    self.wake_margin.copy_((w_max - w_comm).float())
            hb = hn[cl.G :]
            lam = lam_all[cl.G :]
            lg = lg_all[cl.G :]
        else:
            hb = hn[cl.G :]
            lam = e.gate(hb).to(self.acc_dt)
            lg = F.linear(hb, e.lm_head)[:, : self.nv].to(self.acc_dt)

        if kk < self.K - 1:
            self.Xb.copy_(hb)
        if kk == 0:
            self.rem.fill_(1.0)
        if cl.kp < 4:
            pt = lam * self.rem
            self.rem.mul_(1.0 - lam)
        else:
            pt = self.rem
        if kk == 0:
            torch.mul(lg, pt[:, None], out=self.exp)
        else:
            self.exp.add_(lg * pt[:, None])
        if cl.kp == 4:
            self.top.copy_(self.exp.argmax(-1))

    def warm_hdr(self):
        return [(self.NG + 3 - i) * self.D for i in range(self.NG + 1)]

    def capture(self):
        if not self.graphs_on:
            return
        self.hdr.copy_(torch.tensor(self.warm_hdr(), device=self.hdr.device))
        self.ids.fill_(self.mask_id)
        self.comm_next.zero_()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                for kk in range(self.K):
                    self._pass(kk)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        pool = torch.cuda.graph_pool_handle()
        gs = []
        for kk in range(self.K):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                self._pass(kk)
            gs.append(g)
        torch.cuda.synchronize()
        self.graphs = gs
        self._graph_pool = pool

    def replay(self):
        for g in self.graphs:
            g.replay()

    def step(self, hist: list, ids_host: list, comm_next_host: list):
        if self.graphs is not None:
            for i, h in enumerate(hist):
                self.h_hdr[i] = h
            self.h_ids.copy_(torch.as_tensor(ids_host, dtype=torch.int64))
            if self.wake_on:
                self.h_cnext.copy_(torch.as_tensor(comm_next_host, dtype=torch.int64))
            self.hdr.copy_(self.h_hdr, non_blocking=True)
            self.ids.copy_(self.h_ids, non_blocking=True)
            if self.wake_on:
                self.comm_next.copy_(self.h_cnext, non_blocking=True)
            self.replay()
            self.h_top.copy_(self.top, non_blocking=True)
            if self.wake_on:
                self.h_wtop.copy_(self.wake_top, non_blocking=True)
                self.h_wmarg.copy_(self.wake_margin, non_blocking=True)
            self.ev.record()
            self.ev.synchronize()
            return (
                self.h_top.tolist(),
                self.h_wtop.tolist() if self.wake_on else None,
                self.h_wmarg.tolist() if self.wake_on else None,
            )

        self.hdr.copy_(torch.tensor(hist, dtype=torch.int64, device=self.hdr.device))
        self.ids.copy_(torch.as_tensor(ids_host, dtype=torch.int64, device=self.ids.device))
        if self.wake_on:
            self.comm_next.copy_(torch.as_tensor(comm_next_host, dtype=torch.int64, device=self.comm_next.device))
        for kk in range(self.K):
            self._pass(kk)
        return (
            self.top.tolist(),
            self.wake_top.tolist() if self.wake_on else None,
            self.wake_margin.tolist() if self.wake_on else None,
        )

    def next_props(self, top: list, c: int) -> list:
        if not self.sets:
            return []
        a = self.astar[c]
        k = self.kidx[a]
        off, s = self.set_off[k], self.set_s[k]
        i0 = c - a
        n = max(0, min(self.B, s - i0))
        return [int(top[off + i0 + j - 1]) for j in range(1, n + 1)]

    @torch.no_grad()
    def decode(self, prompt_ids: list, max_new: int, stop_fn=None, prefill_attn: str = "auto", prop_override_fn=None):
        e, B, D, NB = self.e, self.B, self.D, self.NB
        P0 = len(prompt_ids)
        assert P0 + max_new + NB + B + 8 <= e.S, ("cache too small", P0, max_new, NB, e.S)
        cuda = str(e.dev).startswith("cuda")
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        first, _ = e.prefill(list(prompt_ids), self.nv, attn="torch" if self.attn == "torch" else prefill_attn)
        if cuda:
            torch.cuda.synchronize()
        t_pref = time.perf_counter() - t0

        known = list(prompt_ids) + [first]
        hist = [P0] * (self.NG + 1)
        props = []
        done = (first in self.eos) or (max_new <= 1)

        log, nprop, cs, step_ms, gpu_ms = [], [], [], [], []
        n_acc = 0
        n_rollbacks = 0
        n_rolled_tokens = 0
        n_drain_steps = 0
        filler = [self.mask_id] * B
        ph_masks = [self.mask_id] * (NB - D)

        def _make_comm_next(hist_cur, known_cur):
            if not self.wake_on:
                return []
            hi_w, lo_w = hist_cur[self.NG - 1], hist_cur[self.NG]
            out = []
            for i in range(D):
                p = hi_w - D + i
                if p >= lo_w and (p + 1) < len(known_cur):
                    tv = int(known_cur[p + 1])
                    out.append(tv if 0 <= tv < self.nv else 0)
                else:
                    out.append(0)
            return out

        while True:
            # Determine whether we are actively generating or draining the wake at EOS/max_new
            need_drain = self.wake_on and done and (hist[self.NG] < hist[0])
            if done and not need_drain:
                break

            ts = time.perf_counter()
            lc = hist[0]

            if not done and self.suffix and B > 0:
                props = suffix_propose(known, B, fallback_props=props, stop_ids=self.eos)
            if not done and prop_override_fn is not None:
                ov = prop_override_fn(len(log), lc, known)
                if ov is not None:
                    props = list(ov)[:B]

            cur_props = [] if done else props
            ga = [known[p] if (p >= 0 and p >= hist[1]) else self.mask_id for p in range(lc - D, lc)] if self.NG else []
            cnext = _make_comm_next(hist, known)
            ids = ga + [known[-1]] + cur_props + filler[len(cur_props) :] + ph_masks

            top, wtop, wmarg = self.step(hist, ids, cnext)
            tg = time.perf_counter()

            # 1. Check Wake Verification on oldest group `[lo_w, hi_w)` FIRST
            rolled_back = False
            if self.wake_on:
                hi_w, lo_w = hist[self.NG - 1], hist[self.NG]
                if hi_w > lo_w:
                    for i in range(D - (hi_w - lo_w), D):
                        p = hi_w - D + i
                        y_true = int(wtop[i])
                        y_comm = int(known[p + 1])
                        if y_true != y_comm and (self.tau <= 0.0 or float(wmarg[i]) > self.tau):
                            # Rollback to exact verified token `y_true` at position `p + 1`!
                            n_rolled_tokens += len(known) - (p + 1)
                            n_rollbacks += 1
                            known = known[: p + 1] + [y_true]
                            lc_rb = p + 1
                            hist = [lc_rb] * (self.NG + 1)
                            props = []
                            done = (y_true in self.eos) or ((len(known) - P0) >= max_new)
                            if stop_fn is not None and not done and stop_fn(known[P0:]):
                                done = True
                            rolled_back = True
                            break

            if rolled_back:
                if need_drain:
                    n_drain_steps += 1
                log.append(0)
                nprop.append(len(cur_props))
                cs.append(0)
                te = time.perf_counter()
                step_ms.append(1000.0 * (te - ts))
                gpu_ms.append(1000.0 * (tg - ts))
                continue

            # 2. If this was an EOS/max_new drain step and wake passed, advance wake history only
            if done:
                n_drain_steps += 1
                hist = [lc] + hist[: self.NG]
                log.append(0)
                nprop.append(0)
                cs.append(0)
                te = time.perf_counter()
                step_ms.append(1000.0 * (te - ts))
                gpu_ms.append(1000.0 * (tg - ts))
                continue

            # 3. Normal optimistic block commitment
            np_ = len(cur_props)
            c = 0
            while c < np_ and int(top[c]) == cur_props[c]:
                c += 1
            n_acc += c
            commit = cur_props[:c] + [int(top[c])]

            emitted = 0
            for tk in commit:
                known.append(tk)
                emitted += 1
                if tk in self.eos:
                    done = True
                    break
                if (len(known) - P0) >= max_new:
                    done = True
                    break

            # Maintain exact invariant `len(known) == hist[0] + 1` even if `eos`/`max_new` truncated mid-block!
            lc_next = len(known) - 1
            props = [] if done else self.next_props(top, c)
            hist = [lc_next] + hist[: self.NG]
            if stop_fn is not None and not done and stop_fn(known[P0:]):
                done = True

            log.append(emitted)
            nprop.append(np_)
            cs.append(c)
            te = time.perf_counter()
            step_ms.append(1000.0 * (te - ts))
            gpu_ms.append(1000.0 * (tg - ts))

        gen_raw = known[P0:]
        gen = []
        for t in gen_raw:
            if t in self.eos:
                break
            gen.append(t)
        return dict(
            gen=gen,
            gen_raw=gen_raw,
            log=log,
            nprop=nprop,
            c=cs,
            steps=len(log),
            step_ms=step_ms,
            gpu_ms=gpu_ms,
            prefill_ms=1000.0 * t_pref,
            n_acc=n_acc,
            n_rollbacks=n_rollbacks,
            n_rolled_tokens=n_rolled_tokens,
            n_drain_steps=n_drain_steps,
        )
