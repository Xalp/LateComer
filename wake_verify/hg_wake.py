"""hg_wake.py -- CUDA-Graphed Paged FA3 Huginn Decoder with:
  1. Wake Verification (`tau = 0` exact lossless -> `tau > 0` margin-gated -> `tau = inf` lossy),
  2. Intra-model Recurrence Depth Scaling (`T` in {8, 16, 32, 64, ...}),
  3. Zero-parameter Suffix-Lookup + MARS Placeholder hybrid drafting (`suffix=True`).

Config string syntax (`parse_hg_wake_cfg`):
  - `blk:<K>:<B>` or `blk:<K>:<B>:<bg>`         (standard `HuginnFastDecoder` configs, at `--T`)
  - `eblk:<K>:<B>`                              (early-exit block over depth-K cache)
  - `wblk:<K>:<B>:<tau>`                        (Wake-Verified late entry at `--T`, e.g. `wblk:16:8:0`, `wblk:8:8:0.25`)
  - `wsblk:<K>:<B>:<tau>`                       (Wake-Verified + Suffix-Ngram lookup drafter, e.g. `wsblk:16:8:0`, `wsblk:32:8:inf`)
  - `sblk:<K>:<B>`                              (Lossy / K=T block + Suffix-Ngram lookup drafter, e.g. `sblk:32:8`, `sblk:16:8`)
"""

import argparse
import gc
import json
import math
import os
import sys
import time
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from huginn.decode.hg_model_io import (
    DEFAULT_TASKS,
    NV_BASE,
    STOP_IDS,
    doc_seed,
    make_noise,
    read_jsonl,
)
from huginn.decode.hg_fast_dec import (
    HuginnFastModel,
    _CoreCallPlan,
    _LayerMeta,
    active_bg_groups,
    attn_fa3_paged,
    attn_torch_paged,
    load_hg_fast_direct,
    n_groups_for,
)
from wake_verify.suffix_draft import suffix_propose


def parse_hg_wake_cfg(cfg: str, T_default: int = 32) -> dict:
    """Parse standard and wake/suffix Huginn configs relative to `T_default`."""
    parts = cfg.split(":")
    head = parts[0]
    T = int(T_default)

    if head == "eblk":
        assert len(parts) == 3
        K, B = int(parts[1]), int(parts[2])
        assert 1 <= K <= T and B >= 0
        return dict(cfg=cfg, kind="eblk", K=K, B=B, T=K, bg="none", D_bg=0, tau=float("inf"), suffix=False)

    if head in ("wblk", "wsblk"):
        assert len(parts) == 4, f"Expected w[s]blk:<K>:<B>:<tau>, got {cfg!r}"
        K, B = int(parts[1]), int(parts[2])
        tau = float("inf") if parts[3] in ("inf", "none") else float(parts[3])
        bg = "none" if K == T else "pipe"
        D_bg = 0 if K == T else T
        return dict(cfg=cfg, kind="blk", K=K, B=B, T=T, bg=bg, D_bg=D_bg, tau=tau, suffix=(head == "wsblk"))

    if head == "sblk":
        assert len(parts) in (3, 4)
        K, B = int(parts[1]), int(parts[2])
        bg = parts[3] if len(parts) == 4 else ("none" if K == T else "pipe")
        if K == T:
            bg = "none"
        D_bg = T if bg in ("pipe", "sync") else (int(bg[1:]) if bg.startswith("p") else 0)
        return dict(cfg=cfg, kind="blk", K=K, B=B, T=T, bg=bg, D_bg=D_bg, tau=float("inf"), suffix=True)

    assert head == "blk" and len(parts) in (3, 4), f"Unsupported Huginn cfg: {cfg!r}"
    K, B = int(parts[1]), int(parts[2])
    assert 1 <= K <= T and B >= 0
    bg = parts[3] if len(parts) == 4 else ("none" if K == T else "pipe")
    if K == T:
        bg = "none"
    if bg == "pipe":
        D_bg = T
    elif bg.startswith("p") and bg[1:].isdigit():
        D_bg = int(bg[1:])
    elif bg == "sync":
        D_bg = T
    elif bg == "none":
        D_bg = 0
    else:
        raise ValueError(f"Unsupported bg in {cfg!r}")
    return dict(cfg=cfg, kind="blk", K=K, B=B, T=T, bg=bg, D_bg=D_bg, tau=float("inf"), suffix=False)


class HuginnWakeDecoder:
    """CUDA-Graphed Paged FA3 Huginn Block Decoder with Wake Verification and Arbitrary `T`."""

    def __init__(
        self,
        fm: HuginnFastModel,
        K: int,
        B: int,
        L_max: int = 1920,
        T: int = 32,
        bg: str = "pipe",
        D_bg: int = None,
        tau: float = float("inf"),
        suffix: bool = False,
        nv: int = NV_BASE,
        mask_id: int = None,
        stop_ids=STOP_IDS,
        attn: str = "paged",
        graphs: bool = None,
    ):
        self.fm = fm
        self.m = fm.m
        self.dev, self.dt = fm.dev, fm.dt
        self.acc_dt = torch.float64 if self.dt == torch.float64 else torch.float32
        self.T = int(T)
        self.K = int(K)
        self.B = int(B)
        assert 1 <= self.K <= self.T and self.B >= 0
        self.Ep = self.T - self.K + 1
        self.bg = "none" if self.K == self.T else bg
        if D_bg is None:
            if self.bg in ("pipe", "sync"):
                D_bg = self.T
            elif self.bg.startswith("p") and self.bg[1:].isdigit():
                D_bg = int(self.bg[1:])
            else:
                D_bg = 0
        self.D_bg = int(D_bg)
        self.Ep_bg = self.T - self.D_bg + 1 if self.D_bg > 0 else 1
        self.NG = n_groups_for(self.K, self.D_bg, self.bg)

        self.tau = float(tau)
        self.suffix = bool(suffix)
        self.wake_on = (self.K < self.T) and (self.bg == "pipe") and (self.D_bg == self.T) and (not math.isinf(self.tau))

        c = self.m.config
        self.nv = min(int(nv), int(c.padded_vocab_size))
        self.mask_id = int(mask_id) if (mask_id is not None and int(mask_id) < int(c.padded_vocab_size)) else int(getattr(c, "pad_token_id", 0) or 0)
        self.stop_ids = set(int(x) for x in stop_ids)

        self.attn_mode = attn if (attn == "paged" and str(self.dev).startswith("cuda")) else "torch"
        self.attn_fn = attn_fa3_paged if self.attn_mode == "paged" else attn_torch_paged
        self.graphs_on = (str(self.dev).startswith("cuda") and self.attn_mode == "paged") if graphs is None else bool(graphs)

        D = self.B + 1
        self.D = D
        self.sets = [] if self.B == 0 else [(a, self.B) for a in range(self.B + 1)]
        offs = []
        o = D
        for _, s in self.sets:
            offs.append(o)
            o += s
        self.set_off = offs
        NB = o
        self.NB = NB
        PH = NB - D
        self.PH = PH

        relpos = list(range(D)) + [a + i for a, s in self.sets for i in range(1, s + 1)]
        tails = [list(range(D))] + [list(range(a + 1)) + [offs[k] + i for i in range(s)] for k, (a, s) in enumerate(self.sets)]
        cub = [0, D] + [offs[k] + s for k, (a, s) in enumerate(self.sets)]
        self.nblk = len(tails)
        self.TM = max(len(x) for x in tails)
        tp = torch.zeros((self.nblk, self.TM), dtype=torch.int64)
        for sq, tl_ in enumerate(tails):
            tp[sq, : len(tl_)] = torch.tensor(tl_, dtype=torch.int64)

        self.L_max = int(L_max)
        self.S = self.L_max + self.TM + D + 8
        self.SL = self.S + max(PH, 1) + 8
        self.R_tot = fm.P + fm.Lc_core * self.T + fm.C
        self.cslot = fm.P + fm.Lc_core * self.T

        max_bg_rows = max(1, self.NG) * D
        self.dummy_base = self.R_tot * self.SL
        tot_kv_blocks = self.dummy_base + max_bg_rows + NB + 8
        self.K_pool = torch.zeros((tot_kv_blocks, 1, fm.NH, fm.HD_kv), device=self.dev, dtype=self.dt)
        self.V_pool = torch.zeros_like(self.K_pool)

        self.e_store = torch.zeros((self.S, fm.E), device=self.dev, dtype=self.dt)
        self.hs_dummy_base = self.S
        self.Hs = torch.zeros((self.S + max_bg_rows + 8, fm.E), device=self.dev, dtype=self.dt)
        self.noise = torch.zeros((self.S, fm.E), device=self.dev, dtype=self.dt)

        self.relpos = torch.tensor(relpos, dtype=torch.int64, device=self.dev)
        self.tailpad = tp.to(self.dev)
        self.lens = torch.tensor([len(x) for x in tails], dtype=torch.int64, device=self.dev)
        self.rr = torch.arange(NB, dtype=torch.int64, device=self.dev)
        self.ar = torch.arange(D, dtype=torch.int64, device=self.dev)
        self.is_tok = self.rr < D
        self.ph_slot = self.S + (self.rr - D)
        self.col = torch.arange(self.S, dtype=torch.int64, device=self.dev)[None, :]

        self.hdr = torch.zeros((self.NG + 1,), dtype=torch.int64, device=self.dev)
        self.ids_blk = torch.full((NB,), self.mask_id, dtype=torch.int64, device=self.dev)
        self.comm_next = torch.zeros((D,), dtype=torch.int64, device=self.dev)
        self.top = torch.zeros((NB,), dtype=torch.int64, device=self.dev)
        self.wake_top = torch.zeros((D,), dtype=torch.int64, device=self.dev)
        self.wake_margin = torch.zeros((D,), dtype=torch.float32, device=self.dev)

        cu_pre = cub
        self.meta_pre = [_LayerMeta(self.nblk, self.S, NB, cu_pre, self.dev) for _ in range(fm.P)]

        self.core_plans = []
        self.has_coda_bg = False
        self.coda_bg_group = None
        for kk in range(self.K):
            pl = _CoreCallPlan()
            pl.kk = kk
            pl.k_b = self.Ep + kk
            pl.act = active_bg_groups(self.K, self.T, self.D_bg, self.NG, kk)
            pl.nA = len(pl.act)
            pl.G = pl.nA * D
            pl.N = pl.G + NB
            pl.nseq = pl.nA + self.nblk
            cu_c = [D * i for i in range(pl.nA + 1)] + [pl.G + x for x in cub[1:]]
            pl.metas = [_LayerMeta(pl.nseq, self.S, pl.N, cu_c, self.dev) for _ in range(fm.Lc_core)]
            for g, pj in pl.act:
                if pj == self.T:
                    self.has_coda_bg = True
                    self.coda_bg_group = g
            self.core_plans.append(pl)

        if self.has_coda_bg:
            cu_coda = [0, D] + [D + x for x in cub[1:]]
            self.meta_coda = [_LayerMeta(1 + self.nblk, self.S, D + NB, cu_coda, self.dev) for _ in range(fm.C)]
        else:
            self.meta_coda = [_LayerMeta(self.nblk, self.S, NB, cu_pre, self.dev) for _ in range(fm.C)]

        if str(self.dev).startswith("cuda"):
            self.h_hdr = torch.zeros((self.NG + 1,), dtype=torch.int64).pin_memory()
            self.h_ids = torch.zeros((NB,), dtype=torch.int64).pin_memory()
            self.h_cnext = torch.zeros((D,), dtype=torch.int64).pin_memory()
            self.h_top = torch.zeros((NB,), dtype=torch.int64).pin_memory()
            self.h_wtop = torch.zeros((D,), dtype=torch.int64).pin_memory()
            self.h_wmarg = torch.zeros((D,), dtype=torch.float32).pin_memory()
            self.ev = torch.cuda.Event()

        self.graph = None
        self._graph_pool = None
        if self.graphs_on:
            self._capture_step_graph()

    def _step_device(self):
        fm = self.fm
        D, NB, SL = self.D, self.NB, self.SL
        lc = self.hdr[0]
        col = self.col

        pos_blk = lc + self.relpos
        wslot_loc = torch.where(self.is_tok, lc + self.rr, self.ph_slot)
        rel = col - lc
        tail = self.tailpad.gather(1, rel.clamp(0, self.TM - 1).expand(self.nblk, -1))
        pt_blk_base = torch.where(rel < 0, col, wslot_loc[tail])
        pt_blk_valid = rel < self.lens[:, None]
        sl_blk = (lc + self.lens).to(torch.int32)

        # 1. Prelude
        x_blk = F.embedding(self.ids_blk, fm.wte) * fm.emb_scale
        for i in range(fm.P):
            mp = self.meta_pre[i]
            off = i * SL
            mp.wslot.copy_(off + wslot_loc)
            mp.seqlens.copy_(sl_blk)
            mp.page_table.copy_(torch.where(pt_blk_valid, off + pt_blk_base, 0).to(torch.int32))
            x_blk = fm.run_layer(fm.pre_layers[i], x_blk, pos_blk, mp, self.K_pool, self.V_pool, self.attn_fn)
        e_blk = x_blk
        self.e_store.index_copy_(0, lc + self.ar, e_blk[:D])

        # 2. K Core steps
        s_blk = self.noise.index_select(0, pos_blk)
        coda_fin = None

        for kk in range(self.K):
            pl = self.core_plans[kk]
            s_parts, e_parts, pos_parts = [], [], []
            bg_info = []

            for gi, (g, pj) in enumerate(pl.act):
                hi = self.hdr[g - 1]
                lo = self.hdr[g]
                posg_raw = hi - D + self.ar
                real = self.ar >= D - (hi - lo)
                posg = posg_raw.clamp(min=0)
                s_g = self.noise.index_select(0, posg) if pj == self.Ep_bg else self.Hs.index_select(0, posg)
                e_g = self.e_store.index_select(0, posg)
                s_parts.append(s_g)
                e_parts.append(e_g)
                pos_parts.append(posg)
                bg_info.append((gi, pj, posg, real, hi, col < hi))

            s_parts.append(s_blk)
            e_parts.append(e_blk)
            pos_parts.append(pos_blk)

            s_all = s_parts[0] if len(s_parts) == 1 else torch.cat(s_parts, dim=0)
            e_all = e_parts[0] if len(e_parts) == 1 else torch.cat(e_parts, dim=0)
            pos_all = pos_parts[0] if len(pos_parts) == 1 else torch.cat(pos_parts, dim=0)

            x_all = F.linear(torch.cat([s_all, e_all], dim=-1), fm.adap_w, fm.adap_b)
            for j in range(fm.Lc_core):
                mc = pl.metas[j]
                reg_blk = fm.P + fm.Lc_core * (pl.k_b - 1) + j
                off_blk = reg_blk * SL
                if pl.nA == 0:
                    mc.wslot.copy_(off_blk + wslot_loc)
                    mc.seqlens.copy_(sl_blk)
                    mc.page_table.copy_(torch.where(pt_blk_valid, off_blk + pt_blk_base, 0).to(torch.int32))
                else:
                    w_l, sl_l, pt_l = [], [], []
                    for gi, pj, posg, real, hi, pt_g_valid in bg_info:
                        reg_g = fm.P + fm.Lc_core * (pj - 1) + j
                        off_g = reg_g * SL
                        dum_g = self.dummy_base + gi * D + self.ar
                        w_l.append(torch.where(real, off_g + posg, dum_g))
                        sl_l.append(hi.view(1).to(torch.int32))
                        pt_l.append(torch.where(pt_g_valid, off_g + col, 0))
                    w_l.append(off_blk + wslot_loc)
                    sl_l.append(sl_blk)
                    pt_l.append(torch.where(pt_blk_valid, off_blk + pt_blk_base, 0))
                    mc.wslot.copy_(torch.cat(w_l, dim=0))
                    mc.seqlens.copy_(torch.cat(sl_l, dim=0))
                    mc.page_table.copy_(torch.cat(pt_l, dim=0).to(torch.int32))
                x_all = fm.run_layer(fm.core_layers[j], x_all, pos_all, mc, self.K_pool, self.V_pool, self.attn_fn)

            for gi, pj, posg, real, hi, _ in bg_info:
                s_out_g = x_all[gi * D : (gi + 1) * D]
                h_dst = torch.where(real, posg, self.hs_dummy_base + gi * D + self.ar)
                self.Hs.index_copy_(0, h_dst, s_out_g)
                if pj == self.T:
                    coda_fin = (s_out_g, posg, real, hi)

            s_blk = x_all[pl.G : pl.G + NB]

        if self.K == self.T:
            self.Hs.index_copy_(0, lc + self.ar, s_blk[:D])

        # 3. Coda + Readout (fusing Wake Verification on `x_coda[:D]` in the single `lm_head` GEMM!)
        if self.has_coda_bg:
            assert coda_fin is not None
            s_fin, posg_fin, real_fin, hi_fin = coda_fin
            pt_fin_valid = col < hi_fin
            sl_coda = torch.cat([hi_fin.view(1).to(torch.int32), sl_blk], dim=0)
            pos_coda = torch.cat([posg_fin, pos_blk], dim=0)
            x_coda = fm.ops.rms(torch.cat([s_fin, s_blk], dim=0), fm.ln_f)
            for c_i in range(fm.C):
                md = self.meta_coda[c_i]
                off_c = (self.cslot + c_i) * SL
                dum_c = self.dummy_base + self.ar
                w_fin = torch.where(real_fin, off_c + posg_fin, dum_c)
                pt_fin = torch.where(pt_fin_valid, off_c + col, 0)
                pt_b = torch.where(pt_blk_valid, off_c + pt_blk_base, 0)
                md.wslot.copy_(torch.cat([w_fin, off_c + wslot_loc], dim=0))
                md.seqlens.copy_(sl_coda)
                md.page_table.copy_(torch.cat([pt_fin, pt_b], dim=0).to(torch.int32))
                x_coda = fm.run_layer(fm.coda_layers[c_i], x_coda, pos_coda, md, self.K_pool, self.V_pool, self.attn_fn)
            if self.wake_on:
                lg_all = F.linear(fm.ops.rms(x_coda, fm.ln_f), fm.lm_head)[:, : self.nv]
                w_lg = lg_all[:D].to(self.acc_dt)
                self.wake_top.copy_(w_lg.argmax(dim=-1))
                w_max = w_lg.max(dim=-1).values
                w_comm = w_lg.gather(1, self.comm_next[:, None]).squeeze(1)
                self.wake_margin.copy_((w_max - w_comm).float())
                self.top.copy_(lg_all[D : D + NB].to(self.acc_dt).argmax(dim=-1))
            else:
                x_blk_out = x_coda[D : D + NB]
                logits = F.linear(fm.ops.rms(x_blk_out, fm.ln_f), fm.lm_head)[:, : self.nv]
                self.top.copy_(logits.to(self.acc_dt).argmax(dim=-1))
        else:
            x_coda = fm.ops.rms(s_blk, fm.ln_f)
            for c_i in range(fm.C):
                md = self.meta_coda[c_i]
                off_c = (self.cslot + c_i) * SL
                md.wslot.copy_(off_c + wslot_loc)
                md.seqlens.copy_(sl_blk)
                md.page_table.copy_(torch.where(pt_blk_valid, off_c + pt_blk_base, 0).to(torch.int32))
                x_coda = fm.run_layer(fm.coda_layers[c_i], x_coda, pos_blk, md, self.K_pool, self.V_pool, self.attn_fn)
            logits = F.linear(fm.ops.rms(x_coda, fm.ln_f), fm.lm_head)[:, : self.nv]
            self.top.copy_(logits.to(self.acc_dt).argmax(dim=-1))

    @torch.no_grad()
    def _capture_step_graph(self):
        warm = [(self.NG + 3 - i) * max(self.D, 4) for i in range(self.NG + 1)]
        self.hdr.copy_(torch.tensor(warm, dtype=torch.int64, device=self.dev))
        self.ids_blk.fill_(self.mask_id)
        self.comm_next.zero_()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._step_device()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(g, pool=pool):
            self._step_device()
        torch.cuda.synchronize()
        self.graph = g
        self._graph_pool = pool

    @torch.no_grad()
    def prefill(self, prompt_ids: list, noise: torch.Tensor) -> int:
        fm = self.fm
        P0 = len(prompt_ids)
        assert P0 + self.TM + 8 <= self.S, (P0, self.S)
        self.noise[: len(noise)].copy_(noise[: self.S] if len(noise) > self.S else noise)

        x_ids = torch.tensor(prompt_ids, dtype=torch.int64, device=self.dev)
        pos = torch.arange(P0, dtype=torch.int64, device=self.dev)

        def blk_pre(lw, x, reg):
            E, NH, HD, HD_kv = fm.E, fm.NH, fm.HD, fm.HD_kv
            h = fm.ops.rms(x, lw["n1"])
            qkv = F.linear(h, lw["qkv_w"], lw["qkv_b"])
            q, k, v = qkv[:, :E], qkv[:, E : 2 * E], qkv[:, 2 * E :]
            q, k = fm.ops.rope(q, k, pos)
            off = reg * self.SL
            self.K_pool[off : off + P0, 0, :, :HD].copy_(k.view(P0, NH, HD))
            self.V_pool[off : off + P0, 0, :, :HD].copy_(v.view(P0, NH, HD))
            if HD_kv > HD:
                self.K_pool[off : off + P0, 0, :, HD:].zero_()
                self.V_pool[off : off + P0, 0, :, HD:].zero_()
            qh = q.view(P0, NH, HD).transpose(0, 1)[None]
            kh = k.view(P0, NH, HD).transpose(0, 1)[None]
            vh = v.view(P0, NH, HD).transpose(0, 1)[None]
            if q.dtype in (torch.float32, torch.float64) and not str(self.dev).startswith("cuda"):
                acc = torch.float64 if q.dtype == torch.float64 else torch.float32
                sc = torch.einsum("bhqd,bhkd->bhqk", qh.to(acc), kh.to(acc)) * (HD ** -0.5)
                cm = torch.tril(torch.ones((P0, P0), dtype=torch.bool, device=self.dev))
                sc = sc.masked_fill(~cm[None, None], float("-inf"))
                yh = torch.einsum("bhqk,bhkd->bhqd", torch.softmax(sc, dim=-1), vh.to(acc)).to(q.dtype)
            else:
                yh = F.scaled_dot_product_attention(qh, kh, vh, dropout_p=0.0, is_causal=True)
            a_out = yh[0].transpose(0, 1).reshape(P0, E)
            a = F.linear(a_out, lw["o_w"])
            x = fm.ops.rms(a + x, lw["n2"])
            m_in = fm.ops.rms(x, lw["n3"])
            m_out = F.linear(fm.ops.silu_mul(F.linear(m_in, lw["fc_w"])), lw["down_w"])
            return fm.ops.rms(m_out + x, lw["n4"])

        x = F.embedding(x_ids, fm.wte) * fm.emb_scale
        for i in range(fm.P):
            x = blk_pre(fm.pre_layers[i], x, i)
        self.e_store[:P0].copy_(x)
        e = x
        s = self.noise[:P0]
        for k_step in range(1, self.T + 1):
            x = F.linear(torch.cat([s, e], dim=-1), fm.adap_w, fm.adap_b)
            for j in range(fm.Lc_core):
                x = blk_pre(fm.core_layers[j], x, fm.P + fm.Lc_core * (k_step - 1) + j)
            s = x
        self.Hs[:P0].copy_(s)
        x = fm.ops.rms(s, fm.ln_f)
        for c_i in range(fm.C):
            x = blk_pre(fm.coda_layers[c_i], x, self.cslot + c_i)
        logits = F.linear(fm.ops.rms(x[-1:], fm.ln_f), fm.lm_head)[0, : self.nv].to(self.acc_dt)
        return int(logits.argmax().item())

    def next_props(self, top: list, c: int) -> list:
        if self.B == 0:
            return []
        off = self.set_off[c]
        return [int(top[off + j]) for j in range(self.B)]

    @torch.no_grad()
    def step_block(self, hist: list, ids_blk: list, comm_next_host: list):
        if self.graph is not None:
            for i, h in enumerate(hist):
                self.h_hdr[i] = h
            self.h_ids.copy_(torch.as_tensor(ids_blk, dtype=torch.int64))
            if self.wake_on:
                self.h_cnext.copy_(torch.as_tensor(comm_next_host, dtype=torch.int64))
            self.hdr.copy_(self.h_hdr, non_blocking=True)
            self.ids_blk.copy_(self.h_ids, non_blocking=True)
            if self.wake_on:
                self.comm_next.copy_(self.h_cnext, non_blocking=True)
            self.graph.replay()
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

        self.hdr.copy_(torch.tensor(hist, dtype=torch.int64, device=self.dev))
        self.ids_blk.copy_(torch.tensor(ids_blk, dtype=torch.int64, device=self.dev))
        if self.wake_on:
            self.comm_next.copy_(torch.tensor(comm_next_host, dtype=torch.int64, device=self.dev))
        self._step_device()
        return (
            self.top.tolist(),
            self.wake_top.tolist() if self.wake_on else None,
            self.wake_margin.tolist() if self.wake_on else None,
        )

    @torch.no_grad()
    def decode_one(
        self,
        prompt_ids: list,
        noise: torch.Tensor,
        max_new: int = 256,
        tok=None,
        until=(),
        prop_override_fn=None,
    ) -> dict:
        cuda = str(self.dev).startswith("cuda")
        P0 = len(prompt_ids)
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        first = self.prefill(prompt_ids, noise)
        if cuda:
            torch.cuda.synchronize()
        t_pref = time.perf_counter() - t0

        known = list(prompt_ids) + [first]
        lc = P0
        hist = [P0] * (self.NG + 1)
        props = []
        done = (first in self.stop_ids) or (max_new <= 1)
        why = "eos" if first in self.stop_ids else ("length" if max_new <= 1 else "running")

        cs = []
        commits = []
        n_prop_steps = 0
        n_acc_total = 0
        n_rollbacks = 0
        n_rolled_tokens = 0
        n_drain_steps = 0
        chk_len = 0
        tail_masks = [self.mask_id] * (self.NB - self.D)
        text_until = [u for u in (until or ()) if u and u not in ("<|end_turn|>", "<|end_text|>")]
        tail_win = max(32, max((len(u) for u in text_until), default=0) + 16)

        def _dec_txt(ids_slice):
            try:
                return tok.decode(ids_slice, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            except TypeError:
                return tok.decode(ids_slice, skip_special_tokens=True)

        def _make_comm_next(hist_cur, known_cur):
            if not self.wake_on:
                return []
            g_c = self.coda_bg_group
            hi_w, lo_w = hist_cur[g_c - 1], hist_cur[g_c]
            out = []
            for i in range(self.D):
                p = hi_w - self.D + i
                if p >= 0 and p >= lo_w and (p + 1) < len(known_cur):
                    tv = int(known_cur[p + 1])
                    out.append(tv if 0 <= tv < self.nv else 0)
                else:
                    out.append(0)
            return out

        if cuda:
            torch.cuda.synchronize()
        t_dec0 = time.perf_counter()

        while True:
            need_drain = self.wake_on and done and (hist[self.NG] < hist[0])
            if done and not need_drain:
                break

            lc = hist[0]
            if not done and self.suffix and self.B > 0:
                props = suffix_propose(known, self.B, fallback_props=props, stop_ids=self.stop_ids)
            if not done and prop_override_fn is not None:
                ov = prop_override_fn(len(commits), lc, known)
                if ov is not None:
                    props = list(ov)[: self.B]

            cur_props = [] if done else props
            filler = [self.mask_id] * (self.B - len(cur_props))
            ids_blk = [known[-1]] + cur_props + filler + tail_masks
            cnext = _make_comm_next(hist, known)

            top, wtop, wmarg = self.step_block(hist, ids_blk, cnext)

            # 1. Check Wake Verification on `coda_bg_group` (`g = NG`) FIRST
            rolled_back = False
            if self.wake_on:
                g_c = self.coda_bg_group
                hi_w, lo_w = hist[g_c - 1], hist[g_c]
                if hi_w > lo_w:
                    for i in range(self.D - (hi_w - lo_w), self.D):
                        p = hi_w - self.D + i
                        y_true = int(wtop[i])
                        y_comm = int(known[p + 1])
                        if y_true != y_comm and (self.tau <= 0.0 or float(wmarg[i]) > self.tau):
                            n_rolled_tokens += len(known) - (p + 1)
                            n_rollbacks += 1
                            known = known[: p + 1] + [y_true]
                            lc_rb = p + 1
                            hist = [lc_rb] * (self.NG + 1)
                            lc = lc_rb
                            props = []
                            chk_len = max(0, len(known) - P0 - 8)
                            if y_true in self.stop_ids:
                                done = True
                                why = "eos"
                            elif (len(known) - P0) >= max_new:
                                done = True
                                why = "length"
                            else:
                                done = False
                                why = "running"
                                if tok is not None and text_until:
                                    txt_now = _dec_txt(known[max(P0, len(known) - tail_win) :])
                                    for u in text_until:
                                        if u in txt_now:
                                            done = True
                                            why = "until"
                                            break
                            rolled_back = True
                            break

            if rolled_back:
                if need_drain:
                    n_drain_steps += 1
                cs.append(0)
                commits.append(0)
                continue

            # 2. Drain step passed wake check -> shift wake pipeline
            if done:
                n_drain_steps += 1
                hist = [lc] + hist[: self.NG]
                cs.append(0)
                commits.append(0)
                continue

            # 3. Normal optimistic block commitment
            np_ = len(cur_props)
            c = 0
            while c < np_ and int(top[c]) == cur_props[c]:
                c += 1
            if np_ > 0:
                n_prop_steps += 1
                n_acc_total += c

            commit = cur_props[:c] + [int(top[c])]
            emitted_this_step = 0
            for tk in commit:
                known.append(tk)
                emitted_this_step += 1
                if tk in self.stop_ids:
                    done = True
                    why = "eos"
                    break
                if (len(known) - P0) >= max_new:
                    done = True
                    why = "length"
                    break

            lc_next = len(known) - 1
            props = [] if done else self.next_props(top, c)
            hist = [lc_next] + hist[: self.NG]
            lc = lc_next

            cs.append(c)
            commits.append(emitted_this_step)

            if not done and tok is not None and text_until and ((len(known) - P0) - chk_len >= 8):
                chk_len = len(known) - P0
                txt_now = _dec_txt(known[max(P0, len(known) - tail_win) :])
                for u in text_until:
                    if u in txt_now:
                        done = True
                        why = "until"
                        break

        if cuda:
            torch.cuda.synchronize()
        t_dec = time.perf_counter() - t_dec0

        gen_ids = known[P0:]
        text = _dec_txt(gen_ids) if tok is not None else ""
        for u in text_until:
            if u in text:
                text = text.split(u)[0]

        n_steps = len(commits)
        net_tokens = len(gen_ids)
        return dict(
            gen_ids=gen_ids,
            text=text,
            tokens=net_tokens,
            dec_tokens=net_tokens,
            gross_tokens=sum(commits),
            steps=n_steps,
            c_list=cs,
            mean_c=(n_acc_total / n_prop_steps) if n_prop_steps > 0 else 0.0,
            mean_block_len=(net_tokens / n_steps) if n_steps > 0 else 1.0,
            prefill_s=t_pref,
            decode_s=t_dec,
            total_s=t_pref + t_dec,
            decode_tok_per_s=(net_tokens / t_dec) if t_dec > 0 else 0.0,
            finish=why,
            prompt_len=P0,
            lc_final=lc,
            hist_final=hist,
            n_rollbacks=n_rollbacks,
            n_rolled_tokens=n_rolled_tokens,
            n_drain_steps=n_drain_steps,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--n-docs", type=int, default=0)
    ap.add_argument("--doc-lo", type=int, default=0)
    ap.add_argument("--doc-hi", type=int, default=0)
    ap.add_argument("--shard-tag", default="")
    ap.add_argument("--cfgs", required=True)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--fused", type=int, default=1)
    ap.add_argument("--graphs", type=int, default=1)
    ap.add_argument("--attn", default="paged")
    ap.add_argument("--burn-s", type=float, default=0.0)
    ap.add_argument("--verify-gate", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    all_raw = read_jsonl(args.tasks)
    if args.doc_hi > 0:
        raw = all_raw[args.doc_lo : args.doc_hi]
    elif args.n_docs > 0:
        raw = all_raw[args.doc_lo : args.doc_lo + args.n_docs]
    else:
        raw = all_raw[args.doc_lo :]
    assert len(raw) > 0

    docs = [
        dict(
            doc_id=int(r["doc_id"]),
            ids=tok.encode(r["context"], add_special_tokens=False),
            until=list(dict.fromkeys(list(r.get("until", [])) + ["<|end_turn|>", "<|end_text|>"])),
        )
        for r in raw
    ]
    L_max = max(len(d["ids"]) for d in docs) + args.max_new + 32

    fm, _, mask_id = load_hg_fast_direct(
        args.model, device="cuda", dtype=torch.bfloat16, max_seq=L_max + 64, fused=bool(args.fused)
    )
    m = fm.m
    print(f"[hg-wake] loaded {args.tag} T={args.T} (vocab={m.config.padded_vocab_size}, mask_id={mask_id}, HD_kv={fm.HD_kv})", flush=True)

    if args.burn_s > 0:
        t_end = time.time() + args.burn_s
        bx = torch.randn((2048, 2048), device="cuda", dtype=torch.bfloat16)
        while time.time() < t_end:
            _ = bx @ bx
        torch.cuda.synchronize()
        del bx
        torch.cuda.empty_cache()

    cfg_list = [c.strip() for c in args.cfgs.split(",") if c.strip()]
    sfx = f".{args.shard_tag}" if args.shard_tag else ""
    t_tag = f"T{args.T}_" if args.T != 32 else ""
    kept_graphs = []

    for cfg in cfg_list:
        pc = parse_hg_wake_cfg(cfg, T_default=args.T)
        safe_cfg = cfg.replace(":", "_").replace(".", "p")
        fn = os.path.join(args.out_dir, f"{t_tag}{safe_cfg}{sfx}.jsonl")
        if os.path.exists(fn) and len(read_jsonl(fn)) >= len(docs):
            print(f"[skip] {fn} already complete ({len(docs)} docs)", flush=True)
            continue

        use_graphs = bool(args.graphs)
        r_eager_gate = None
        if args.verify_gate and use_graphs:
            d_eag = HuginnWakeDecoder(
                fm, K=pc["K"], B=pc["B"], L_max=L_max, T=pc["T"], bg=pc["bg"], D_bg=pc["D_bg"],
                tau=pc["tau"], suffix=pc["suffix"], nv=NV_BASE, mask_id=mask_id, attn=args.attn, graphs=False,
            )
            g_ns = make_noise(m, L_max, doc_seed(args.seed, 777777), fm.dev, fm.dt)
            r_eager_gate = d_eag.decode_one(docs[0]["ids"][:96], g_ns, max_new=32, tok=None, until=())
            del d_eag
            gc.collect()
            torch.cuda.empty_cache()

        dec = HuginnWakeDecoder(
            fm, K=pc["K"], B=pc["B"], L_max=L_max, T=pc["T"], bg=pc["bg"], D_bg=pc["D_bg"],
            tau=pc["tau"], suffix=pc["suffix"], nv=NV_BASE, mask_id=mask_id, attn=args.attn, graphs=use_graphs,
        )
        if dec.graph is not None:
            kept_graphs.append((dec.graph, dec._graph_pool))

        if r_eager_gate is not None:
            g_ns = make_noise(m, L_max, doc_seed(args.seed, 777777), fm.dev, fm.dt)
            r_gr = dec.decode_one(docs[0]["ids"][:96], g_ns, max_new=32, tok=None, until=())
            diffs = sum(1 for a, b in zip(r_eager_gate["gen_ids"], r_gr["gen_ids"]) if a != b)
            print(
                f"[gpu-gate] T={args.T} {cfg}: len={len(r_gr['gen_ids'])} graph_vs_eager_diffs={diffs} "
                f"rollbacks={r_gr['n_rollbacks']} tok/s={r_gr['decode_tok_per_s']:.1f}",
                flush=True,
            )
            assert diffs == 0, f"CUDA graph vs eager mismatch on {cfg}: {diffs} diffs"
        else:
            w_ns = make_noise(m, L_max, doc_seed(args.seed, 999999), fm.dev, fm.dt)
            _ = dec.decode_one(docs[0]["ids"][:64], w_ns, max_new=16, tok=tok, until=())

        recs = []
        t0 = time.time()
        for idx, d in enumerate(docs):
            ns = make_noise(m, L_max, doc_seed(args.seed, d["doc_id"]), fm.dev, fm.dt)
            r = dec.decode_one(d["ids"], ns, max_new=args.max_new, tok=tok, until=d["until"])
            r["doc_id"] = d["doc_id"]
            r["cfg"] = cfg
            r["T"] = args.T
            recs.append(r)
            if (idx + 1) % 50 == 0 or (idx + 1) == len(docs):
                tot_tok = sum(int(x["dec_tokens"]) for x in recs)
                tot_sec = sum(float(x["decode_s"]) for x in recs)
                tot_rb = sum(int(x["n_rollbacks"]) for x in recs)
                tps = tot_tok / max(1e-9, tot_sec)
                mb = tot_tok / max(1, sum(int(x["steps"]) for x in recs))
                print(
                    f"[{args.tag} T={args.T}{sfx} {cfg}] {idx + 1}/{len(docs)} ({time.time() - t0:.1f}s) | "
                    f"net_blk={mb:.2f} | rb={tot_rb} | tok/s={tps:.2f}",
                    flush=True,
                )

        import socket
        tmp_fn = f"{fn}.{socket.gethostname()}.{os.getpid()}.tmp"
        with open(tmp_fn, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp_fn, fn)
        del dec
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
