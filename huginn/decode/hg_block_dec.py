"""MARS + Late-Entry Block Decoder for Huginn-0125 (batch=1, greedy verification).

Supports:
  - Sequential baseline:         `blk:32:0` (B=0, K=32, 1 token/step)
  - Full-depth MARS block:       `blk:32:4`, `blk:32:8` (lossless exact match to r:32)
  - Late-Entry MARS block:       `blk:<K>:<B>` (e.g. K in {4,6,8,12,16,24}, B in {2,4,6,8})
      * `bg=pipe` (default): Cross-step pipelined full-depth KV completion (`NG = ceil(T/K)` groups
        fused into the K Core calls + Coda call per step; zero extra sequential layer calls, zero
        Prelude recomputation since committed token slots already have exact Prelude KV & `e`).
      * `bg=sync`: Synchronous full-depth KV backfill immediately after each block step.
      * `bg=none`: No-catchup ablation (`lknb`: retains provisional K-step KV indefinitely).
"""

import argparse
import gc
import glob
import json
import os
import time
import torch
import torch.nn.functional as F

from huginn.decode.hg_model_io import (
    DEFAULT_GOLD,
    DEFAULT_TASKS,
    NV_BASE,
    STOP_IDS,
    apply_rotary_emb_complex_like,
    doc_seed,
    ensure_freqs_cis,
    load_gold,
    load_hg_model_and_tok,
    make_noise,
    paired_compare,
    read_jsonl,
    score_one,
)


def parse_blk_cfg(cfg: str, T_default: int = 32):
    """Parse `blk:<K>:<B>[:<bg>]` (Late Entry on T=32) or `eblk:<K>:<B>` (Early Exit on depth-K cache, T=K)."""
    parts = cfg.split(":")
    assert len(parts) in (3, 4) and parts[0] in ("blk", "eblk"), f"Invalid block cfg: {cfg!r}"
    kind = parts[0]
    K = int(parts[1])
    B = int(parts[2])
    if kind == "eblk":
        assert len(parts) == 3 and 1 <= K <= int(T_default) and B >= 0, (cfg, K, B)
        return dict(cfg=cfg, kind="eblk", K=K, B=B, T=K, bg="none")
    T = int(T_default)
    assert 1 <= K <= T and B >= 0, (cfg, K, B, T)
    if len(parts) == 4:
        bg = parts[3]
    else:
        bg = "none" if K == T else "pipe"
    assert bg in ("pipe", "sync", "none") or (bg.startswith("p") and bg[1:].isdigit()), f"Unknown bg mode in {cfg!r}"
    if K == T:
        bg = "none"
    return dict(cfg=cfg, kind="blk", K=K, B=B, T=T, bg=bg)


def n_groups(K: int, T: int, bg: str) -> int:
    return -(-T // K) if (bg == "pipe" and K < T) else 0


def active_groups(K: int, NG: int, kk: int, T: int):
    return [g for g in range(1, NG + 1) if K * (g - 1) + kk + 1 <= T]


class HuginnBlockDecoder:
    """Single-sequence MARS `<|mask|>` lattice block decoder with fused pipelined KV catch-up."""

    def __init__(
        self,
        m,
        K: int,
        B: int,
        L_max: int = 1920,
        T: int = 32,
        bg: str = "pipe",
        nv: int = NV_BASE,
        mask_id: int = None,
        stop_ids=STOP_IDS,
    ):
        self.m = m
        self.T = int(T)
        self.K = int(K)
        self.B = int(B)
        assert 1 <= self.K <= self.T and self.B >= 0
        self.Ep = self.T - self.K + 1  # 1-based entry Core step in 1..T
        self.bg = "none" if self.K == self.T else bg
        assert self.bg in ("pipe", "sync", "none")
        self.NG = n_groups(self.K, self.T, self.bg)

        c = m.config
        tr = m.transformer
        self.nv = min(int(nv), int(c.padded_vocab_size))
        if mask_id is not None and int(mask_id) < int(c.padded_vocab_size):
            self.mask_id = int(mask_id)
        else:
            self.mask_id = int(getattr(c, "pad_token_id", 0) or 0)
        self.stop_ids = set(int(x) for x in stop_ids)

        self.P, self.Lc_core, self.C = len(tr.prelude), len(tr.core_block), len(tr.coda)
        self.H = c.num_attention_heads
        self.hd = c.n_embd // c.num_attention_heads
        self.E = c.n_embd
        self.S_slots = self.P + self.Lc_core * self.T + self.C
        self.cslot = self.P + self.Lc_core * self.T

        p = next(m.parameters())
        self.dev, self.dt = p.device, p.dtype
        self.emb_scale = float(m.emb_scale)
        self.freqs = m.freqs_cis  # (1, max_seq, 1, hd/2, 2)

        # Build MARS (1 + B)^2 lattice geometry
        self.D_blk = self.B + 1
        if self.B == 0:
            self.sets = []
        else:
            self.sets = [(a, self.B) for a in range(self.B + 1)]

        offs = []
        o = self.D_blk
        for _, s in self.sets:
            offs.append(o)
            o += s
        self.set_off = offs
        self.NB = o

        relpos = list(range(self.D_blk)) + [a + i for a, s in self.sets for i in range(1, s + 1)]
        wm = torch.zeros((self.NB, self.NB), dtype=torch.bool, device=self.dev)
        for j in range(self.D_blk):
            wm[j, : j + 1] = True
        for k_idx, (a, s) in enumerate(self.sets):
            off = offs[k_idx]
            for i in range(1, s + 1):
                r = off + i - 1
                wm[r, : a + 1] = True
                wm[r, off : r + 1] = True

        self.relpos = torch.tensor(relpos, dtype=torch.long, device=self.dev)
        self.wm = wm  # (NB, NB) bool

        # Contiguous zero-copy KV cache: (S_slots, H, L_tot, hd)
        self.L_max = int(L_max)
        self.L_tot = self.L_max + self.NB + 8
        self.freqs = ensure_freqs_cis(m, self.L_tot, self.dev)  # extends beyond 4096 if needed (e.g. MBPP doc 482)
        self.K_buf = torch.zeros((self.S_slots, self.H, self.L_tot, self.hd), device=self.dev, dtype=self.dt)
        self.V_buf = torch.zeros_like(self.K_buf)
        self.e_store = torch.zeros((self.L_tot, self.E), device=self.dev, dtype=self.dt)
        self.Hs = torch.zeros((self.L_tot, self.E), device=self.dev, dtype=self.dt)
        self.noise = torch.zeros((self.L_tot, self.E), device=self.dev, dtype=self.dt)

        # Pre-built all-True context visibility row buffer to avoid reallocations
        self._true_ctx = torch.ones((self.NB, self.L_tot), dtype=torch.bool, device=self.dev)

    def _qkv_th(self, attn, h: torch.Tensor, freqs: torch.Tensor):
        """h: (1, R, E), freqs: (1, R, 1, hd/2, 2) -> q, k, v each (1, H, R, hd)."""
        R = h.shape[1]
        q, k, v = attn.Wqkv(h).split(attn.chunks, dim=2)
        q = q.view(1, R, self.H, self.hd)
        k = k.view(1, R, self.H, self.hd)
        v = v.view(1, R, self.H, self.hd)
        if self.m.config.qk_bias:
            qb, kb = attn.qk_bias.split(1, dim=0)
            q, k = (q + qb).to(q.dtype), (k + kb).to(q.dtype)
        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def _run_segmented_block(self, blk, x: torch.Tensor, freqs: torch.Tensor, segs: list) -> torch.Tensor:
        """Execute one SandwichBlock with fused norms/Wqkv/proj/MLP over R rows and zero-copy per-segment SDPA.

        Each item in `segs` is `(r0, r1, slot, w0, w1, L_read, attn_mask_or_None_if_causal_1tok)`.
        Segments are processed in list order so background writes to `< lc` are visible to the active block.
        """
        R = x.shape[1]
        q, k, v = self._qkv_th(blk.attn, blk.norm_1(x), freqs)  # (1, H, R, hd)
        ys = []
        for r0, r1, slot, w0, w1, L_read, mask in segs:
            self.K_buf[slot, :, w0:w1, :] = k[0, :, r0:r1, :]
            self.V_buf[slot, :, w0:w1, :] = v[0, :, r0:r1, :]
            Kv = self.K_buf[slot, :, :L_read, :][None]
            Vv = self.V_buf[slot, :, :L_read, :][None]
            qv = q[:, :, r0:r1, :]
            if mask is None:
                # Single query attending to all `0 .. L_read-1`
                y = F.scaled_dot_product_attention(qv, Kv, Vv, dropout_p=0.0, is_causal=False)
            else:
                y = F.scaled_dot_product_attention(qv, Kv, Vv, attn_mask=mask, dropout_p=0.0, is_causal=False)
            ys.append(y)
        y_all = ys[0] if len(ys) == 1 else torch.cat(ys, dim=2)
        a = blk.attn.proj(y_all.transpose(1, 2).reshape(1, R, self.E).contiguous())
        x = blk.norm_2(a + x)
        return blk.norm_4(blk.mlp(blk.norm_3(x)) + x)

    def _causal_tail_mask(self, lo: int, hi: int) -> torch.Tensor:
        """Boolean lower-right causal mask of shape (1, 1, hi - lo, hi) for positions `lo .. hi-1` reading `0 .. hi-1`."""
        n = hi - lo
        if n == 1:
            return None  # single token at hi-1 sees all 0..hi-1
        q_pos = torch.arange(lo, hi, device=self.dev)[:, None]
        k_pos = torch.arange(0, hi, device=self.dev)[None, :]
        return (k_pos <= q_pos)[None, None]

    def _block_mask(self, lc: int) -> torch.Tensor:
        """Boolean mask of shape (1, 1, NB, lc + NB): all context `0..lc-1` True, intra-block `wm`."""
        if self.NB == 1:
            return None  # single token at lc sees all 0..lc
        ctx_part = self._true_ctx[:, :lc]
        return torch.cat([ctx_part, self.wm], dim=1)[None, None]

    @torch.no_grad()
    def prefill(self, prompt_ids: list, noise: torch.Tensor) -> int:
        tr = self.m.transformer
        P0 = len(prompt_ids)
        assert P0 + self.NB + 4 <= self.L_tot, (P0, self.L_tot)
        self.noise[: len(noise)].copy_(noise)

        x_ids = torch.tensor(prompt_ids, dtype=torch.long, device=self.dev)[None]
        freqs = self.freqs[:, :P0]

        def blk_pre(blk, x, slot):
            q, k, v = self._qkv_th(blk.attn, blk.norm_1(x), freqs)
            self.K_buf[slot, :, :P0, :] = k[0]
            self.V_buf[slot, :, :P0, :] = v[0]
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
            a = blk.attn.proj(y.transpose(1, 2).reshape(1, P0, self.E).contiguous())
            x = blk.norm_2(a + x)
            return blk.norm_4(blk.mlp(blk.norm_3(x)) + x)

        x = tr.wte(x_ids) * self.emb_scale
        for i, blk in enumerate(tr.prelude):
            x = blk_pre(blk, x, i)
        self.e_store[:P0] = x[0]
        e = x
        s = self.noise[:P0][None]
        for k_step in range(1, self.T + 1):
            x = tr.adapter(torch.cat([s, e], dim=-1))
            for j, blk in enumerate(tr.core_block):
                x = blk_pre(blk, x, self.P + self.Lc_core * (k_step - 1) + j)
            s = x
        self.Hs[:P0] = s[0]
        x = tr.ln_f(s)
        for c_i, blk in enumerate(tr.coda):
            x = blk_pre(blk, x, self.cslot + c_i)
        logits = self.m.lm_head(tr.ln_f(x[:, -1:]))[0, -1, : self.nv].float()
        return int(logits.argmax().item())

    @torch.no_grad()
    def _sync_backfill(self, lo: int, hi: int):
        """Synchronous full-depth Core 1..T + Coda backfill for newly committed slice `lo .. hi-1`."""
        if hi <= lo:
            return
        tr = self.m.transformer
        n = hi - lo
        freqs = self.freqs[:, lo:hi]
        mask = self._causal_tail_mask(lo, hi)
        e = self.e_store[lo:hi][None]
        s = self.noise[lo:hi][None]
        for k_step in range(1, self.T + 1):
            x = tr.adapter(torch.cat([s, e], dim=-1))
            for j, blk in enumerate(tr.core_block):
                slot = self.P + self.Lc_core * (k_step - 1) + j
                x = self._run_segmented_block(blk, x, freqs, [(0, n, slot, lo, hi, hi, mask)])
            s = x
        self.Hs[lo:hi] = s[0]
        x = tr.ln_f(s)
        for c_i, blk in enumerate(tr.coda):
            x = self._run_segmented_block(blk, x, freqs, [(0, n, self.cslot + c_i, lo, hi, hi, mask)])

    def next_props(self, top: list, c: int) -> list:
        if self.B == 0:
            return []
        off = self.set_off[c]
        return [int(top[off + j]) for j in range(self.B)]

    @torch.no_grad()
    def step_block(self, lc: int, hist: list, ids_blk: list, forced_s0_blk: torch.Tensor = None):
        """Execute one MARS Late-Entry block step at `lc` alongside active pipelined catch-up groups.

        Returns `(top_tokens_list, e_blk_tensor)` where `e_blk_tensor` has shape `(NB, E)`.
        """
        tr = self.m.transformer
        NB = self.NB
        pos_blk = lc + self.relpos  # (NB,)
        freqs_blk = self.freqs[0, pos_blk][None]  # (1, NB, 1, hd/2, 2)
        mask_blk = self._block_mask(lc)

        # 1. Prelude on active block (NB rows). Token slots 0..c write exact Prelude KV at lc..lc+c!
        x_blk = tr.wte(torch.tensor(ids_blk, dtype=torch.long, device=self.dev)[None]) * self.emb_scale
        for i, blk in enumerate(tr.prelude):
            x_blk = self._run_segmented_block(blk, x_blk, freqs_blk, [(0, NB, i, lc, lc + NB, lc + NB, mask_blk)])
        e_blk = x_blk  # (1, NB, E)

        # 2. K Core steps (kk = 0 .. K - 1), fusing active background groups + active block
        s_blk = self.noise[pos_blk][None] if forced_s0_blk is None else forced_s0_blk[None]
        coda_ready = None  # optional (lo, hi, s_T_tensor) when a background group finishes step T

        for kk in range(self.K):
            k_b = self.Ep + kk  # 1-based Core step for active block (in Ep..T)
            s_parts = []
            e_parts = []
            freq_parts = []
            meta_bg = []  # (r0, r1, pj, lo, hi, mask_g)
            cursor = 0

            if self.NG > 0:
                # Process older groups first (smaller positions first) for clean causal ordering
                act = active_groups(self.K, self.NG, kk, self.T)
                for g in reversed(act):
                    lo, hi = int(hist[g]), int(hist[g - 1])
                    if hi <= lo:
                        continue
                    pj = self.K * (g - 1) + kk + 1
                    n_g = hi - lo
                    s_g = self.noise[lo:hi] if pj == 1 else self.Hs[lo:hi]
                    s_parts.append(s_g[None])
                    e_parts.append(self.e_store[lo:hi][None])
                    freq_parts.append(self.freqs[:, lo:hi])
                    meta_bg.append((cursor, cursor + n_g, pj, lo, hi, self._causal_tail_mask(lo, hi)))
                    cursor += n_g

            r_blk0 = cursor
            s_parts.append(s_blk)
            e_parts.append(e_blk)
            freq_parts.append(freqs_blk)

            s_all = s_parts[0] if len(s_parts) == 1 else torch.cat(s_parts, dim=1)
            e_all = e_parts[0] if len(e_parts) == 1 else torch.cat(e_parts, dim=1)
            f_all = freq_parts[0] if len(freq_parts) == 1 else torch.cat(freq_parts, dim=1)

            x_all = tr.adapter(torch.cat([s_all, e_all], dim=-1))
            for j, blk in enumerate(tr.core_block):
                segs = [
                    (r0, r1, self.P + self.Lc_core * (pj - 1) + j, lo, hi, hi, m_g)
                    for (r0, r1, pj, lo, hi, m_g) in meta_bg
                ]
                segs.append((r_blk0, r_blk0 + NB, self.P + self.Lc_core * (k_b - 1) + j, lc, lc + NB, lc + NB, mask_blk))
                x_all = self._run_segmented_block(blk, x_all, f_all, segs)

            for r0, r1, pj, lo, hi, _ in meta_bg:
                s_out_g = x_all[0, r0:r1]
                if pj < self.T:
                    self.Hs[lo:hi] = s_out_g
                else:
                    self.Hs[lo:hi] = s_out_g
                    coda_ready = (lo, hi, s_out_g)
            s_blk = x_all[:, r_blk0 : r_blk0 + NB]

        if self.K == self.T:
            # Token slots ran full depth 1..T
            self.Hs[lc : lc + self.D_blk] = s_blk[0, : self.D_blk]

        # 3. Coda readout (fusing `coda_ready` background group if one finished step T in this step)
        if coda_ready is not None:
            lo, hi, s_fin = coda_ready
            n_g = hi - lo
            mask_g = self._causal_tail_mask(lo, hi)
            x_coda = tr.ln_f(torch.cat([s_fin[None], s_blk], dim=1))
            f_coda = torch.cat([self.freqs[:, lo:hi], freqs_blk], dim=1)
            for c_i, blk in enumerate(tr.coda):
                slot_c = self.cslot + c_i
                segs = [
                    (0, n_g, slot_c, lo, hi, hi, mask_g),
                    (n_g, n_g + NB, slot_c, lc, lc + NB, lc + NB, mask_blk),
                ]
                x_coda = self._run_segmented_block(blk, x_coda, f_coda, segs)
            x_blk_out = x_coda[:, n_g : n_g + NB]
        else:
            x_coda = tr.ln_f(s_blk)
            for c_i, blk in enumerate(tr.coda):
                slot_c = self.cslot + c_i
                x_coda = self._run_segmented_block(
                    blk, x_coda, freqs_blk, [(0, NB, slot_c, lc, lc + NB, lc + NB, mask_blk)]
                )
            x_blk_out = x_coda

        logits_blk = self.m.lm_head(tr.ln_f(x_blk_out))[0, :, : self.nv].float()
        top = logits_blk.argmax(-1).tolist()
        return top, e_blk[0]

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
        chk_len = 0

        if cuda:
            torch.cuda.synchronize()
        t_dec0 = time.perf_counter()

        while not done and (len(known) - P0) < max_new:
            if prop_override_fn is not None:
                ov = prop_override_fn(len(commits), lc, known)
                if ov is not None:
                    props = list(ov)

            filler = [self.mask_id] * (self.B - len(props))
            ids_blk = [known[-1]] + props + filler + [self.mask_id] * (self.NB - self.D_blk)
            top, e_blk = self.step_block(lc, hist, ids_blk)

            np_ = len(props)
            c = 0
            while c < np_ and top[c] == props[c]:
                c += 1
            if np_ > 0:
                n_prop_steps += 1
                n_acc_total += c

            commit = props[:c] + [int(top[c])]
            # Save exact Prelude embeddings for committed token slots 0..c (`lc .. lc + c`)
            self.e_store[lc : lc + c + 1] = e_blk[: c + 1]
            if self.bg == "sync" and self.K < self.T:
                self._sync_backfill(lc, lc + c + 1)

            props = self.next_props(top, c)
            lc_next = lc + c + 1
            hist = [lc_next] + hist[: self.NG]
            lc = lc_next

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

            cs.append(c)
            commits.append(emitted_this_step)

            if not done and tok is not None and until and ((len(known) - P0) - chk_len >= 8):
                chk_len = len(known) - P0
                txt_now = tok.decode(known[P0:], skip_special_tokens=True)
                for u in until:
                    if u and u in txt_now:
                        done = True
                        why = "until"
                        break

        if cuda:
            torch.cuda.synchronize()
        t_dec = time.perf_counter() - t_dec0

        gen_ids = known[P0:]
        text = tok.decode(gen_ids, skip_special_tokens=True) if tok is not None else ""
        for u in until:
            if u and u in text:
                text = text.split(u)[0]

        n_steps = len(commits)
        dec_tokens = sum(commits)  # tokens generated by step_block loop (excluding prefill's 1st token)
        return dict(
            gen_ids=gen_ids,
            text=text,
            tokens=len(gen_ids),
            dec_tokens=dec_tokens,
            steps=n_steps,
            c_list=cs,
            mean_c=(n_acc_total / n_prop_steps) if n_prop_steps > 0 else 0.0,
            mean_block_len=(dec_tokens / n_steps) if n_steps > 0 else 1.0,
            prefill_s=t_pref,
            decode_s=t_dec,
            total_s=t_pref + t_dec,
            decode_tok_per_s=(dec_tokens / t_dec) if t_dec > 0 else 0.0,
            finish=why,
            prompt_len=P0,
            lc_final=lc,
            hist_final=hist,
        )


def score_block_records(records: list, gold_map: dict = None, ref_map: dict = None):
    n = len(records)
    if n == 0:
        return dict(n=0)
    s_hits, f_hits = 0, 0
    has_gold = bool(gold_map)
    tot_toks, tot_dec_toks, tot_steps = 0, 0, 0
    tot_dec_s, tot_pref_s = 0.0, 0.0
    c_hist = {}
    exact_matches = 0
    per_doc = {}

    for r in records:
        did = int(r["doc_id"])
        if has_gold and did in gold_map:
            sc = score_one(r.get("text", ""), gold_map[did])
            s_hits += sc["strict"]
            f_hits += sc["flex"]
        else:
            sc = dict(strict=0, flex=0)
        tot_toks += int(r["tokens"])
        tot_dec_toks += int(r["dec_tokens"])
        tot_steps += int(r["steps"])
        tot_dec_s += float(r["decode_s"])
        tot_pref_s += float(r["prefill_s"])
        for c_val in r.get("c_list", []):
            c_hist[str(c_val)] = c_hist.get(str(c_val), 0) + 1
        if ref_map and did in ref_map:
            if r["gen_ids"] == ref_map[did]["gen_ids"]:
                exact_matches += 1
        per_doc[did] = dict(strict=sc["strict"], flex=sc["flex"], tokens=int(r["tokens"]))

    cfg_str = records[0]["cfg"]
    pc = parse_blk_cfg(cfg_str)
    mean_blk = tot_dec_toks / max(1, tot_steps)
    # Sequential layer calls per block step: 2 Prelude + 4*K Core + 2 Coda = 4 + 4*K
    seq_layers_per_step = 4.0 + 4.0 * pc["K"]
    seq_layers_per_tok = seq_layers_per_step / max(1e-9, mean_blk)
    r32_vals = [float(r["same_gpu_r32_tok_per_s"]) for r in records if r.get("same_gpu_r32_tok_per_s")]
    mean_r32_tps = (sum(r32_vals) / len(r32_vals)) if r32_vals else None
    return dict(
        cfg=cfg_str,
        kind=pc.get("kind", "blk"),
        K=pc["K"],
        B=pc["B"],
        T=pc["T"],
        bg=pc["bg"],
        n=n,
        has_gold=has_gold,
        strict_acc=(s_hits / n) if has_gold else None,
        flex_acc=(f_hits / n) if has_gold else None,
        strict_hits=s_hits if has_gold else None,
        flex_hits=f_hits if has_gold else None,
        mean_tokens=tot_toks / n,
        mean_steps=tot_steps / n,
        mean_block_len=mean_blk,
        seq_layers_per_tok=seq_layers_per_tok,
        theo_layer_speedup_vs_r32=132.0 / seq_layers_per_tok,
        tot_dec_tokens=tot_dec_toks,
        tot_decode_s=tot_dec_s,
        decode_tok_per_s=tot_dec_toks / max(1e-9, tot_dec_s),
        same_gpu_r32_tok_per_s=mean_r32_tps,
        end2end_tok_per_s=tot_toks / max(1e-9, tot_pref_s + tot_dec_s),
        mean_prefill_ms=1000.0 * tot_pref_s / n,
        mean_decode_s=tot_dec_s / n,
        exact_seq_match=(exact_matches / n) if ref_map else None,
        c_hist=c_hist,
        per_doc=per_doc,
    )


def load_merged_block_dir(out_dir: str) -> dict:
    """Load and merge both unsharded `{cfg}.jsonl` and sharded `{cfg}.s*.jsonl` per cfg sorted by doc_id."""
    by_cfg_doc = {}
    patterns = [os.path.join(out_dir, "blk_*.jsonl"), os.path.join(out_dir, "eblk_*.jsonl")]
    fps = sorted({fp for pat in patterns for fp in glob.glob(pat)})
    for fp in fps:
        if fp.endswith(".part.jsonl") or fp.endswith(".tmp"):
            continue
        recs = read_jsonl(fp)
        for r in recs:
            c = r["cfg"]
            by_cfg_doc.setdefault(c, {})[int(r["doc_id"])] = r
    return {c: [by_cfg_doc[c][d] for d in sorted(by_cfg_doc[c])] for c in by_cfg_doc}


def summarize_block_dir(out_dir: str, gold_path: str, tag: str = ""):
    gold_map = load_gold(gold_path) if (gold_path and os.path.exists(gold_path)) else None
    raw_by_cfg = load_merged_block_dir(out_dir)

    ref_map = None
    if "blk:32:0" in raw_by_cfg:
        ref_map = {int(r["doc_id"]): r for r in raw_by_cfg["blk:32:0"]}

    scored = {cfg: score_block_records(recs, gold_map, ref_map=ref_map) for cfg, recs in raw_by_cfg.items()}
    base_tps = scored["blk:32:0"]["decode_tok_per_s"] if "blk:32:0" in scored else None

    rows = []
    for cfg, sc in sorted(scored.items(), key=lambda kv: (kv[1].get("kind", "blk"), kv[1]["B"], -kv[1]["K"], kv[1]["bg"])):
        row = {k: v for k, v in sc.items() if k != "per_doc"}
        # Prefer exact common-doc blk:32:0 decode_tok_per_s when blk:32:0 covers the same docs!
        ref_tps = base_tps or sc.get("same_gpu_r32_tok_per_s")
        if ref_map:
            common_ids = [int(r["doc_id"]) for r in raw_by_cfg[cfg] if int(r["doc_id"]) in ref_map]
            if len(common_ids) == len(raw_by_cfg[cfg]) and len(common_ids) > 0:
                c_toks = sum(int(ref_map[i]["dec_tokens"]) for i in common_ids)
                c_sec = sum(float(ref_map[i]["decode_s"]) for i in common_ids)
                if c_sec > 0:
                    ref_tps = c_toks / c_sec
        row["wall_speedup_vs_r32"] = (sc["decode_tok_per_s"] / ref_tps) if ref_tps else None
        if gold_map and "blk:32:0" in scored and cfg != "blk:32:0":
            row["vs_r32_flex"] = paired_compare(sc["per_doc"], scored["blk:32:0"]["per_doc"], "flex")
        rows.append(row)

    out_json = os.path.join(out_dir, "summary_block.json")
    with open(out_json + ".tmp", "w") as f:
        json.dump(dict(tag=tag, base_decode_tok_per_s=base_tps, rows=rows), f, indent=2)
    os.replace(out_json + ".tmp", out_json)

    lines = [
        f"# Huginn MARS + Late-Entry / Early-Exit Block Decoding Summary (`{tag}`)",
        "",
        "| Config | Kind | B | K | BG | N | Strict EM | Flex EM | `1+E[c]` Tok/Step | Seq Layers/Tok | Decode Tok/s | H100 Speedup | Exact Match vs `r:32` |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        sp = f"{r['wall_speedup_vs_r32']:.2f}x" if r["wall_speedup_vs_r32"] is not None else "-"
        ex = f"{r['exact_seq_match']*100:.1f}%" if r["exact_seq_match"] is not None else "-"
        st_s = f"{r['strict_acc']*100:.2f}%" if r["strict_acc"] is not None else "-"
        fl_s = f"{r['flex_acc']*100:.2f}%" if r["flex_acc"] is not None else "-"
        lines.append(
            f"| `{r['cfg']}` | {r.get('kind', 'blk')} | {r['B']} | {r['K']} | {r['bg']} | {r['n']} | "
            f"{st_s} | {fl_s} | {r['mean_block_len']:.2f} | "
            f"{r['seq_layers_per_tok']:.1f} | {r['decode_tok_per_s']:.2f} | {sp} | {ex} |"
        )
    out_md = os.path.join(out_dir, "summary_block.md")
    with open(out_md + ".tmp", "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(out_md + ".tmp", out_md)
    print(f"[block-summary] wrote {len(rows)} configs -> {out_json} and {out_md}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-docs", type=int, default=200)
    ap.add_argument("--doc-lo", type=int, default=0)
    ap.add_argument("--doc-hi", type=int, default=0)
    ap.add_argument("--shard-tag", default="")
    ap.add_argument("--cfgs", default="blk:32:0,blk:32:4,blk:16:4,blk:12:4,blk:8:4,blk:4:4")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--speed-anchor-docs", type=int, default=0)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.summary_only:
        summarize_block_dir(args.out_dir, args.gold, tag=args.tag)
        return

    m, tok, mask_id = load_hg_model_and_tok(args.model, dtype=torch.bfloat16, device="cuda")
    print(f"[block-eval] loaded {args.tag} from {args.model} (vocab={m.config.padded_vocab_size}, mask_id={mask_id})", flush=True)

    all_raw = read_jsonl(args.tasks)
    if args.doc_hi > 0:
        raw = all_raw[args.doc_lo : args.doc_hi]
    else:
        raw = all_raw[args.doc_lo : args.doc_lo + args.n_docs]
    assert len(raw) > 0, f"Empty slice doc_lo={args.doc_lo} doc_hi={args.doc_hi} on {args.tasks}"

    docs = [
        dict(
            doc_id=int(r["doc_id"]),
            ids=tok.encode(r["context"], add_special_tokens=False),
            until=list(dict.fromkeys(list(r.get("until", [])) + ["<|end_turn|>", "<|end_text|>"])),
        )
        for r in raw
    ]
    gold_map = load_gold(args.gold) if (args.gold and os.path.exists(args.gold)) else None
    L_max = max(len(d["ids"]) for d in docs) + args.max_new + 32
    dev = next(m.parameters()).device
    dt = next(m.parameters()).dtype

    same_gpu_r32_tps = None
    if args.speed_anchor_docs > 0:
        n_anc = min(args.speed_anchor_docs, len(docs))
        anc_dec = HuginnBlockDecoder(m, K=32, B=0, L_max=L_max, T=32, bg="none", nv=NV_BASE, mask_id=mask_id)
        w_ns = make_noise(m, L_max, doc_seed(args.seed, 999999), dev, dt)
        _ = anc_dec.decode_one(docs[0]["ids"][:64], w_ns, max_new=12, tok=tok, until=())
        anc_toks, anc_sec = 0, 0.0
        for d in docs[:n_anc]:
            ns = make_noise(m, L_max, doc_seed(args.seed, d["doc_id"]), dev, dt)
            ra = anc_dec.decode_one(d["ids"], ns, max_new=args.max_new, tok=tok, until=d["until"])
            anc_toks += int(ra["dec_tokens"])
            anc_sec += float(ra["decode_s"])
        same_gpu_r32_tps = anc_toks / max(1e-9, anc_sec)
        print(f"[speed-anchor:{args.tag}] same-GPU blk:32:0 over {n_anc} docs = {same_gpu_r32_tps:.3f} tok/s ({anc_toks} toks / {anc_sec:.1f}s)", flush=True)
        del anc_dec
        gc.collect()
        torch.cuda.empty_cache()

    sfx = f".{args.shard_tag}" if args.shard_tag else ""
    for cfg in [c.strip() for c in args.cfgs.split(",") if c.strip()]:
        pc = parse_blk_cfg(cfg)
        fn = os.path.join(args.out_dir, f"{cfg.replace(':', '_')}{sfx}.jsonl")
        if os.path.exists(fn) and len(read_jsonl(fn)) >= len(docs):
            sc = score_block_records(read_jsonl(fn)[: len(docs)], gold_map)
            if cfg == "blk:32:0" and same_gpu_r32_tps is None:
                same_gpu_r32_tps = sc["decode_tok_per_s"]
            fl_str = f"{sc['flex_acc']*100:.2f}%" if sc["flex_acc"] is not None else "n/a"
            print(
                f"[skip] {cfg}{sfx} already complete ({sc['n']} docs): flex={fl_str} "
                f"1+E[c]={sc['mean_block_len']:.2f} tok/s={sc['decode_tok_per_s']:.2f}",
                flush=True,
            )
            continue

        dec = HuginnBlockDecoder(
            m,
            K=pc["K"],
            B=pc["B"],
            L_max=L_max,
            T=pc["T"],
            bg=pc["bg"],
            nv=NV_BASE,
            mask_id=mask_id,
        )
        # Warmup 1 short decode on GPU so Triton/cuBLAS kernels are compiled before timing
        w_ns = make_noise(m, L_max, doc_seed(args.seed, 999999), dev, dt)
        _ = dec.decode_one(docs[0]["ids"][:64], w_ns, max_new=12, tok=tok, until=())

        recs = []
        t0 = time.time()
        for idx, d in enumerate(docs):
            ns = make_noise(m, L_max, doc_seed(args.seed, d["doc_id"]), dev, dt)
            r = dec.decode_one(d["ids"], ns, max_new=args.max_new, tok=tok, until=d["until"])
            r["doc_id"] = d["doc_id"]
            r["cfg"] = cfg
            if same_gpu_r32_tps is not None:
                r["same_gpu_r32_tok_per_s"] = same_gpu_r32_tps
            recs.append(r)
            if (idx + 1) % 25 == 0 or (idx + 1) == len(docs):
                sc_now = score_block_records(recs, gold_map)
                fl_str = f"{sc_now['flex_acc']*100:.2f}%" if sc_now["flex_acc"] is not None else "n/a"
                print(
                    f"[{args.tag}{sfx} {cfg}] {idx + 1}/{len(docs)} ({time.time() - t0:.1f}s) | "
                    f"flex={fl_str} | 1+E[c]={sc_now['mean_block_len']:.2f} | "
                    f"tok/s={sc_now['decode_tok_per_s']:.2f}",
                    flush=True,
                )

        if cfg == "blk:32:0" and same_gpu_r32_tps is None:
            sc_r32 = score_block_records(recs, gold_map)
            same_gpu_r32_tps = sc_r32["decode_tok_per_s"]
            for r in recs:
                r["same_gpu_r32_tok_per_s"] = same_gpu_r32_tps

        with open(fn + ".tmp", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        os.replace(fn + ".tmp", fn)
        del dec
        gc.collect()
        torch.cuda.empty_cache()

    if not args.shard_tag:
        summarize_block_dir(args.out_dir, args.gold, tag=args.tag)


if __name__ == "__main__":
    main()
