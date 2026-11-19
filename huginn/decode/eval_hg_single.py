"""Single-token autoregressive GSM8K-CoT evaluation and teacher-forced mechanism probe
for Base Huginn-0125, Stage-1 ar_sft, and Stage-2 mars_* checkpoints.

Always slices output logits to `nv` (default 65536) so `<|mask|>` (65536) and padded rows
(65537..65599) cannot be emitted during autoregressive decoding.
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


def stride_steps(D: int, t: int):
    assert 1 <= t <= D, (D, t)
    if t == 1:
        return [D]
    ks = [1 + ((i * (D - 1) + (t - 1) // 2) // (t - 1)) for i in range(t)]
    assert len(ks) == t and ks == sorted(set(ks)) and ks[0] == 1 and ks[-1] == D
    return ks


def parse_single_cfg(cfg: str):
    f = cfg.split(":")
    mo = f[0]
    if mo in ("r", "ws"):
        D = int(f[1])
        return dict(mode=mo, D=D, t=D, ks=list(range(1, D + 1)))
    if mo in ("late", "latezero", "early", "wslate"):
        D, t = int(f[1]), int(f[2])
        assert 1 <= t <= D, cfg
        ks = list(range(1, t + 1)) if mo == "early" else list(range(D - t + 1, D + 1))
        return dict(mode=mo, D=D, t=t, ks=ks)
    if mo == "anchor":
        D, a, b = int(f[1]), int(f[2]), int(f[3])
        t = a + b
        assert a >= 1 and b >= 1 and t <= D, cfg
        ks = list(range(1, a + 1)) + list(range(D - b + 1, D + 1))
        return dict(mode=mo, D=D, a=a, b=b, t=t, ks=ks)
    if mo == "stride":
        D, t = int(f[1]), int(f[2])
        assert 1 <= t <= D, cfg
        return dict(mode=mo, D=D, t=t, ks=stride_steps(D, t))
    raise ValueError(f"Unknown single-token cfg: {cfg!r}")


class SftHuginnDecoder:
    """Batched greedy single-token decoder with explicit per-recurrence-step KV cache and `:nv` logit slice."""

    def __init__(self, m, cfg: str, B: int, L: int, nv: int = NV_BASE):
        self.m = m
        self.cfg = parse_single_cfg(cfg)
        self.nv = min(int(nv), int(m.config.padded_vocab_size))
        c = m.config
        tr = m.transformer
        self.P, self.Lc, self.C = len(tr.prelude), len(tr.core_block), len(tr.coda)
        self.D = self.cfg["D"]
        self.mode = self.cfg["mode"]
        self.t = self.cfg["t"]
        self.ks = self.cfg["ks"]
        self.ks_set = set(self.ks)
        self.H = c.num_attention_heads
        self.hd = c.n_embd // c.num_attention_heads
        self.E = c.n_embd
        self.S = self.P + self.Lc * self.D + self.C
        self.B, self.L = B, L
        p = next(m.parameters())
        self.dev, self.dt = p.device, p.dtype
        self.K = torch.zeros(self.S, B, L, self.H, self.hd, device=self.dev, dtype=self.dt)
        self.V = torch.zeros_like(self.K)
        self.pos = torch.zeros(B, dtype=torch.long, device=self.dev)
        self.bidx = torch.arange(B, device=self.dev)
        self.noise = torch.zeros(B, L, self.E, device=self.dev, dtype=self.dt)
        self.prev_lat = torch.zeros(B, 1, self.E, device=self.dev, dtype=self.dt)
        self.emb_scale = float(m.emb_scale)
        self.freqs = ensure_freqs_cis(m, L + 8, self.dev)
        self.cslot = self.P + self.Lc * self.D
        self.keep_logits = False

    def _qkv(self, attn, h, freqs):
        B, R, _ = h.shape
        q, k, v = attn.Wqkv(h).split(attn.chunks, dim=2)
        q = q.view(B, R, self.H, self.hd)
        k = k.view(B, R, self.H, self.hd)
        v = v.view(B, R, self.H, self.hd)
        if self.m.config.qk_bias:
            qb, kb = attn.qk_bias.split(1, dim=0)
            q, k = (q + qb).to(q.dtype), (k + kb).to(q.dtype)
        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs)
        return q, k, v

    def _attn_dec(self, attn, h, freqs, slot, cols, mask, Lk):
        B, R, E = h.shape
        q, k, v = self._qkv(attn, h, freqs)
        Ks, Vs = self.K[slot], self.V[slot]
        Ks[self.bidx[:, None], cols] = k
        Vs[self.bidx[:, None], cols] = v
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            Ks[:, :Lk].transpose(1, 2),
            Vs[:, :Lk].transpose(1, 2),
            attn_mask=mask,
            dropout_p=0.0,
        )
        return attn.proj(y.transpose(1, 2).reshape(B, R, E).contiguous())

    def _block_dec(self, blk, x, freqs, slot, cols, mask, Lk):
        a = self._attn_dec(blk.attn, blk.norm_1(x), freqs, slot, cols, mask, Lk)
        x = blk.norm_2(a + x)
        return blk.norm_4(blk.mlp(blk.norm_3(x)) + x)

    def _core_dec(self, s, e, k, freqs, cols, mask, Lk):
        x = self.m.transformer.adapter(torch.cat([s, e.expand(-1, s.shape[1], -1)], dim=-1))
        for j, blk in enumerate(self.m.transformer.core_block):
            x = self._block_dec(blk, x, freqs, self.P + self.Lc * (k - 1) + j, cols, mask, Lk)
        return x

    def _readout_dec(self, s, freqs, cols, mask, Lk):
        tr = self.m.transformer
        x = tr.ln_f(s)
        for c, blk in enumerate(tr.coda):
            x = self._block_dec(blk, x, freqs, self.cslot + c, cols, mask, Lk)
        return self.m.lm_head(tr.ln_f(x))[..., : self.nv].float()

    @torch.no_grad()
    def prefill(self, b: int, ids, noise: torch.Tensor):
        tr = self.m.transformer
        S0 = len(ids)
        assert S0 + 2 <= self.L, (S0, self.L)
        self.noise[b].copy_(noise[: self.L])
        D = self.D
        x_ids = torch.tensor(ids, device=self.dev)[None]
        freqs = self.freqs[:, :S0]

        def blk_pre(blk, x, slot):
            h = blk.norm_1(x)
            q, k, v = self._qkv(blk.attn, h, freqs)
            self.K[slot, b, 1 : S0 + 1] = k[0]
            self.V[slot, b, 1 : S0 + 1] = v[0]
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.0, is_causal=True
            )
            a = blk.attn.proj(y.transpose(1, 2).reshape(1, S0, self.E).contiguous())
            x = blk.norm_2(a + x)
            return blk.norm_4(blk.mlp(blk.norm_3(x)) + x)

        x = tr.wte(x_ids) * self.emb_scale
        for i, blk in enumerate(tr.prelude):
            x = blk_pre(blk, x, i)
        e = x
        s = self.noise[b, :S0][None]
        for k in range(1, D + 1):
            x = tr.adapter(torch.cat([s, e], dim=-1))
            for j, blk in enumerate(tr.core_block):
                x = blk_pre(blk, x, self.P + self.Lc * (k - 1) + j)
            s = x
        lat = tr.ln_f(s)
        x = lat
        for c, blk in enumerate(tr.coda):
            x = blk_pre(blk, x, self.cslot + c)
        logits = self.m.lm_head(tr.ln_f(x[:, -1:]))[..., : self.nv].float()
        self.pos[b] = S0
        self.prev_lat[b] = lat[0, -1:]
        return int(logits[0, -1].argmax()), logits[0, -1]

    def _masks(self, Lk: int, kinds: str):
        cols = torch.arange(Lk, device=self.dev)[None]
        p = self.pos[:, None]
        ms, cs = [], []
        for kd in kinds:
            if kd == "c":
                ms.append((cols >= 1) & (cols <= p + 1))
                cs.append(p[:, 0] + 1)
            else:
                ms.append((cols == 0) | ((cols >= 1) & (cols <= p)))
                cs.append(torch.zeros_like(p[:, 0]))
        return torch.stack(cs, 1), torch.stack(ms, 1)[:, None]

    @torch.no_grad()
    def step(self, tok: torch.Tensor, active: torch.Tensor):
        tr = self.m.transformer
        B, D, mo = self.B, self.D, self.mode
        Lk = int(self.pos.max()) + 2
        freqs = self.freqs[0, self.pos][:, None]
        cC, mC = self._masks(Lk, "c")
        cCS, mCS = self._masks(Lk, "cs")
        cSc, mSc = self._masks(Lk, "s")
        x = tr.wte(tok[:, None]) * self.emb_scale
        for i, blk in enumerate(tr.prelude):
            x = self._block_dec(blk, x, freqs, i, cC, mC, Lk)
        e = x
        s0 = self.noise[self.bidx, self.pos][:, None]
        out = dict(agree=None)
        if mo in ("r", "ws", "early"):
            s = self.prev_lat.clone() if mo == "ws" else s0
            t = self.t
            early_logits = None
            for k in range(1, D + 1):
                s = self._core_dec(s, e, k, freqs, cC, mC, Lk)
                if mo == "early" and k == t and t < D:
                    early_logits = self._readout_dec(s, freqs, cSc, mSc, Lk)[:, 0]
            lg = self._readout_dec(s, freqs, cC, mC, Lk)
            clean = emit = lg[:, 0]
            if early_logits is not None:
                emit = early_logits
            nxt = emit.argmax(-1)
            if mo == "early":
                out["agree"] = nxt == clean.argmax(-1)
            crit = torch.full((B,), t if mo != "r" else D, device=self.dev)
            rowsteps = torch.full((B,), D, device=self.dev)
            lat = tr.ln_f(s)
        else:
            t = self.t
            s_clean = s0
            s_scr = None
            for k in range(1, D + 1):
                if k in self.ks_set:
                    if s_scr is None:
                        if mo == "wslate":
                            s_scr = self.prev_lat.clone()
                        elif mo == "latezero":
                            s_scr = torch.zeros_like(s0)
                        else:
                            s_scr = s0
                    pair = self._core_dec(torch.cat([s_clean, s_scr], dim=1), e, k, freqs, cCS, mCS, Lk)
                    s_clean, s_scr = pair[:, :1], pair[:, 1:]
                else:
                    s_clean = self._core_dec(s_clean, e, k, freqs, cC, mC, Lk)
            lg = self._readout_dec(torch.cat([s_clean, s_scr], dim=1), freqs, cCS, mCS, Lk)
            clean, emit = lg[:, 0], lg[:, 1]
            nxt = emit.argmax(-1)
            out["agree"] = nxt == clean.argmax(-1)
            crit = torch.full((B,), t, device=self.dev)
            rowsteps = torch.full((B,), D + t, device=self.dev)
            lat = tr.ln_f(s_clean)

        self.prev_lat[active] = lat[active]
        self.pos[active] += 1
        out.update(next=nxt, crit=crit, rowsteps=rowsteps, emit_logits=emit if self.keep_logits else None)
        return out


def run_single_docs(
    m,
    tok,
    cfg: str,
    docs: list,
    B: int,
    max_new: int = 256,
    stop_ids=STOP_IDS,
    until=(),
    seed: int = 20261023,
    nv: int = NV_BASE,
    on_done=None,
    log=print,
):
    dev = next(m.parameters()).device
    dt = next(m.parameters()).dtype
    pc_tmp = parse_single_cfg(cfg)
    if str(dev).startswith("cuda"):
        safe_b = 12 if pc_tmp["D"] >= 24 else 16
        B = min(B, safe_b)
    B = min(B, max(1, len(docs)))
    L = max(len(d["ids"]) for d in docs) + max_new + 2
    dec = SftHuginnDecoder(m, cfg, B, L, nv=nv)
    queue = list(docs)
    rows = [None] * B
    tokbuf = torch.zeros(B, dtype=torch.long, device=dev)
    active = torch.zeros(B, dtype=torch.bool, device=dev)
    n_done = 0
    t_start = time.time()
    out_recs = []

    def finish(b, why):
        nonlocal n_done
        r = rows[b]
        ids = r["gen"]
        text = tok.decode(ids, skip_special_tokens=True)
        for u in until:
            if u and u in text:
                text = text.split(u)[0]
        rec = dict(
            doc_id=r["doc_id"],
            cfg=cfg,
            gen_ids=ids,
            text=text,
            tokens=len(ids),
            finish=why,
            crit_steps=r["crit"],
            row_steps=r["rows"],
            agree=(r["agree"] / r["agree_n"]) if r["agree_n"] else None,
            agree_n=r["agree_n"],
            sec=time.time() - r["t0"],
            prompt_len=r["plen"],
        )
        rows[b] = None
        active[b] = False
        n_done += 1
        out_recs.append(rec)
        if on_done:
            on_done(rec)

    def stop_reason(b):
        r = rows[b]
        ids = r["gen"]
        if ids and ids[-1] in stop_ids:
            return "eos"
        if len(ids) >= max_new:
            return "length"
        if until and (len(ids) - r["chk"] >= 8):
            r["chk"] = len(ids)
            txt = tok.decode(ids, skip_special_tokens=True)
            for u in until:
                if u and u in txt:
                    return "until"
        return None

    step_i = 0
    while queue or any(rows):
        for b in range(B):
            while rows[b] is None and queue:
                d = queue.pop(0)
                t0 = time.time()
                ns = make_noise(m, L, doc_seed(seed, d["doc_id"]), dev, dt)
                first, _ = dec.prefill(b, d["ids"], ns)
                rows[b] = dict(
                    doc_id=d["doc_id"],
                    plen=len(d["ids"]),
                    gen=[first],
                    crit=dec.D,
                    rows=dec.D,
                    agree=0,
                    agree_n=0,
                    t0=t0,
                    chk=0,
                )
                tokbuf[b] = first
                active[b] = True
                why = stop_reason(b)
                if why:
                    finish(b, why)
        if not any(rows):
            break
        o = dec.step(tokbuf, active)
        nxt = o["next"].tolist()
        cr = o["crit"].tolist()
        rs = o["rowsteps"].tolist()
        ag = o["agree"].tolist() if o["agree"] is not None else None
        for b in range(B):
            if rows[b] is None:
                continue
            r = rows[b]
            r["gen"].append(nxt[b])
            r["crit"] += cr[b]
            r["rows"] += rs[b]
            if ag is not None:
                r["agree"] += int(ag[b])
                r["agree_n"] += 1
            tokbuf[b] = nxt[b]
            why = stop_reason(b)
            if why:
                finish(b, why)
        step_i += 1
        if step_i % 64 == 0:
            log(f"[{cfg}] step={step_i} done={n_done}/{n_done + len(queue) + sum(r is not None for r in rows)} elapsed={time.time() - t_start:.1f}s")
    out_recs.sort(key=lambda r: int(r["doc_id"]))
    return out_recs


class TeacherForcedProbe:
    """Vectorized teacher-forced mechanism evaluator over generated positions (`:nv` logit slice)."""

    def __init__(self, m, nv: int = NV_BASE):
        self.m = m
        self.nv = min(int(nv), int(m.config.padded_vocab_size))
        c = m.config
        tr = m.transformer
        self.P, self.Lc, self.C = len(tr.prelude), len(tr.core_block), len(tr.coda)
        self.H = c.num_attention_heads
        self.hd = c.n_embd // c.num_attention_heads
        self.E = c.n_embd
        self.emb_scale = float(m.emb_scale)
        self.freqs = m.freqs_cis
        self.dev = next(m.parameters()).device
        self.dt = next(m.parameters()).dtype

    def _qkv(self, attn, h, freqs):
        B, T, _ = h.shape
        q, k, v = attn.Wqkv(h).split(attn.chunks, dim=2)
        q = q.view(B, T, self.H, self.hd)
        k = k.view(B, T, self.H, self.hd)
        v = v.view(B, T, self.H, self.hd)
        if self.m.config.qk_bias:
            qb, kb = attn.qk_bias.split(1, dim=0)
            q, k = (q + qb).to(q.dtype), (k + kb).to(q.dtype)
        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs)
        return q, k, v

    @torch.no_grad()
    def build_clean_context(self, seq_ids, noise: torch.Tensor, D: int, record_states: bool = False):
        tr = self.m.transformer
        T = len(seq_ids)
        S = self.P + self.Lc * D + self.C
        cslot = self.P + self.Lc * D
        Kc = torch.empty(S, T, self.H, self.hd, device=self.dev, dtype=self.dt)
        Vc = torch.empty_like(Kc)
        freqs = self.freqs[:, :T]

        def blk_causal(blk, x, slot):
            q, k, v = self._qkv(blk.attn, blk.norm_1(x), freqs)
            Kc[slot] = k[0]
            Vc[slot] = v[0]
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.0, is_causal=True
            )
            a = blk.attn.proj(y.transpose(1, 2).reshape(1, T, self.E).contiguous())
            x = blk.norm_2(a + x)
            return blk.norm_4(blk.mlp(blk.norm_3(x)) + x)

        x = tr.wte(torch.tensor(seq_ids, device=self.dev)[None]) * self.emb_scale
        for i, blk in enumerate(tr.prelude):
            x = blk_causal(blk, x, i)
        e = x[0]
        s = noise[:T][None]
        states = {0: s[0].clone()} if record_states else None
        for k in range(1, D + 1):
            x = tr.adapter(torch.cat([s, e[None]], dim=-1))
            for j, blk in enumerate(tr.core_block):
                x = blk_causal(blk, x, self.P + self.Lc * (k - 1) + j)
            s = x
            if record_states:
                states[k] = s[0].clone()
        lat = tr.ln_f(s)
        x = lat
        for c, blk in enumerate(tr.coda):
            x = blk_causal(blk, x, cslot + c)
        logits = self.m.lm_head(tr.ln_f(x))[0, :, : self.nv].float()
        return dict(Kc=Kc, Vc=Vc, e=e, lat=lat[0], states=states, logits=logits, D=D, T=T, cslot=cslot)

    @torch.no_grad()
    def eval_trajectory(self, ctx: dict, S0: int, s_init: torch.Tensor, ks: list):
        tr = self.m.transformer
        T, cslot = ctx["T"], ctx["cslot"]
        G = T - S0
        freqs_g = self.freqs[:, S0:T]
        e_g = ctx["e"][S0:T][None]
        idx_g = torch.arange(G, device=self.dev)[:, None]
        idx_t = torch.arange(T, device=self.dev)[None, :]
        mask_past = idx_t < (S0 + idx_g)
        mask_self = torch.eye(G, dtype=torch.bool, device=self.dev)
        attn_mask = torch.cat([mask_past, mask_self], dim=1)[None, None]

        def blk_intervene(blk, x, slot):
            q, k, v = self._qkv(blk.attn, blk.norm_1(x), freqs_g)
            Kall = torch.cat([ctx["Kc"][slot][None], k], dim=1)
            Vall = torch.cat([ctx["Vc"][slot][None], v], dim=1)
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), Kall.transpose(1, 2), Vall.transpose(1, 2), attn_mask=attn_mask, dropout_p=0.0
            )
            a = blk.attn.proj(y.transpose(1, 2).reshape(1, G, self.E).contiguous())
            x = blk.norm_2(a + x)
            return blk.norm_4(blk.mlp(blk.norm_3(x)) + x)

        s = s_init[None].to(self.dt)
        for k in ks:
            x = tr.adapter(torch.cat([s, e_g], dim=-1))
            for j, blk in enumerate(tr.core_block):
                x = blk_intervene(blk, x, self.P + self.Lc * (k - 1) + j)
            s = x
        x = tr.ln_f(s)
        for c, blk in enumerate(tr.coda):
            x = blk_intervene(blk, x, cslot + c)
        return self.m.lm_head(tr.ln_f(x))[0, :, : self.nv].float()


def score_single_records(records, gold_map):
    n = len(records)
    if n == 0:
        return dict(n=0)
    s_hits, f_hits, toks, crits, ags = 0, 0, 0, 0, []
    per_doc = {}
    for r in records:
        did = int(r["doc_id"])
        sc = score_one(r.get("text", ""), gold_map[did])
        s_hits += sc["strict"]
        f_hits += sc["flex"]
        toks += int(r.get("tokens", 0))
        crits += int(r.get("crit_steps", 0))
        if r.get("agree") is not None:
            ags.append(float(r["agree"]))
        per_doc[did] = dict(strict=sc["strict"], flex=sc["flex"], tokens=int(r.get("tokens", 0)))
    cfg = records[0].get("cfg", "")
    pc = parse_single_cfg(cfg) if cfg else {}
    mean_crit = crits / max(toks, 1)
    crit_layers = 4.0 + 4.0 * mean_crit
    return dict(
        cfg=cfg,
        mode=pc.get("mode", ""),
        D=pc.get("D"),
        t=pc.get("t"),
        n=n,
        strict_acc=s_hits / n,
        flex_acc=f_hits / n,
        strict_hits=s_hits,
        flex_hits=f_hits,
        mean_tokens=toks / n,
        mean_crit_steps_per_tok=mean_crit,
        mean_crit_layers_per_tok=crit_layers,
        layer_speedup_vs_r32=132.0 / crit_layers,
        mean_agree=(sum(ags) / len(ags)) if ags else None,
        per_doc=per_doc,
    )


def summarize_single_dir(out_dir: str, gold_path: str, tag: str = ""):
    gold_map = load_gold(gold_path)
    by_cfg_doc = {}
    for fp in sorted(glob.glob(os.path.join(out_dir, "*.jsonl"))):
        if fp.endswith(".part.jsonl") or fp.endswith(".tmp") or os.path.basename(fp).startswith("_"):
            continue
        for r in read_jsonl(fp):
            by_cfg_doc.setdefault(r["cfg"], {})[int(r["doc_id"])] = r
    scored = {
        c: score_single_records([by_cfg_doc[c][d] for d in sorted(by_cfg_doc[c])], gold_map)
        for c in by_cfg_doc
        if by_cfg_doc[c]
    }

    rows = []
    for cfg, sc in sorted(scored.items(), key=lambda kv: (kv[1]["D"] or 0, kv[1]["t"] or 99, kv[0])):
        row = {k: v for k, v in sc.items() if k != "per_doc"}
        D, t = sc["D"], sc["t"]
        if "r:32" in scored and cfg != "r:32":
            row["vs_r32_flex"] = paired_compare(sc["per_doc"], scored["r:32"]["per_doc"], "flex")
        if t is not None and f"r:{t}" in scored and cfg != f"r:{t}":
            row["vs_rt_flex"] = paired_compare(sc["per_doc"], scored[f"r:{t}"]["per_doc"], "flex")
        if D is not None and t is not None and f"early:{D}:{t}" in scored and cfg != f"early:{D}:{t}":
            row["vs_early_flex"] = paired_compare(sc["per_doc"], scored[f"early:{D}:{t}"]["per_doc"], "flex")
        rows.append(row)

    out_json = os.path.join(out_dir, "summary_single.json")
    with open(out_json, "w") as f:
        json.dump(dict(tag=tag, n_cfgs=len(rows), rows=rows), f, indent=2)

    lines = [
        f"# Huginn Single-Token GSM8K-CoT Summary ({tag})",
        "",
        "| Config | Mode | D | K | Crit Layers | Layer Speedup | N | Strict EM | Flex EM | Agree w/ D | Mean Tok | vs early(D,K) Flex (p) | vs r(K) Flex (p) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        ve = r.get("vs_early_flex")
        vr = r.get("vs_rt_flex")
        ve_s = f"{ve['diff']*100:+.2f}% ({ve['p_sign']:.3g})" if ve else "-"
        vr_s = f"{vr['diff']*100:+.2f}% ({vr['p_sign']:.3g})" if vr else "-"
        ag_s = f"{r['mean_agree']*100:.1f}%" if r["mean_agree"] is not None else "-"
        lines.append(
            f"| `{r['cfg']}` | {r['mode']} | {r['D']} | {r['mean_crit_steps_per_tok']:.1f} | "
            f"{r['mean_crit_layers_per_tok']:.1f} | {r['layer_speedup_vs_r32']:.2f}x | {r['n']} | "
            f"{r['strict_acc']*100:.2f}% | {r['flex_acc']*100:.2f}% | {ag_s} | {r['mean_tokens']:.1f} | "
            f"{ve_s} | {vr_s} |"
        )
    out_md = os.path.join(out_dir, "summary_single.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[single-summary] wrote {len(rows)} configs -> {out_json}", flush=True)


def run_probe_cli(m, tok, tasks_path: str, out_dir: str, tag: str, n_docs: int = 100, B_gen: int = 32, seed: int = 20261023):
    os.makedirs(out_dir, exist_ok=True)
    raw = read_jsonl(tasks_path)[:n_docs]
    docs = [dict(doc_id=int(r["doc_id"]), ids=tok.encode(r["context"], add_special_tokens=False)) for r in raw]
    until = list(raw[0].get("until", [])) + ["<|end_text|>"]

    ref_path = os.path.join(out_dir, "_ref_r32.jsonl")
    if os.path.exists(ref_path) and len(read_jsonl(ref_path)) >= n_docs:
        ref_recs = read_jsonl(ref_path)[:n_docs]
        print(f"[probe:{tag}] loaded cached r:32 trajectories ({len(ref_recs)} docs) from {ref_path}", flush=True)
    else:
        print(f"[probe:{tag}] generating r:32 reference trajectories for {len(docs)} docs...", flush=True)
        ref_recs = run_single_docs(m, tok, "r:32", docs, B=B_gen, until=until, seed=seed)
        with open(ref_path, "w") as f:
            for rec in ref_recs:
                f.write(json.dumps(rec) + "\n")

    probe = TeacherForcedProbe(m, nv=NV_BASE)
    ks_grid = [2, 4, 6, 8, 12, 16, 20, 24, 28, 32]
    stats = {
        f"{m_name}:{K}": dict(agree_hits=0, kl_sum=0.0, n_toks=0)
        for m_name in ("late", "late_zero", "early", "r")
        for K in ks_grid
    }
    stats["oralate:8"] = dict(agree_hits=0, kl_sum=0.0, n_toks=0)

    prompt_map = {d["doc_id"]: d["ids"] for d in docs}
    t0 = time.time()
    for idx, rec in enumerate(ref_recs):
        did = int(rec["doc_id"])
        p_ids = prompt_map[did]
        g_ids = [int(x) for x in rec["gen_ids"] if int(x) < NV_BASE]
        if len(g_ids) < 2:
            continue
        seq_ids = p_ids + g_ids[:-1]  # predict g_ids[0..G-1] from positions S0-1..T-1; probe generated positions S0..T-1
        S0 = len(p_ids)
        G = len(seq_ids) - S0
        if G <= 0:
            continue
        ns = make_noise(m, len(seq_ids) + 4, doc_seed(seed, did), probe.dev, probe.dt)
        ctx32 = probe.build_clean_context(seq_ids, ns, D=32, record_states=(idx == 0))
        ref_lg = ctx32["logits"][S0:]  # (G, NV_BASE)
        ref_top = ref_lg.argmax(-1)
        ref_lp = F.log_softmax(ref_lg, dim=-1)
        ref_p = ref_lp.exp()

        def accum(key, lg):
            lp = F.log_softmax(lg, dim=-1)
            kl = float((ref_p * (ref_lp - lp)).sum(-1).sum().item())
            ag = int((lg.argmax(-1) == ref_top).sum().item())
            stats[key]["agree_hits"] += ag
            stats[key]["kl_sum"] += kl
            stats[key]["n_toks"] += G

        if idx == 0:
            lg_ora = probe.eval_trajectory(ctx32, S0, ctx32["states"][24][S0:], list(range(25, 33)))
            accum("oralate:8", lg_ora)

        s0_g = ns[S0 : len(seq_ids)]
        zero_g = torch.zeros_like(s0_g)
        for K in ks_grid:
            if K == 32:
                accum(f"late:{K}", ref_lg)
                accum(f"early:{K}", ref_lg)
                accum(f"r:{K}", ref_lg)
                lg_z32 = probe.eval_trajectory(ctx32, S0, zero_g, list(range(1, 33)))
                accum(f"late_zero:{K}", lg_z32)
                continue
            ks_late = list(range(32 - K + 1, 33))
            ks_early = list(range(1, K + 1))
            accum(f"late:{K}", probe.eval_trajectory(ctx32, S0, s0_g, ks_late))
            accum(f"late_zero:{K}", probe.eval_trajectory(ctx32, S0, zero_g, ks_late))
            accum(f"early:{K}", probe.eval_trajectory(ctx32, S0, s0_g, ks_early))
            ctx_k = probe.build_clean_context(seq_ids, ns, D=K, record_states=False)
            accum(f"r:{K}", ctx_k["logits"][S0:])

        if (idx + 1) % 20 == 0 or idx + 1 == len(ref_recs):
            print(f"[probe:{tag}] {idx + 1}/{len(ref_recs)} docs ({time.time() - t0:.1f}s)", flush=True)

    summary = dict(tag=tag, n_docs=len(ref_recs), grid=ks_grid, results={})
    for k, v in stats.items():
        nt = max(1, v["n_toks"])
        summary["results"][k] = dict(
            top1_agree=v["agree_hits"] / nt,
            mean_kl=v["kl_sum"] / nt,
            n_toks=v["n_toks"],
        )
    out_json = os.path.join(out_dir, f"probe_{tag}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        f"# Huginn Teacher-Forced Mechanism Probe (`{tag}`, {len(ref_recs)} GSM8K-CoT docs)",
        "",
        "| K | `late(32,K)` Agree | `late_zero(32,K)` Agree | `early(32,K)` Agree | `r(K)` Agree | `late` KL | `early` KL | `r(K)` KL |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for K in ks_grid:
        rl = summary["results"][f"late:{K}"]
        rz = summary["results"][f"late_zero:{K}"]
        re_ = summary["results"][f"early:{K}"]
        rr = summary["results"][f"r:{K}"]
        lines.append(
            f"| {K} | {rl['top1_agree']*100:.2f}% | {rz['top1_agree']*100:.2f}% | "
            f"{re_['top1_agree']*100:.2f}% | {rr['top1_agree']*100:.2f}% | "
            f"{rl['mean_kl']:.3f} | {re_['mean_kl']:.3f} | {rr['mean_kl']:.3f} |"
        )
    out_md = os.path.join(out_dir, f"probe_{tag}.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[probe:{tag}] saved -> {out_json} and {out_md}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=["probe", "decode"], required=True)
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-docs", type=int, default=200)
    ap.add_argument("--doc-lo", type=int, default=0)
    ap.add_argument("--doc-hi", type=int, default=0)
    ap.add_argument("--shard-tag", default="")
    ap.add_argument("--cfgs", default="r:32,late:32:16,late:32:12,late:32:8,late:32:4")
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.summary_only:
        summarize_single_dir(args.out_dir, args.gold, tag=args.tag)
        return

    m, tok, mask_id = load_hg_model_and_tok(args.model, dtype=torch.bfloat16, device="cuda")
    print(f"[eval-single] loaded {args.tag} from {args.model} (vocab={m.config.padded_vocab_size}, mask_id={mask_id})", flush=True)

    if args.mode == "probe":
        run_probe_cli(m, tok, args.tasks, args.out_dir, args.tag, n_docs=args.n_docs, B_gen=args.B, seed=args.seed)
        return

    all_raw = read_jsonl(args.tasks)
    if args.doc_hi > 0:
        raw = all_raw[args.doc_lo : args.doc_hi]
    else:
        raw = all_raw[args.doc_lo : args.doc_lo + args.n_docs]
    assert len(raw) > 0, f"Empty slice doc_lo={args.doc_lo} doc_hi={args.doc_hi}"

    docs = [dict(doc_id=int(r["doc_id"]), ids=tok.encode(r["context"], add_special_tokens=False)) for r in raw]
    until = list(dict.fromkeys(list(raw[0].get("until", [])) + ["<|end_turn|>", "<|end_text|>"]))
    gold_map = load_gold(args.gold) if (args.gold and os.path.exists(args.gold)) else None
    sfx = f".{args.shard_tag}" if args.shard_tag else ""

    for cfg in [c.strip() for c in args.cfgs.split(",") if c.strip()]:
        fn = os.path.join(args.out_dir, f"{cfg.replace(':', '_')}{sfx}.jsonl")
        if os.path.exists(fn) and len(read_jsonl(fn)) >= len(docs):
            if gold_map:
                sc = score_single_records(read_jsonl(fn)[: len(docs)], gold_map)
                print(f"[skip] {cfg}{sfx} already complete ({sc['n']} docs): flex={sc['flex_acc']*100:.2f}%", flush=True)
            else:
                print(f"[skip] {cfg}{sfx} already complete ({len(docs)} docs)", flush=True)
            continue
        t0 = time.time()
        recs = run_single_docs(m, tok, cfg, docs, B=args.B, max_new=args.max_new, until=until, seed=args.seed)
        with open(fn + ".tmp", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        os.replace(fn + ".tmp", fn)
        if gold_map:
            sc = score_single_records(recs, gold_map)
            print(
                f"[done:{args.tag}{sfx}] {cfg} ({time.time() - t0:.1f}s) | strict={sc['strict_acc']*100:.2f}% "
                f"flex={sc['flex_acc']*100:.2f}% | agree={sc['mean_agree']} | tok={sc['mean_tokens']:.1f}",
                flush=True,
            )
        else:
            print(f"[done:{args.tag}{sfx}] {cfg} ({time.time() - t0:.1f}s, {len(recs)} docs)", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    if not args.shard_tag and gold_map:
        summarize_single_dir(args.out_dir, args.gold, tag=args.tag)


if __name__ == "__main__":
    main()
