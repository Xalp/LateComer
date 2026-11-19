"""CUDA-Graphed Paged FlashAttention-3 Block Decoder for Huginn-3.5B (T = 32).

Eliminates the eager Python multi-segment boolean-mask SDPA dispatch bottleneck in
`hg_block_dec.py` by executing the entire Prelude -> K Core steps -> Coda -> lm_head
decode step inside a single captured `torch.cuda.CUDAGraph`, with a single paged
FlashAttention-3 (`sgl_kernel.flash_attn.flash_attn_with_kvcache`, page_size=1,
bottom-right causal) call per layer across all pipelined catch-up groups and all
`(1 + B)^2` MARS mask-placeholder branches.

Supports:
  - Sequential baseline:      `blk:32:0`
  - Lossless full block:      `blk:32:8`, `blk:32:4`
  - Pipelined late entry:     `blk:<K>:<B>` or `blk:<K>:<B>:pipe` (full 1..32 catch-up)
  - Partial-depth catch-up:   `blk:<K>:<B>:p16` (17..32, 2-stage) / `blk:<K>:<B>:p24` (9..32, 3-stage)
  - Synchronous catch-up:     `blk:<K>:<B>:sync`
  - No-catch-up ablation:     `blk:<K>:<B>:none`
  - Trained early-exit block: `eblk:<K>:<B>` (initial K passes over depth-K cache, same engine)
"""

import argparse
import gc
import json
import math
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
    make_noise,
    read_jsonl,
)
from huginn.model.raven_modeling_minimal import precompute_freqs_cis
from huginn.decode.hg_block_dec import score_block_records, summarize_block_dir


def parse_fast_cfg(cfg: str, T_default: int = 32):
    """Parse `blk:<K>:<B>[:<bg>]` or `eblk:<K>:<B>`, supporting `bg` in {pipe, p16, p24, sync, none}."""
    parts = cfg.split(":")
    assert len(parts) in (3, 4) and parts[0] in ("blk", "eblk"), f"Invalid cfg: {cfg!r}"
    kind = parts[0]
    K = int(parts[1])
    B = int(parts[2])
    if kind == "eblk":
        assert len(parts) == 3 and 1 <= K <= int(T_default) and B >= 0, (cfg, K, B)
        return dict(cfg=cfg, kind="eblk", K=K, B=B, T=K, bg="none", D_bg=0)
    T = int(T_default)
    assert 1 <= K <= T and B >= 0, (cfg, K, B, T)
    bg = parts[3] if len(parts) == 4 else ("none" if K == T else "pipe")
    if K == T:
        bg = "none"
    if bg == "pipe":
        D_bg = T
    elif bg.startswith("p") and bg[1:].isdigit():
        D_bg = int(bg[1:])
        assert K < D_bg <= T, f"Invalid partial catch-up depth {D_bg} for K={K}, T={T}"
    elif bg in ("sync", "none"):
        D_bg = T if bg == "sync" else 0
    else:
        raise AssertionError(f"Unsupported bg mode in {cfg!r}")
    return dict(cfg=cfg, kind="blk", K=K, B=B, T=T, bg=bg, D_bg=D_bg)


def n_groups_for(K: int, D_bg: int, bg: str) -> int:
    if bg in ("none", "sync") or D_bg <= 0:
        return 0
    return -(-D_bg // K)


def active_bg_groups(K: int, T: int, D_bg: int, NG: int, kk: int):
    """Return list of `(g, pj)` in oldest-first order (`g` from `NG` down to `1`) active in Core call `kk`."""
    if NG <= 0:
        return []
    ep_bg = T - D_bg + 1
    out = []
    for g in range(NG, 0, -1):
        pj = (ep_bg - 1) + K * (g - 1) + kk + 1
        if pj <= T:
            out.append((g, pj))
    return out


_FA3 = None


def _get_fa3():
    global _FA3
    if _FA3 is None:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        _FA3 = flash_attn_with_kvcache
    return _FA3


class _LayerMeta:
    __slots__ = ("page_table", "seqlens", "cu_q", "max_q", "wslot")

    def __init__(self, nseq: int, S: int, N: int, cu_list: list, dev):
        self.cu_q = torch.tensor(cu_list, dtype=torch.int32, device=dev)
        self.max_q = max(cu_list[i + 1] - cu_list[i] for i in range(nseq))
        self.page_table = torch.zeros((nseq, S), dtype=torch.int32, device=dev)
        self.seqlens = torch.zeros((nseq,), dtype=torch.int32, device=dev)
        self.wslot = torch.zeros((N,), dtype=torch.int64, device=dev)


def attn_torch_paged(q: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, meta: _LayerMeta, NH: int, HD: int, HD_kv: int = None) -> torch.Tensor:
    """Exact fp32/fp64 reference matching bottom-right causal `flash_attn_with_kvcache` (page_size=1)."""
    if HD_kv is None:
        HD_kv = HD
    N = q.shape[0]
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    out = torch.zeros((N, NH, HD), dtype=q.dtype, device=q.device)
    pt = meta.page_table.cpu()
    sl = meta.seqlens.cpu().tolist()
    cu = meta.cu_q.cpu().tolist()
    kf = kc.view(-1, NH, HD_kv)[:, :, :HD]
    vf = vc.view(-1, NH, HD_kv)[:, :, :HD]
    scale = HD ** -0.5
    for s in range(len(sl)):
        q0, q1, Lk = cu[s], cu[s + 1], sl[s]
        if q1 == q0 or Lk <= 0:
            continue
        idx = pt[s, :Lk].long().to(q.device)
        K_mat = kf[idx].to(acc)
        V_mat = vf[idx].to(acc)
        Q_mat = q[q0:q1].view(-1, NH, HD).to(acc)
        sc = torch.einsum("qhd,khd->hqk", Q_mat, K_mat) * scale
        Lq = q1 - q0
        ii = torch.arange(Lq, device=q.device)[:, None]
        jj = torch.arange(Lk, device=q.device)[None, :]
        vis = jj <= (Lk - Lq + ii)
        sc = sc.masked_fill(~vis[None], float("-inf"))
        p = torch.softmax(sc, dim=-1)
        p = torch.where(vis.any(-1)[None, :, None], p, torch.zeros_like(p))
        out[q0:q1] = torch.einsum("hqk,khd->qhd", p, V_mat).to(q.dtype)
    return out.view(N, NH * HD)


def attn_fa3_paged(q: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, meta: _LayerMeta, NH: int, HD: int, HD_kv: int = None) -> torch.Tensor:
    FA = _get_fa3()
    if HD_kv is None:
        HD_kv = HD
    N = q.shape[0]
    sm_scale = 1.0 / math.sqrt(HD)
    qv = q.view(N, NH, HD)
    if HD_kv > HD:
        qv = F.pad(qv, (0, HD_kv - HD))
    else:
        qv = qv.contiguous()
    o = FA(
        qv,
        kc,
        vc,
        cache_seqlens=meta.seqlens,
        page_table=meta.page_table,
        cu_seqlens_q=meta.cu_q,
        max_seqlen_q=meta.max_q,
        softmax_scale=sm_scale,
        causal=True,
    )
    o = o[0] if isinstance(o, (tuple, list)) else o
    if HD_kv > HD:
        o = o[:, :, :HD].contiguous()
    return o.reshape(N, NH * HD)


class HuginnFastOps:
    """Fused RMSNorm, SwiGLU, and interleaved complex RoPE with automatic accuracy validation."""

    def __init__(self, m, max_seq: int, fused: bool = True, log=print):
        c = m.config
        p = next(m.parameters())
        self.dev, self.dt = p.device, p.dtype
        self.E = int(c.n_embd)
        self.NH = int(c.num_attention_heads)
        self.HD = self.E // self.NH
        self.INTER = int(c.intermediate_size)
        self.eps = float(c.norm_eps)
        self.log = log

        fc = ensure_freqs_cis(m, max_seq, self.dev)  # (1, max_len, 1, HD/2, 2) float32
        self.freqs = fc
        cos32 = fc[0, :, 0, :, 0].float().contiguous()
        sin32 = fc[0, :, 0, :, 1].float().contiguous()
        self.cos_sin32 = torch.cat([cos32, sin32], dim=-1).contiguous()

        self.chosen = {"rms": "torch", "add_rms": "torch", "silu": "torch", "rope": "torch"}
        self._norm_mods = {}
        self.rms = self.t_rms
        self.add_rms = self.t_add_rms
        self.silu_mul = self.t_silu_mul
        self.rope = self.t_rope

        if fused and str(self.dev).startswith("cuda") and self.dt == torch.bfloat16:
            self._pick_fused()

    def t_rms(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * w

    def t_add_rms(self, x: torch.Tensor, fresh_res: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return self.t_rms(fresh_res + x, w)

    def t_silu_mul(self, gu: torch.Tensor) -> torch.Tensor:
        g, u = gu[:, : self.INTER], gu[:, self.INTER :]
        return F.silu(g) * u

    def t_rope(self, q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor):
        N = q.shape[0]
        f = self.freqs.index_select(1, pos)  # (1, N, 1, HD/2, 2)
        qv = q.view(1, N, self.NH, self.HD)
        kv = k.view(1, N, self.NH, self.HD)
        qr, kr = apply_rotary_emb_complex_like(qv, kv, freqs_cis=f)
        return qr.reshape(N, self.E), kr.reshape(N, self.E)

    def _pick_fused(self):
        dev, dt, E, INTER, HD = self.dev, self.dt, self.E, self.INTER, self.HD
        xt = torch.randn(9, E, device=dev, dtype=dt)
        wt = torch.rand(E, device=dev, dtype=dt) + 0.5
        try:
            from sglang.srt.layers.layernorm import RMSNorm as SglRMS

            def mk(w):
                key = w.data_ptr()
                if key not in self._norm_mods:
                    mod = SglRMS(E, eps=self.eps).to(dev, dt)
                    mod.weight = torch.nn.Parameter(w, requires_grad=False)
                    self._norm_mods[key] = mod
                return self._norm_mods[key]

            def s_rms(x, w):
                return mk(w)(x)

            def s_add_rms(x, fresh_res, w):
                out = mk(w)(x, fresh_res)
                return out[0] if isinstance(out, tuple) else out

            y = s_rms(xt, wt)
            torch.cuda.synchronize()
            err = (y.float() - self.t_rms(xt, wt).float()).abs().max().item()
            if err < 0.05:
                self.rms = s_rms
                self.chosen["rms"] = f"sglang.RMSNorm(err={err:.4f})"
            rt = torch.randn(9, E, device=dev, dtype=dt)
            y2 = s_add_rms(xt.clone(), rt.clone(), wt)
            torch.cuda.synchronize()
            err2 = (y2.float() - self.t_add_rms(xt, rt, wt).float()).abs().max().item()
            if err2 < 0.06:
                self.add_rms = s_add_rms
                self.chosen["add_rms"] = f"sglang.RMSNorm.fused_add(err={err2:.4f})"
        except Exception as e:
            self.log(f"[hg-ops] RMSNorm fallback: {e!r}"[:200])

        for mod, name in (
            ("sglang.jit_kernel.activation", "silu_and_mul"),
            ("sgl_kernel", "silu_and_mul"),
            ("flashinfer.activation", "silu_and_mul"),
        ):
            try:
                fn = getattr(__import__(mod, fromlist=[name]), name)
                gu = torch.randn(7, 2 * INTER, device=dev, dtype=dt)
                out = torch.empty(7, INTER, device=dev, dtype=dt)
                fn(gu, out)
                torch.cuda.synchronize()
                err = (out.float() - self.t_silu_mul(gu).float()).abs().max().item()
                if err < 0.05:
                    def silu_k(x_gu, fn=fn):
                        o = torch.empty((x_gu.shape[0], INTER), device=x_gu.device, dtype=x_gu.dtype)
                        fn(x_gu, o)
                        return o
                    self.silu_mul = silu_k
                    self.chosen["silu"] = f"{mod}.{name}(err={err:.4f})"
                    break
            except Exception as e:
                self.log(f"[hg-ops] {mod}.{name} fallback: {e!r}"[:200])

        pos = torch.tensor([0, 1, 7, 31, 255, 1023, 1500], dtype=torch.int64, device=dev)
        for mod, name in (
            ("sgl_kernel", "apply_rope_with_cos_sin_cache_inplace"),
            ("flashinfer.rope", "apply_rope_with_cos_sin_cache_inplace"),
        ):
            try:
                fn = getattr(__import__(mod, fromlist=[name]), name)
                qkv0 = torch.randn(len(pos), 3 * E, device=dev, dtype=dt)
                rq, rk = self.t_rope(qkv0[:, :E].clone(), qkv0[:, E : 2 * E].clone(), pos)
                tab = self.cos_sin32
                # Probe zero-copy strided in-place RoPE directly on qkv[:, :E] and qkv[:, E:2*E]
                qkv_s = qkv0.clone()
                v_orig = qkv0[:, 2 * E :].clone()
                qs, ks = qkv_s[:, :E], qkv_s[:, E : 2 * E]
                strided_ok = False
                try:
                    fn(positions=pos, query=qs, key=ks, head_size=HD, cos_sin_cache=tab, is_neox=False)
                    torch.cuda.synchronize()
                    err_s = max(
                        (qs.float() - rq.float()).abs().max().item(),
                        (ks.float() - rk.float()).abs().max().item(),
                        (qkv_s[:, 2 * E :].float() - v_orig.float()).abs().max().item(),
                    )
                    strided_ok = err_s < 0.05
                except Exception:
                    strided_ok = False

                if strided_ok:
                    def rope_k(q_in, k_in, p_in, fn=fn, tab=tab, hd=HD):
                        fn(positions=p_in, query=q_in, key=k_in, head_size=hd, cos_sin_cache=tab, is_neox=False)
                        return q_in, k_in
                    self.rope = rope_k
                    self.chosen["rope"] = f"{mod}.rope_strided_inplace(err={err_s:.4f})"
                    break

                qt, kt = qkv0[:, :E].contiguous(), qkv0[:, E : 2 * E].contiguous()
                fn(positions=pos, query=qt, key=kt, head_size=HD, cos_sin_cache=tab, is_neox=False)
                torch.cuda.synchronize()
                err = max((qt.float() - rq.float()).abs().max().item(), (kt.float() - rk.float()).abs().max().item())
                if err < 0.05:
                    def rope_k(q_in, k_in, p_in, fn=fn, tab=tab, hd=HD):
                        q_c, k_c = q_in.contiguous(), k_in.contiguous()
                        fn(positions=p_in, query=q_c, key=k_c, head_size=hd, cos_sin_cache=tab, is_neox=False)
                        return q_c, k_c
                    self.rope = rope_k
                    self.chosen["rope"] = f"{mod}.rope_interleaved(err={err:.4f})"
                    break
            except Exception as e:
                self.log(f"[hg-ops] {mod}.{name} fallback: {e!r}"[:200])

        self.log(f"[hg-ops] chosen={self.chosen}")


def pack_block_weights(blk, dev, dt):
    """Extract contiguous weight/bias tensors for one Huginn SandwichBlock."""
    qkv_w = blk.attn.Wqkv.weight.detach().to(dev, dt).contiguous()
    E = qkv_w.shape[1]
    if blk.attn.config.qk_bias:
        qb, kb = blk.attn.qk_bias.detach().to(dev, dt).split(1, dim=0)
        zb = torch.zeros((E,), device=dev, dtype=dt)
        qkv_b = torch.cat([qb.reshape(E), kb.reshape(E), zb], dim=0).contiguous()
    else:
        qkv_b = None
    return dict(
        n1=blk.norm_1.weight.detach().to(dev, dt).contiguous(),
        qkv_w=qkv_w,
        qkv_b=qkv_b,
        o_w=blk.attn.proj.weight.detach().to(dev, dt).contiguous(),
        n2=blk.norm_2.weight.detach().to(dev, dt).contiguous(),
        n3=blk.norm_3.weight.detach().to(dev, dt).contiguous(),
        fc_w=blk.mlp.fc.weight.detach().to(dev, dt).contiguous(),
        down_w=blk.mlp.proj.weight.detach().to(dev, dt).contiguous(),
        n4=blk.norm_4.weight.detach().to(dev, dt).contiguous(),
    )


class _DirectCfgStub:
    def __init__(self, d: dict):
        self.n_embd = int(d["n_embd"])
        self.num_attention_heads = int(d.get("num_attention_heads", d.get("n_heads", 55)))
        self.intermediate_size = int(d["intermediate_size"])
        self.norm_eps = float(d.get("norm_eps", 1e-6))
        self.rope_base = float(d.get("rope_base", 50000.0))
        self.qk_bias = bool(d.get("qk_bias", True))
        self.padded_vocab_size = int(d.get("padded_vocab_size", d.get("vocab_size", 65536)))
        self.pad_token_id = int(d.get("pad_token_id", 65509))
        self.n_layers_in_prelude = int(d.get("n_layers_in_prelude", 2))
        self.n_layers_in_recurrent_block = int(d.get("n_layers_in_recurrent_block", 4))
        self.n_layers_in_coda = int(d.get("n_layers_in_coda", 2))
        eff = self.n_layers_in_prelude + self.n_layers_in_coda + self.n_layers_in_recurrent_block * int(d.get("mean_recurrence", 32))
        iv = d.get("init_values") or {}
        self.init_values = {
            "std": float(iv.get("std", math.sqrt(2.0 / (5.0 * self.n_embd)))),
            "embed_scale": float(iv.get("embed_scale", math.sqrt(self.n_embd))),
            "out_proj": float(iv.get("out_proj", math.sqrt(2.0 / (5.0 * self.n_embd)) / math.sqrt(2.0 * eff))),
        }


class _DirectModelStub:
    def __init__(self, cfg: _DirectCfgStub, freqs_cis: torch.Tensor):
        self.config = cfg
        self.freqs_cis = freqs_cis
        self.emb_scale = cfg.init_values["embed_scale"]


def load_hg_fast_direct(model_dir: str, device: str = "cuda", dtype=torch.bfloat16, max_seq: int = 2304, fused: bool = True, log=print):
    """Version-independent safetensors loader into `HuginnFastModel` + tokenizer."""
    from safetensors import safe_open
    from transformers import AutoTokenizer

    cfg_d = json.load(open(os.path.join(model_dir, "config.json")))
    cfg = _DirectCfgStub(cfg_d)
    dev = torch.device(device)

    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx_path))["weight_map"].values())) if os.path.exists(idx_path) else ["model.safetensors"]
    W = {}
    for fn in files:
        with safe_open(os.path.join(model_dir, fn), framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if k == "freqs_cis":
                    W[k] = t.to(device=dev, dtype=torch.float32)
                else:
                    W[k] = t.to(device=dev, dtype=dtype).contiguous()

    hd = cfg.n_embd // cfg.num_attention_heads
    fc = W.get("freqs_cis")
    if fc is None or fc.shape[1] < max_seq:
        fc = precompute_freqs_cis(hd, max(max_seq, 4096), cfg.rope_base, 1).to(device=dev, dtype=torch.float32)
    stub = _DirectModelStub(cfg, fc)

    def pack_prefix(pfx: str):
        qkv_w = W[pfx + "attn.Wqkv.weight"]
        E = qkv_w.shape[1]
        if cfg.qk_bias and (pfx + "attn.qk_bias") in W:
            qb, kb = W[pfx + "attn.qk_bias"].split(1, dim=0)
            qkv_b = torch.cat([qb.reshape(E), kb.reshape(E), torch.zeros((E,), device=dev, dtype=dtype)], dim=0).contiguous()
        else:
            qkv_b = None
        return dict(
            n1=W[pfx + "norm_1.weight"],
            qkv_w=qkv_w,
            qkv_b=qkv_b,
            o_w=W[pfx + "attn.proj.weight"],
            n2=W[pfx + "norm_2.weight"],
            n3=W[pfx + "norm_3.weight"],
            fc_w=W[pfx + "mlp.fc.weight"],
            down_w=W[pfx + "mlp.proj.weight"],
            n4=W[pfx + "norm_4.weight"],
        )

    fm = object.__new__(HuginnFastModel)
    fm.m = stub
    fm.dev, fm.dt = dev, dtype
    fm.E = cfg.n_embd
    fm.NH = cfg.num_attention_heads
    fm.HD = hd
    fm.HD_kv = hd
    fm.P = cfg.n_layers_in_prelude
    fm.Lc_core = cfg.n_layers_in_recurrent_block
    fm.C = cfg.n_layers_in_coda
    fm.emb_scale = stub.emb_scale
    fm.ops = HuginnFastOps.__new__(HuginnFastOps)
    # Initialize HuginnFastOps cleanly on `dev, dtype`
    ops = fm.ops
    ops.dev, ops.dt = dev, dtype
    ops.E, ops.NH, ops.HD, ops.INTER, ops.eps, ops.log = cfg.n_embd, cfg.num_attention_heads, hd, cfg.intermediate_size, cfg.norm_eps, log
    ops.freqs = fc
    ops.cos_sin32 = torch.cat([fc[0, :, 0, :, 0].float().contiguous(), fc[0, :, 0, :, 1].float().contiguous()], dim=-1).contiguous()
    ops.chosen = {"rms": "torch", "add_rms": "torch", "silu": "torch", "rope": "torch"}
    ops._norm_mods = {}
    ops.rms, ops.add_rms, ops.silu_mul, ops.rope = ops.t_rms, ops.t_add_rms, ops.t_silu_mul, ops.t_rope
    if fused and str(dev).startswith("cuda") and dtype == torch.bfloat16:
        ops._pick_fused()
    if str(dev).startswith("cuda") and dtype == torch.bfloat16:
        fm.HD_kv = fm._probe_fa3_hd(log=log)

    fm.wte = W["transformer.wte.weight"]
    fm.adap_w = W["transformer.adapter.weight"]
    fm.adap_b = W.get("transformer.adapter.bias")
    fm.ln_f = W["transformer.ln_f.weight"]
    fm.lm_head = W.get("lm_head.weight", fm.wte)
    fm.pre_layers = [pack_prefix(f"transformer.prelude.{i}.") for i in range(fm.P)]
    fm.core_layers = [pack_prefix(f"transformer.core_block.{i}.") for i in range(fm.Lc_core)]
    fm.coda_layers = [pack_prefix(f"transformer.coda.{i}.") for i in range(fm.C)]

    tok = AutoTokenizer.from_pretrained(model_dir, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vocab = tok.get_vocab()
    mask_id = tok.convert_tokens_to_ids("<|mask|>") if "<|mask|>" in vocab else cfg.pad_token_id
    return fm, tok, mask_id


class HuginnFastModel:
    """Shared unpacked weights + fast ops across multiple (K, B, bg) decoders on the same GPU."""

    def __init__(self, m, max_seq: int = 2304, fused: bool = True, log=print):
        self.m = m
        c = m.config
        tr = m.transformer
        p = next(m.parameters())
        self.dev, self.dt = p.device, p.dtype
        self.E = int(c.n_embd)
        self.NH = int(c.num_attention_heads)
        self.HD = self.E // self.NH
        self.HD_kv = self.HD
        self.P = len(tr.prelude)
        self.Lc_core = len(tr.core_block)
        self.C = len(tr.coda)
        self.emb_scale = float(m.emb_scale)
        self.ops = HuginnFastOps(m, max_seq=max_seq, fused=fused, log=log)
        if str(self.dev).startswith("cuda") and self.dt == torch.bfloat16:
            self.HD_kv = self._probe_fa3_hd(log=log)

        self.wte = tr.wte.weight.detach().to(self.dev, self.dt).contiguous()
        self.adap_w = tr.adapter.weight.detach().to(self.dev, self.dt).contiguous()
        self.adap_b = tr.adapter.bias.detach().to(self.dev, self.dt).contiguous() if tr.adapter.bias is not None else None
        self.ln_f = tr.ln_f.weight.detach().to(self.dev, self.dt).contiguous()
        self.lm_head = m.lm_head.weight.detach().to(self.dev, self.dt).contiguous()

        self.pre_layers = [pack_block_weights(b, self.dev, self.dt) for b in tr.prelude]
        self.core_layers = [pack_block_weights(b, self.dev, self.dt) for b in tr.core_block]
        self.coda_layers = [pack_block_weights(b, self.dev, self.dt) for b in tr.coda]

    def _probe_fa3_hd(self, log=print) -> int:
        """Probe whether FA3 supports `self.HD` natively (e.g. 96) or needs zero-pad to 128."""
        cands = [self.HD]
        p2 = 1 << (self.HD - 1).bit_length()
        if p2 != self.HD:
            cands.append(max(64, p2))
        for hd_try in cands:
            try:
                kc = torch.randn((16, 1, self.NH, hd_try), device=self.dev, dtype=self.dt)
                vc = torch.randn_like(kc)
                if hd_try > self.HD:
                    kc[:, :, :, self.HD :] = 0
                    vc[:, :, :, self.HD :] = 0
                q = torch.randn((4, self.E), device=self.dev, dtype=self.dt)
                mt = _LayerMeta(1, 16, 4, [0, 4], self.dev)
                mt.seqlens.fill_(8)
                mt.page_table[0, :8] = torch.arange(8, dtype=torch.int32, device=self.dev)
                out_fa = attn_fa3_paged(q, kc, vc, mt, self.NH, self.HD, hd_try)
                out_ref = attn_torch_paged(q, kc, vc, mt, self.NH, self.HD, hd_try)
                torch.cuda.synchronize()
                err = (out_fa.float() - out_ref.float()).abs().max().item()
                if err < 0.05:
                    log(f"[hg-fa3] selected HD_kv={hd_try} for HD={self.HD} (err={err:.5f})")
                    return hd_try
            except Exception as e:
                log(f"[hg-fa3] HD_kv={hd_try} unsupported: {e!r}"[:220])
        raise RuntimeError(f"No working FA3 head_dim found for HD={self.HD} among {cands}")

    def run_layer(self, lw: dict, x: torch.Tensor, pos: torch.Tensor, meta: _LayerMeta, kc: torch.Tensor, vc: torch.Tensor, attn_fn) -> torch.Tensor:
        ops = self.ops
        E, NH, HD, HD_kv = self.E, self.NH, self.HD, self.HD_kv
        h = ops.rms(x, lw["n1"])
        qkv = F.linear(h, lw["qkv_w"], lw["qkv_b"])
        q, k, v = qkv[:, :E], qkv[:, E : 2 * E], qkv[:, 2 * E :]
        q, k = ops.rope(q, k, pos)
        if HD_kv > HD:
            kp = F.pad(k.view(-1, NH, HD), (0, HD_kv - HD)).reshape(-1, NH * HD_kv)
            vp = F.pad(v.view(-1, NH, HD), (0, HD_kv - HD)).reshape(-1, NH * HD_kv)
            kc.view(-1, NH * HD_kv).index_copy_(0, meta.wslot, kp)
            vc.view(-1, NH * HD_kv).index_copy_(0, meta.wslot, vp)
        else:
            kc.view(-1, E).index_copy_(0, meta.wslot, k)
            vc.view(-1, E).index_copy_(0, meta.wslot, v)
        a_out = attn_fn(q, kc, vc, meta, NH, HD, HD_kv)
        a = F.linear(a_out, lw["o_w"])
        x = ops.add_rms(x, a, lw["n2"])
        m_in = ops.rms(x, lw["n3"])
        m_out = F.linear(ops.silu_mul(F.linear(m_in, lw["fc_w"])), lw["down_w"])
        return ops.add_rms(x, m_out, lw["n4"])


class _CoreCallPlan:
    pass


class HuginnFastDecoder:
    """Single-sequence MARS Late-Entry / Early-Exit block decoder with single-graph step execution."""

    def __init__(
        self,
        fm: HuginnFastModel,
        K: int,
        B: int,
        L_max: int = 1920,
        T: int = 32,
        bg: str = "pipe",
        D_bg: int = None,
        nv: int = NV_BASE,
        mask_id: int = None,
        stop_ids=STOP_IDS,
        attn: str = "paged",
        graphs: bool = None,
    ):
        self.fm = fm
        self.m = fm.m
        self.dev, self.dt = fm.dev, fm.dt
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

        c = self.m.config
        self.nv = min(int(nv), int(c.padded_vocab_size))
        self.mask_id = int(mask_id) if (mask_id is not None and int(mask_id) < int(c.padded_vocab_size)) else int(getattr(c, "pad_token_id", 0) or 0)
        self.stop_ids = set(int(x) for x in stop_ids)

        self.attn_mode = attn if (attn == "paged" and str(self.dev).startswith("cuda")) else "torch"
        self.attn_fn = attn_fa3_paged if self.attn_mode == "paged" else attn_torch_paged
        self.graphs_on = (str(self.dev).startswith("cuda") and self.attn_mode == "paged") if graphs is None else bool(graphs)

        # Build MARS (1 + B)^2 lattice geometry and bottom-right causal tail decomposition
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

        # Verify tail decomposition identically matches explicit boolean MARS mask
        wm = torch.zeros((NB, NB), dtype=torch.bool)
        for j in range(D):
            wm[j, : j + 1] = True
        for k_idx, (a, s) in enumerate(self.sets):
            off = offs[k_idx]
            for i in range(1, s + 1):
                r = off + i - 1
                wm[r, : a + 1] = True
                wm[r, off : r + 1] = True
        for sq, tl_ in enumerate(tails):
            for qi, r in enumerate(range(cub[sq], cub[sq + 1])):
                vis = set(tl_[: len(tl_) - (cub[sq + 1] - cub[sq]) + qi + 1])
                assert vis == set(wm[r].nonzero().flatten().tolist()), (self.B, sq, r)

        # Sequence & KV slot geometry
        self.L_max = int(L_max)
        self.S = self.L_max + self.TM + D + 8  # max page_table width (context + block tail)
        self.SL = self.S + max(PH, 1) + 8      # per-region slot stride (positions 0..S-1 + placeholder slots S..S+PH-1)
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

        # Pre-allocated device index helpers
        self.relpos = torch.tensor(relpos, dtype=torch.int64, device=self.dev)
        self.tailpad = tp.to(self.dev)
        self.lens = torch.tensor([len(x) for x in tails], dtype=torch.int64, device=self.dev)
        self.rr = torch.arange(NB, dtype=torch.int64, device=self.dev)
        self.ar = torch.arange(D, dtype=torch.int64, device=self.dev)
        self.is_tok = self.rr < D
        self.ph_slot = self.S + (self.rr - D)
        self.col = torch.arange(self.S, dtype=torch.int64, device=self.dev)[None, :]  # (1, S)

        # Static step input/output tensors
        self.hdr = torch.zeros((self.NG + 1,), dtype=torch.int64, device=self.dev)
        self.ids_blk = torch.full((NB,), self.mask_id, dtype=torch.int64, device=self.dev)
        self.top = torch.zeros((NB,), dtype=torch.int64, device=self.dev)

        # Per-layer static Meta structs for Prelude, Core (K x 4), and Coda
        cu_pre = cub
        self.meta_pre = [_LayerMeta(self.nblk, self.S, NB, cu_pre, self.dev) for _ in range(fm.P)]

        self.core_plans = []
        self.has_coda_bg = False
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
            if any(pj == self.T for _, pj in pl.act):
                self.has_coda_bg = True
            self.core_plans.append(pl)

        if self.has_coda_bg:
            cu_coda = [0, D] + [D + x for x in cub[1:]]
            self.meta_coda = [_LayerMeta(1 + self.nblk, self.S, D + NB, cu_coda, self.dev) for _ in range(fm.C)]
        else:
            self.meta_coda = [_LayerMeta(self.nblk, self.S, NB, cu_pre, self.dev) for _ in range(fm.C)]

        self.sync_hdr = torch.zeros((2,), dtype=torch.int64, device=self.dev)
        if self.bg == "sync" and self.K < self.T:
            n_sync_layers = fm.Lc_core * self.T + fm.C
            self.meta_sync = [_LayerMeta(1, self.S, D, [0, D], self.dev) for _ in range(n_sync_layers)]
        else:
            self.meta_sync = []

        if str(self.dev).startswith("cuda"):
            self.h_hdr = torch.zeros((self.NG + 1,), dtype=torch.int64).pin_memory()
            self.h_sync_hdr = torch.zeros((2,), dtype=torch.int64).pin_memory()
            self.h_ids = torch.zeros((NB,), dtype=torch.int64).pin_memory()
            self.h_top = torch.zeros((NB,), dtype=torch.int64).pin_memory()
            self.ev = torch.cuda.Event()

        self.graph = None
        self.sync_graph = None
        self._graph_pool = None
        self._sync_graph_pool = None
        if self.graphs_on:
            self._capture_step_graph()

    def _step_device(self):
        """Full Prelude -> K Core -> Coda -> argmax step (zero host sync, CUDA-graph safe)."""
        fm = self.fm
        D, NB, SL = self.D, self.NB, self.SL
        lc = self.hdr[0]
        col = self.col

        # Shared active-block page-table base & positions
        pos_blk = lc + self.relpos  # (NB,)
        wslot_loc = torch.where(self.is_tok, lc + self.rr, self.ph_slot)  # (NB,)
        rel = col - lc  # (1, S)
        tail = self.tailpad.gather(1, rel.clamp(0, self.TM - 1).expand(self.nblk, -1))
        pt_blk_base = torch.where(rel < 0, col, wslot_loc[tail])  # (nblk, S)
        pt_blk_valid = rel < self.lens[:, None]  # (nblk, S)
        sl_blk = (lc + self.lens).to(torch.int32)  # (nblk,)

        # 1. Prelude (2 layers on active block NB rows)
        x_blk = F.embedding(self.ids_blk, fm.wte) * fm.emb_scale
        for i in range(fm.P):
            mp = self.meta_pre[i]
            off = i * SL
            mp.wslot.copy_(off + wslot_loc)
            mp.seqlens.copy_(sl_blk)
            mp.page_table.copy_(torch.where(pt_blk_valid, off + pt_blk_base, 0).to(torch.int32))
            x_blk = fm.run_layer(fm.pre_layers[i], x_blk, pos_blk, mp, self.K_pool, self.V_pool, self.attn_fn)
        e_blk = x_blk
        # Store exact Prelude anchors for token slots lc .. lc + D - 1
        self.e_store.index_copy_(0, lc + self.ar, e_blk[:D])

        # 2. K Core steps, fusing all active catch-up groups + active block
        s_blk = self.noise.index_select(0, pos_blk)
        coda_fin = None  # (s_fin, posg_fin, real_fin, hi_fin)

        for kk in range(self.K):
            pl = self.core_plans[kk]
            s_parts, e_parts, pos_parts = [], [], []
            bg_info = []  # (gi, pj, posg, real, hi, pt_g_valid)

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

        # 3. Coda (2 layers, fusing `coda_fin` group if present)
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
            x_blk_out = x_coda[D : D + NB]
        else:
            x_coda = fm.ops.rms(s_blk, fm.ln_f)
            for c_i in range(fm.C):
                md = self.meta_coda[c_i]
                off_c = (self.cslot + c_i) * SL
                md.wslot.copy_(off_c + wslot_loc)
                md.seqlens.copy_(sl_blk)
                md.page_table.copy_(torch.where(pt_blk_valid, off_c + pt_blk_base, 0).to(torch.int32))
                x_coda = fm.run_layer(fm.coda_layers[c_i], x_coda, pos_blk, md, self.K_pool, self.V_pool, self.attn_fn)
            x_blk_out = x_coda

        logits = F.linear(fm.ops.rms(x_blk_out, fm.ln_f), fm.lm_head)[:, : self.nv]
        self.top.copy_(logits.float().argmax(dim=-1))

    @torch.no_grad()
    def _capture_step_graph(self):
        warm = [(self.NG + 3 - i) * max(self.D, 4) for i in range(self.NG + 1)]
        self.hdr.copy_(torch.tensor(warm, dtype=torch.int64, device=self.dev))
        self.ids_blk.fill_(self.mask_id)
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

        if self.bg == "sync" and self.K < self.T:
            w_lo = warm[0]
            w_hi = w_lo + max(1, min(self.D, 2))
            self.sync_hdr.copy_(torch.tensor([w_hi, w_lo], dtype=torch.int64, device=self.dev))
            s2 = torch.cuda.Stream()
            s2.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s2):
                for _ in range(2):
                    self._sync_backfill_device()
            torch.cuda.current_stream().wait_stream(s2)
            torch.cuda.synchronize()
            gs = torch.cuda.CUDAGraph()
            pool_s = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(gs, pool=pool_s):
                self._sync_backfill_device()
            torch.cuda.synchronize()
            self.sync_graph = gs
            self._sync_graph_pool = pool_s

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
                # Exact fp32/fp64 CPU causal attention
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
        logits = F.linear(fm.ops.rms(x[-1:], fm.ln_f), fm.lm_head)[0, : self.nv].float()
        return int(logits.argmax().item())

    def _sync_backfill_device(self):
        """CUDA-graph-safe static D-row Core 1..T + Coda backfill for slice `lo .. hi-1` (1 <= hi - lo <= D)."""
        fm = self.fm
        D, SL = self.D, self.SL
        hi = self.sync_hdr[0]
        lo = self.sync_hdr[1]
        posg_raw = hi - D + self.ar
        real = self.ar >= D - (hi - lo)
        posg = posg_raw.clamp(min=0)
        pt_valid = self.col < hi
        sl_val = hi.view(1).to(torch.int32)
        dum = self.dummy_base + self.ar

        e = self.e_store.index_select(0, posg)
        s = self.noise.index_select(0, posg)
        m_idx = 0
        for k_step in range(1, self.T + 1):
            x = F.linear(torch.cat([s, e], dim=-1), fm.adap_w, fm.adap_b)
            for j in range(fm.Lc_core):
                ms = self.meta_sync[m_idx]
                m_idx += 1
                off = (fm.P + fm.Lc_core * (k_step - 1) + j) * SL
                ms.wslot.copy_(torch.where(real, off + posg, dum))
                ms.seqlens.copy_(sl_val)
                ms.page_table.copy_(torch.where(pt_valid, off + self.col, 0).to(torch.int32))
                x = fm.run_layer(fm.core_layers[j], x, posg, ms, self.K_pool, self.V_pool, self.attn_fn)
            s = x

        h_dst = torch.where(real, posg, self.hs_dummy_base + self.ar)
        self.Hs.index_copy_(0, h_dst, s)
        x = fm.ops.rms(s, fm.ln_f)
        for c_i in range(fm.C):
            ms = self.meta_sync[m_idx]
            m_idx += 1
            off = (self.cslot + c_i) * SL
            ms.wslot.copy_(torch.where(real, off + posg, dum))
            ms.seqlens.copy_(sl_val)
            ms.page_table.copy_(torch.where(pt_valid, off + self.col, 0).to(torch.int32))
            x = fm.run_layer(fm.coda_layers[c_i], x, posg, ms, self.K_pool, self.V_pool, self.attn_fn)

    @torch.no_grad()
    def _sync_backfill(self, lo: int, hi: int):
        """Synchronous full-depth Core 1..T + Coda backfill for newly committed slice `lo .. hi-1`."""
        if hi <= lo:
            return
        if self.sync_graph is not None:
            self.h_sync_hdr[0] = hi
            self.h_sync_hdr[1] = lo
            self.sync_hdr.copy_(self.h_sync_hdr, non_blocking=True)
            self.sync_graph.replay()
            return
        self.sync_hdr[0] = hi
        self.sync_hdr[1] = lo
        self._sync_backfill_device()

    def next_props(self, top: list, c: int) -> list:
        if self.B == 0:
            return []
        off = self.set_off[c]
        return [int(top[off + j]) for j in range(self.B)]

    @torch.no_grad()
    def step_block(self, hist: list, ids_blk: list) -> list:
        if self.graph is not None:
            for i, h in enumerate(hist):
                self.h_hdr[i] = h
            self.h_ids.copy_(torch.as_tensor(ids_blk, dtype=torch.int64))
            self.hdr.copy_(self.h_hdr, non_blocking=True)
            self.ids_blk.copy_(self.h_ids, non_blocking=True)
            self.graph.replay()
            self.h_top.copy_(self.top, non_blocking=True)
            self.ev.record()
            self.ev.synchronize()
            return self.h_top.tolist()

        self.hdr.copy_(torch.tensor(hist, dtype=torch.int64, device=self.dev))
        self.ids_blk.copy_(torch.tensor(ids_blk, dtype=torch.int64, device=self.dev))
        self._step_device()
        return self.top.tolist()

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
        tail_masks = [self.mask_id] * (self.NB - self.D)
        # Filter special-token strings already caught by `self.stop_ids` (and stripped by skip_special_tokens=True)
        # so GSM8K / IFEval / MBPP run zero CPU detokenization calls inside the timed GPU loop.
        text_until = [u for u in (until or ()) if u and u not in ("<|end_turn|>", "<|end_text|>")]
        tail_win = max(32, max((len(u) for u in text_until), default=0) + 16)

        def _dec_txt(ids_slice):
            try:
                return tok.decode(ids_slice, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            except TypeError:
                return tok.decode(ids_slice, skip_special_tokens=True)

        if cuda:
            torch.cuda.synchronize()
        t_dec0 = time.perf_counter()

        while not done and (len(known) - P0) < max_new:
            if prop_override_fn is not None:
                ov = prop_override_fn(len(commits), lc, known)
                if ov is not None:
                    props = list(ov)

            filler = [self.mask_id] * (self.B - len(props))
            ids_blk = [known[-1]] + props + filler + tail_masks
            top = self.step_block(hist, ids_blk)

            np_ = len(props)
            c = 0
            while c < np_ and top[c] == props[c]:
                c += 1
            if np_ > 0:
                n_prop_steps += 1
                n_acc_total += c

            commit = props[:c] + [int(top[c])]
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
        dec_tokens = sum(commits)
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


def verify_gpu_graph_vs_eager(fm: HuginnFastModel, docs: list, seed: int, mask_id: int, cfgs: list, L_max: int):
    """Online GPU exactness gate: verify CUDA-graphed FA3 matches un-graphed torch reference."""
    dev, dt = fm.dev, fm.dt
    probe_ids = docs[0]["ids"][:96]
    for cfg in cfgs:
        pc = parse_fast_cfg(cfg)
        if pc["bg"] == "sync":
            continue
        ns = make_noise(fm.m, L_max, doc_seed(seed, 777777), dev, dt)
        d_ref = HuginnFastDecoder(
            fm, K=pc["K"], B=pc["B"], L_max=L_max, T=pc["T"], bg=pc["bg"], D_bg=pc["D_bg"],
            nv=NV_BASE, mask_id=mask_id, attn="torch", graphs=False,
        )
        r_ref = d_ref.decode_one(probe_ids, ns, max_new=32, tok=None, until=())
        del d_ref
        gc.collect()
        torch.cuda.empty_cache()

        d_fast = HuginnFastDecoder(
            fm, K=pc["K"], B=pc["B"], L_max=L_max, T=pc["T"], bg=pc["bg"], D_bg=pc["D_bg"],
            nv=NV_BASE, mask_id=mask_id, attn="paged", graphs=True,
        )
        r_fast = d_fast.decode_one(probe_ids, ns, max_new=32, tok=None, until=())
        diffs = sum(1 for a, b in zip(r_ref["gen_ids"], r_fast["gen_ids"]) if a != b)
        print(
            f"[gpu-gate] {cfg}: ref_len={len(r_ref['gen_ids'])} fast_len={len(r_fast['gen_ids'])} "
            f"token_diffs={diffs} fast_tok/s={r_fast['decode_tok_per_s']:.1f}",
            flush=True,
        )
        del d_fast
        gc.collect()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--gold", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-docs", type=int, default=0)
    ap.add_argument("--doc-lo", type=int, default=0)
    ap.add_argument("--doc-hi", type=int, default=0)
    ap.add_argument("--shard-tag", default="")
    ap.add_argument("--cfgs", default="blk:32:0,blk:32:8,blk:16:8,blk:12:8,blk:8:8")
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
    assert len(raw) > 0, "Empty document slice"

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
    print(f"[hg-fast] loaded {args.tag} directly (vocab={m.config.padded_vocab_size}, mask_id={mask_id}, HD_kv={fm.HD_kv})", flush=True)

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
    same_gpu_r32_tps = None
    kept_graphs = []  # keep CUDAGraph pools alive until process exit (PyTorch 2.11 allocator safety)

    for cfg in cfg_list:
        pc = parse_fast_cfg(cfg)
        defer_marker = os.path.join(args.out_dir, f".defer_{cfg.replace(':', '_')}")
        if os.path.exists(defer_marker):
            print(f"[defer] {cfg}{sfx} deferred via {defer_marker}", flush=True)
            continue
        fn = os.path.join(args.out_dir, f"{cfg.replace(':', '_')}{sfx}.jsonl")
        if os.path.exists(fn) and len(read_jsonl(fn)) >= len(docs):
            recs_prev = read_jsonl(fn)[: len(docs)]
            if cfg == "blk:32:0" and same_gpu_r32_tps is None:
                tot_tok = sum(int(r["dec_tokens"]) for r in recs_prev)
                tot_sec = sum(float(r["decode_s"]) for r in recs_prev)
                same_gpu_r32_tps = tot_tok / max(1e-9, tot_sec)
            print(f"[skip] {cfg}{sfx} already complete ({len(recs_prev)} docs)", flush=True)
            continue

        use_graphs = bool(args.graphs)
        r_eager_gate = None
        if args.verify_gate and use_graphs:
            d_eag = HuginnFastDecoder(
                fm, K=pc["K"], B=pc["B"], L_max=L_max, T=pc["T"], bg=pc["bg"], D_bg=pc["D_bg"],
                nv=NV_BASE, mask_id=mask_id, attn=args.attn, graphs=False,
            )
            g_ns = make_noise(m, L_max, doc_seed(args.seed, 777777), fm.dev, fm.dt)
            r_eager_gate = d_eag.decode_one(docs[0]["ids"][:96], g_ns, max_new=24, tok=None, until=())
            del d_eag
            gc.collect()
            torch.cuda.empty_cache()

        dec = HuginnFastDecoder(
            fm,
            K=pc["K"],
            B=pc["B"],
            L_max=L_max,
            T=pc["T"],
            bg=pc["bg"],
            D_bg=pc["D_bg"],
            nv=NV_BASE,
            mask_id=mask_id,
            attn=args.attn,
            graphs=use_graphs,
        )
        if dec.graph is not None:
            kept_graphs.append((dec.graph, dec._graph_pool))
        if dec.sync_graph is not None:
            kept_graphs.append((dec.sync_graph, dec._sync_graph_pool))

        if r_eager_gate is not None:
            g_ns = make_noise(m, L_max, doc_seed(args.seed, 777777), fm.dev, fm.dt)
            r_gr = dec.decode_one(docs[0]["ids"][:96], g_ns, max_new=24, tok=None, until=())
            diffs = sum(1 for a, b in zip(r_eager_gate["gen_ids"], r_gr["gen_ids"]) if a != b)
            print(
                f"[gpu-gate] {cfg}: len={len(r_gr['gen_ids'])} graph_vs_eager_diffs={diffs} "
                f"tok/s={r_gr['decode_tok_per_s']:.1f}",
                flush=True,
            )
            assert diffs == 0, f"CUDA graph vs eager FA3 mismatch on {cfg}: {diffs} diffs"
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
            if same_gpu_r32_tps is not None:
                r["same_gpu_r32_tok_per_s"] = same_gpu_r32_tps
            recs.append(r)
            if (idx + 1) % 50 == 0 or (idx + 1) == len(docs):
                tot_tok = sum(int(x["dec_tokens"]) for x in recs)
                tot_sec = sum(float(x["decode_s"]) for x in recs)
                tps = tot_tok / max(1e-9, tot_sec)
                mb = tot_tok / max(1, sum(int(x["steps"]) for x in recs))
                print(
                    f"[{args.tag}{sfx} {cfg}] {idx + 1}/{len(docs)} ({time.time() - t0:.1f}s) | "
                    f"1+E[c]={mb:.2f} | tok/s={tps:.2f}",
                    flush=True,
                )

        if cfg == "blk:32:0" and same_gpu_r32_tps is None:
            tot_tok = sum(int(x["dec_tokens"]) for x in recs)
            tot_sec = sum(float(x["decode_s"]) for x in recs)
            same_gpu_r32_tps = tot_tok / max(1e-9, tot_sec)
            for r in recs:
                r["same_gpu_r32_tok_per_s"] = same_gpu_r32_tps

        tmp_fn = f"{fn}.{os.getpid()}.tmp"
        with open(tmp_fn, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp_fn, fn)
        del dec
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
