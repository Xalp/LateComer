"""core_ops.py -- Ouro-2.6B numerics for the late lattice inside SGLang.

COPIED (numerics unchanged) from decode/engine.py (Cfg, Ops, attn_torch, attn_paged, Meta,
OuroEngine.loop_pass / gate / _prefill_body) and late.py (LateEngine.late_pass). The only change is the KV addressing: every K/V write and
every page-table entry is an ABSOLUTE index into one flat per-layer buffer [T*STRIDE, 1, NH, HD] (loop t of base slot s = t*STRIDE + s,
kv_pool.LoopStridedMHAPool), where engine.py used region-relative indices into per-loop views. The kernels (cuBLAS F.linear on fused
qkv / gate_up, SGLang RMSNorm, sgl_kernel silu_and_mul, flashinfer RoPE with the fp32 cos/sin table, sgl_kernel FA3
flash_attn_with_kvcache page_size 1 causal bottom-right) and their order are the same, so a row computes the same bits as late.py's row.
"""
import json, os
import torch
import torch.nn.functional as F


class Cfg:
    """copy of engine.Cfg"""
    def __init__(self, d):
        self.H = d["hidden_size"]; self.NH = d["num_attention_heads"]; self.HD = d.get("head_dim") or self.H // self.NH
        assert d.get("num_key_value_heads", self.NH) == self.NH, "MHA only (Ouro)"
        self.INTER = d["intermediate_size"]; self.NL = d["num_hidden_layers"]; self.V = d["vocab_size"]
        self.T = d.get("total_ut_steps", 4); self.eps = d.get("rms_norm_eps", 1e-6); self.theta = d.get("rope_theta", 1e6)
        assert not d.get("rope_scaling"), "default RoPE only"
        assert all(t == "full_attention" for t in d.get("layer_types", ["full_attention"])), "full attention only"


def load_weights(model_dir, device, dtype=torch.bfloat16):
    """copy of engine.load_weights: safetensors -> {name: tensor}"""
    from safetensors import safe_open
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
    W = {}
    for fn in files:
        with safe_open(os.path.join(model_dir, fn), framework="pt", device="cpu") as f:
            for k in f.keys(): W[k] = f.get_tensor(k).to(device=device, dtype=dtype)
    return W, Cfg(json.load(open(os.path.join(model_dir, "config.json"))))


class Ops:
    """copy of engine.Ops (torch ops = HF numerics; fused ops each verified against the torch op before use)"""

    def __init__(self, cfg, device, dtype, fused=True, log=print):
        self.c, self.dev, self.dt, self.log = cfg, device, dtype, log
        self.chosen = {"rms": "torch", "add_rms": "torch", "silu": "torch", "rope": "torch"}
        HD = cfg.HD
        inv = 1.0 / (cfg.theta ** (torch.arange(0, HD, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / HD))
        self.inv_freq = inv
        self._tab_len = 0
        self.ensure_rope(8192)
        self.rms, self.add_rms, self.silu_mul, self.rope = self.t_rms, self.t_add_rms, self.t_silu_mul, self.t_rope
        self._norm_mods = {}
        if fused and str(device).startswith("cuda"):
            self._pick_fused()

    def ensure_rope(self, n):
        if n <= self._tab_len: return
        # (review 10-28) captured executor graphs hold the old table's address: growing it after capture would leave them reading freed memory
        assert not getattr(self, "frozen", False), ("RoPE table cannot grow after the executor graphs were captured", n, self._tab_len)
        pos =torch.arange(n, device=self.dev, dtype=torch.float32)
        freqs = pos[:, None] * self.inv_freq[None, :]
        emb = torch.cat([freqs, freqs], -1)
        self.cos, self.sin = emb.cos().to(self.dt), emb.sin().to(self.dt)
        self.cos_sin32 = torch.cat([freqs.cos(), freqs.sin()], -1).contiguous()
        self.cos_sin_bf = torch.cat([self.cos[:, : self.c.HD // 2], self.sin[:, : self.c.HD // 2]], -1).contiguous()
        self._tab_len = n
        if self.chosen.get("rope", "torch") != "torch":      # re-bind the fused rope to the new table
            self._bind_rope()

    def t_rms(self, x, w):
        xf = x.to(torch.float32); var = xf.pow(2).mean(-1, keepdim=True)
        return w * (xf * torch.rsqrt(var + self.c.eps)).to(x.dtype)

    def t_add_rms(self, x, res, w):
        res = res + x
        return self.t_rms(res, w), res

    def t_silu_mul(self, gu):
        g, u = gu[:, : self.c.INTER], gu[:, self.c.INTER :]
        return F.silu(g) * u

    def t_rope(self, q, k, pos):
        N = q.shape[0]; NH, HD = self.c.NH, self.c.HD
        cos, sin = self.cos[pos][:, None, :], self.sin[pos][:, None, :]
        def rot(x):
            x = x.reshape(N, NH, HD); x1, x2 = x[..., : HD // 2], x[..., HD // 2 :]
            return (x * cos + torch.cat((-x2, x1), -1) * sin).reshape(N, NH * HD)
        return rot(q), rot(k)

    def _bind_rope(self):
        fn, tab_name = self._rope_fn, self._rope_tab
        tab = getattr(self, tab_name); hd = self.c.HD
        def rope_k(q, k, pos, fn=fn, tab=tab):
            fn(positions=pos, query=q, key=k, head_size=hd, cos_sin_cache=tab, is_neox=True); return q, k
        self.rope = rope_k

    def _pick_fused(self):
        c, dev, dt = self.c, self.dev, self.dt
        H = c.H
        xt = torch.randn(7, H, device=dev, dtype=dt); wt = torch.rand(H, device=dev, dtype=dt) + 0.5
        rt = torch.randn(7, H, device=dev, dtype=dt)
        try:
            from sglang.srt.layers.layernorm import RMSNorm as SglRMS
            def mk(w):
                key = w.data_ptr()
                if key not in self._norm_mods:
                    m = SglRMS(H, eps=c.eps).to(dev, dt); m.weight = torch.nn.Parameter(w, requires_grad=False); self._norm_mods[key] = m
                return self._norm_mods[key]
            def s_rms(x, w): return mk(w)(x)
            def s_add_rms(x, res, w):
                out = mk(w)(x, res)
                return out if isinstance(out, tuple) else (out, res)
            y = s_rms(xt, wt); torch.cuda.synchronize()
            e1 = (y.float() - self.t_rms(xt, wt).float()).abs().max().item()
            xx, rr = xt.clone(), rt.clone(); got, gres = s_add_rms(xx, rr, wt); torch.cuda.synchronize()
            en, er = self.t_add_rms(xt, rt, wt)
            e2 = max((got.float() - en.float()).abs().max().item(), (gres.float() - er.float()).abs().max().item())
            self.log(f"[ops] sglang RMSNorm err rms {e1:.4f} add_rms {e2:.4f}")
            if e1 < 0.05: self.rms = s_rms; self.chosen["rms"] = "sglang.RMSNorm"
            if e2 < 0.08: self.add_rms = s_add_rms; self.chosen["add_rms"] = "sglang.RMSNorm(fused add)"
        except Exception as e:
            self.log(f"[ops] sglang RMSNorm unavailable: {e!r}"[:300])
        for mod, name in (("sglang.jit_kernel.activation", "silu_and_mul"), ("sgl_kernel", "silu_and_mul"), ("flashinfer.activation", "silu_and_mul")):
            try:
                fn = getattr(__import__(mod, fromlist=[name]), name)
                gu = torch.randn(5, 2 * c.INTER, device=dev, dtype=dt); o = torch.empty(5, c.INTER, device=dev, dtype=dt)
                fn(gu, o); torch.cuda.synchronize()
                err = (o.float() - self.t_silu_mul(gu).float()).abs().max().item()
                if err < 0.05:
                    def silu_k(gu, fn=fn):
                        out = torch.empty(gu.shape[0], c.INTER, device=gu.device, dtype=gu.dtype); fn(gu, out); return out
                    self.silu_mul = silu_k; self.chosen["silu"] = f"{mod}.{name}"; break
            except Exception as e:
                self.log(f"[ops] {mod}.{name} failed: {e!r}"[:300])
        N = 9; pos = torch.tensor([0, 1, 5, 17, 300, 301, 1023, 2047, 4000], device=dev)
        for mod, name in (("sgl_kernel", "apply_rope_with_cos_sin_cache_inplace"), ("flashinfer.rope", "apply_rope_with_cos_sin_cache_inplace")):
            for tab_name in ("cos_sin32", "cos_sin_bf"):
                try:
                    fn = getattr(__import__(mod, fromlist=[name]), name)
                    tab = getattr(self, tab_name)
                    qkv = torch.randn(N, 3 * H, device=dev, dtype=dt)
                    rq, rk = self.t_rope(qkv[:, :H], qkv[:, H : 2 * H], pos)
                    q, k = qkv[:, :H], qkv[:, H : 2 * H]
                    fn(positions=pos, query=q, key=k, head_size=c.HD, cos_sin_cache=tab, is_neox=True); torch.cuda.synchronize()
                    err = max((q.float() - rq.float()).abs().max().item(), (k.float() - rk.float()).abs().max().item())
                    self.log(f"[ops] rope {mod} table {tab_name} err {err:.4f}")
                    if err < 0.06:
                        self._rope_fn, self._rope_tab = fn, tab_name; self._bind_rope()
                        self.chosen["rope"] = f"{mod}[{tab_name}]"; break
                except Exception as e:
                    self.log(f"[ops] rope {mod} {tab_name} failed: {e!r}"[:300])
            if self.chosen["rope"] != "torch": break
        self.log(f"[ops] chosen {self.chosen}")


class Meta:
    """per-forward attention metadata: pos [N], wslot [N] (absolute flat K/V index), page_table [nseq, W] int32 (absolute),
    seqlens [nseq] int32, cu_q [nseq+1] int32, max_q (python int)"""
    pass


_FA = None
def get_fa3():
    global _FA
    if _FA is None:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        _FA = flash_attn_with_kvcache
    return _FA


def attn_torch(q, kc, vc, meta, NH, HD):
    """copy of engine.attn_torch (reference; CPU tests). One change: a query row with no visible key returns 0 (as FA3 does for
    bottom-right causal rows with seqlen_q > seqlen_k) instead of NaN -- only short-prompt pad rows hit it."""
    N = q.shape[0]; acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    out = torch.zeros(N, NH, HD, dtype=q.dtype, device=q.device)
    pt = meta.page_table.cpu(); sl = meta.seqlens.cpu().tolist(); cu = meta.cu_q.cpu().tolist()
    kf, vf = kc.view(-1, NH, HD), vc.view(-1, NH, HD); scale = HD ** -0.5
    for s in range(len(sl)):
        q0, q1, Lk = cu[s], cu[s + 1], sl[s]
        if q1 == q0: continue
        idx = pt[s, :Lk].long().to(q.device)
        K, V = kf[idx].to(acc), vf[idx].to(acc); Q = q[q0:q1].view(-1, NH, HD).to(acc)
        sc = torch.einsum("qhd,khd->hqk", Q, K) * scale
        Lq = q1 - q0; ii = torch.arange(Lq, device=q.device)[:, None]; jj = torch.arange(Lk, device=q.device)[None]
        vis = (jj <= Lk - Lq + ii)
        sc = sc.masked_fill(~vis[None], float("-inf"))
        p = torch.softmax(sc, -1)
        p = torch.where(vis.any(-1)[None, :, None], p, torch.zeros_like(p))
        out[q0:q1] = torch.einsum("hqk,khd->qhd", p, V).to(q.dtype)
    return out.view(N, NH * HD)


def attn_paged(q, kc, vc, meta, NH, HD):
    """copy of engine.attn_paged"""
    FA = get_fa3(); N = q.shape[0]
    o = FA(q.contiguous().view(N, NH, HD), kc, vc, cache_seqlens=meta.seqlens, page_table=meta.page_table, cu_seqlens_q=meta.cu_q,
           max_seqlen_q=meta.max_q, causal=True)
    o = o[0] if isinstance(o, (tuple, list)) else o
    return o.reshape(N, NH * HD)


class OuroCore:
    """weights + ops of Ouro (fused qkv / gate_up), as engine.OuroEngine minus its own caches. `W` maps HF names to tensors (as
    engine.load_weights / weights_from_hf), or pass `layers`/... directly (the SGLang model passes its parameters, no copy)."""

    def __init__(self, cfg, device, dtype, W=None, params=None, fused=True, log=print):
        self.c, self.dev, self.dt, self.log = cfg, device, dtype, log
        c = cfg
        if params is not None:
            self.layers, self.embed, self.norm, self.gw, self.gb, self.lm_head = (params[k] for k in ("layers", "embed", "norm", "gw", "gb", "lm_head"))
        else:
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
        self.ops = Ops(c, device, dtype, fused=fused, log=log)

    # ---- one 48-layer pass (late.LateEngine.late_pass == engine.OuroEngine.loop_pass with absolute addressing) ----
    def pass_rows(self, X, meta, kv, attn="paged"):
        """X [N, H]; kv.kflat[li] / kv.vflat[li] = [T*STRIDE, 1, NH, HD] flat buffers; K/V of every row written at meta.wslot BEFORE the
        layer's attention; returns the normed state (next pass input)"""
        c, ops = self.c, self.ops; H, NH, HD = c.H, c.NH, c.HD
        res = X.clone()
        h = ops.rms(res, self.layers[0]["n1"])
        for li, Lw in enumerate(self.layers):
            qkv = F.linear(h, Lw["qkv"])
            q, k, v = qkv[:, :H], qkv[:, H : 2 * H], qkv[:, 2 * H :]
            q, k = ops.rope(q, k, meta.pos)
            kb, vb = kv.kflat[li], kv.vflat[li]
            kb.view(-1, H).index_copy_(0, meta.wslot, k); vb.view(-1, H).index_copy_(0, meta.wslot, v)
            o = attn_paged(q, kb, vb, meta, NH, HD) if attn == "paged" else attn_torch(q, kb, vb, meta, NH, HD)
            ao = ops.rms(F.linear(o, Lw["o"]), Lw["n2"])
            h, res = ops.add_rms(ao, res, Lw["n3"])
            m = ops.rms(F.linear(ops.silu_mul(F.linear(h, Lw["gu"])), Lw["down"]), Lw["n4"])
            if li + 1 < c.NL: h, res = ops.add_rms(m, res, self.layers[li + 1]["n1"])
            else: res = res + m
        return ops.rms(res, self.norm)

    def gate(self, hn):
        return torch.sigmoid(F.linear(hn, self.gw, self.gb))[:, 0]

    # ---- full-depth T-pass forward of rows (prefill / chunked extend / plain decode), engine._prefill_body numerics ----
    def full_depth(self, ids, metas, kv, sel, nv, attn="paged"):
        """ids [N]; metas[t] = Meta of pass t (wslot / page_table in region t); sel [n] rows to read out.
        Returns exp [n, nv] fp32 = Ouro's gate mixture over the T passes (as engine.OuroEngine._prefill_body)."""
        X = F.embedding(ids, self.embed)
        rem = torch.ones(sel.shape[0], device=X.device); exp = None
        T = self.c.T
        for t in range(T):
            X = self.pass_rows(X, metas[t], kv, attn)
            hn = X.index_select(0, sel); lam = self.gate(hn)
            p = lam * rem if t < T - 1 else rem; rem = rem * (1 - lam)
            lg = F.linear(hn, self.lm_head)[:, :nv].float()
            exp = lg * p[:, None] if exp is None else exp + lg * p[:, None]
        return exp


class FlatKV:
    """flat per-layer K/V buffers [T*STRIDE, 1, NH, HD] (loop t of base slot s at t*STRIDE + s); region t's placeholder scratch starts at
    t*STRIDE + SCR0. Built from a LoopStridedMHAPool (server) or allocated directly (tests / harness)."""

    def __init__(self, kflat, vflat, T, stride, scr0, scr_len):
        self.kflat, self.vflat, self.T, self.STRIDE, self.SCR0, self.SCR = kflat, vflat, T, stride, scr0, scr_len

    @classmethod
    def alloc(cls, cfg, size, scr_len, device, dtype):
        """size = number of base slots (slot 0 = dummy, 1..size-1 usable), scr_len = placeholder scratch rows per region"""
        stride = size + scr_len
        mk = lambda: [torch.zeros(cfg.T * stride, 1, cfg.NH, cfg.HD, device=device, dtype=dtype) for _ in range(cfg.NL)]
        return cls(mk(), mk(), cfg.T, stride, size, scr_len)


def extend_metas(T, kv, pos, wbase, base_rows, seqlens, cu, max_q, W):
    """Meta per pass for a full-depth forward of rows: wbase [N] base slots the rows write (0 = dummy for pad rows), base_rows [nseq, W]
    base slots of each sequence's key columns (columns >= seqlen ignored), seqlens/cu int32."""
    metas = []
    col = torch.arange(W, device=pos.device)[None]
    for t in range(T):
        m = Meta(); m.pos = pos; m.wslot = t * kv.STRIDE + wbase
        pt = torch.where(col < seqlens[:, None].long(), t * kv.STRIDE + base_rows.long(), torch.zeros_like(base_rows, dtype=torch.long))
        m.page_table = pt.to(torch.int32).contiguous(); m.seqlens = seqlens; m.cu_q = cu; m.max_q = max_q
        metas.append(m)
    return metas
