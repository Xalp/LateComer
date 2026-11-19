"""decode -- MARS-style synchronous lattice decoder for Ouro (looped LM), standalone PyTorch, batch size 1, greedy.

Semantics = `Lossless` (lossless:B:0:0) of MARS/integration/probes/lmeval_ouro/vendor_grid/ouro_stream.py, re-implemented on an
optimized substrate (fused qkv / gate_up GEMMs via cuBLAS, SGLang fused RMSNorm / silu_and_mul, sgl_kernel or flashinfer RoPE,
FA3 attention, static per-loop KV caches, one CUDA graph per (B, pass)). The AR baseline is the same machinery with B = 0
(one token slot, no placeholders): one token per 4 graph replays of a 1-row pass.

Lattice step (B proposals, D = B + 1 token slots, W = D + D*B = (B+1)^2 rows):
  rows 0..D-1          token slots: slot 0 = last committed token x_c (its KV is not in the prefix yet), slots 1..B = proposals
  rows D + a*B + i-1   placeholder i (1..B) of the set that follows token slot a (input = <|mask|> embedding)
  row positions: token slot j at Lc + j, placeholder (a, i) at Lc + a + i  (Lc = number of positions in the prefix caches)
  pass t (0..T-1) of every row reads the prefix's pass-t KV + the window rows allowed by the lattice mask (their pass-t KV):
    token slot j      : token slots 0..j
    placeholder (a,i) : token slots 0..a and placeholders (a,1..i)
  the hidden state carries across passes (normed state = next pass input, as OuroModel.forward); readout = Ouro's gate mixture over
  the T passes (fp32 accumulation, as the reference). Token slot j predicts position Lc+j+1; proposal j+1 is accepted iff it equals
  that prediction; c = #accepted; commit proposals 1..c + token slot c's prediction; the KV of token slots 0..c (written at cache
  slots Lc..Lc+c, every pass) IS their full-depth KV -> Lc += c+1, no recomputation; the next proposals = argmax of the placeholders
  that follow token slot c. First step after prefill: there are no proposals yet (the reference runs D=1 there); we run the full
  shape with dummy proposals and FORCE c=0 -- token slot 0 and placeholder set 0 never see the dummies, so the step is identical.

Attention backends (selected per decoder, all checked against `torch` in check_kernels()):
  torch  : reference (per-sequence gather through the page table + explicit causal softmax) -- CPU tests and GPU checks
  paged  : ONE FA3 call per layer, varlen over D+1 sequences [token slots | placeholder set 0 | ... | set B] with a page_size-1
           page table into the per-loop cache (prefix pages + the window slots each sequence may see), causal (bottom-right)
  split  : FA3 over the contiguous prefix for all W rows at once (non-causal, returns LSE) + a Triton kernel that attends the W
           window keys under the static lattice mask and merges with the prefix part via the LSE (prefix read once)
"""
import json, math, os, time
import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------------------------- weights
class Cfg:
    def __init__(self, d):
        self.H = d["hidden_size"]; self.NH = d["num_attention_heads"]; self.HD = d.get("head_dim") or self.H // self.NH
        assert d.get("num_key_value_heads", self.NH) == self.NH, "MHA only (Ouro)"
        self.INTER = d["intermediate_size"]; self.NL = d["num_hidden_layers"]; self.V = d["vocab_size"]
        self.T = d.get("total_ut_steps", 4); self.eps = d.get("rms_norm_eps", 1e-6); self.theta = d.get("rope_theta", 1e6)
        assert not d.get("rope_scaling"), "default RoPE only"
        assert all(t == "full_attention" for t in d.get("layer_types", ["full_attention"])), "full attention only"


def load_weights(model_dir, device, dtype=torch.bfloat16):
    """safetensors -> {name: tensor} on device (no transformers dependency)"""
    from safetensors import safe_open
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
    W = {}
    for fn in files:
        with safe_open(os.path.join(model_dir, fn), framework="pt", device="cpu") as f:
            for k in f.keys(): W[k] = f.get_tensor(k).to(device=device, dtype=dtype)
    return W, Cfg(json.load(open(os.path.join(model_dir, "config.json"))))


def weights_from_hf(model):
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    return sd, Cfg(model.config.to_dict())


# ----------------------------------------------------------------------------------------------------------------- ops
class Ops:
    """torch ops reproduce the HF modeling numerics exactly (OuroRMSNorm computes in fp32 and multiplies the weight in the input dtype,
    rotate_half RoPE with cos/sin cast to the input dtype); fused ops = SGLang / sgl_kernel / flashinfer kernels, each verified against
    the torch op before use (as engine_cost/bench_loop.py)."""

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
        pos = torch.arange(n, device=self.dev, dtype=torch.float32)
        freqs = pos[:, None] * self.inv_freq[None, :]                            # == HF (inv_freq @ position_ids, K=1)
        emb = torch.cat([freqs, freqs], -1)
        self.cos, self.sin = emb.cos().to(self.dt), emb.sin().to(self.dt)       # HF: cos/sin cast to the activation dtype
        self.cos_sin32 = torch.cat([freqs.cos(), freqs.sin()], -1).contiguous()  # [n, HD] fp32 (cos | sin), sglang layout
        self.cos_sin_bf = torch.cat([self.cos[:, : self.c.HD // 2], self.sin[:, : self.c.HD // 2]], -1).contiguous()
        self._tab_len = n

    # ---- torch (HF-exact) ----
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

    # ---- fused ----
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
        # RoPE: in-place kernels on the strided q/k views of the fused qkv output
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
                        def rope_k(q, k, pos, fn=fn, tab=tab):
                            fn(positions=pos, query=q, key=k, head_size=c.HD, cos_sin_cache=tab, is_neox=True); return q, k
                        self.rope = rope_k; self.chosen["rope"] = f"{mod}[{tab_name}]"; break
                except Exception as e:
                    self.log(f"[ops] rope {mod} {tab_name} failed: {e!r}"[:300])
            if self.chosen["rope"] != "torch": break
        self.log(f"[ops] chosen {self.chosen}")


# ----------------------------------------------------------------------------------------------------------------- attention
_FA = None
def get_fa3():
    global _FA
    if _FA is None:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        _FA = flash_attn_with_kvcache
    return _FA


_WIN_KERNEL = None
def get_win_kernel():
    """Triton kernel: window attention under a static lattice mask, merged with a prefix partial (o1, lse1) via online softmax"""
    global _WIN_KERNEL
    if _WIN_KERNEL is not None: return _WIN_KERNEL
    import triton, triton.language as tl

    @triton.jit
    def _win_merge(Q, K, V, O1, LSE, MSK, OUT, N, s_q, s_k, s_v, s_o1, s_lh, s_lr, s_out, scale,
                   HD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
        pm = tl.program_id(0); h = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM); rd = tl.arange(0, HD); mrow = rm < N
        q = tl.load(Q + rm[:, None] * s_q + h * HD + rd[None, :], mask=mrow[:, None], other=0.0)
        m_i = tl.load(LSE + h * s_lh + rm * s_lr, mask=mrow, other=0.0).to(tl.float32)       # prefix logsumexp (natural log)
        l_i = tl.full([BM], 1.0, tl.float32)                                                     # sum exp(s - m_i) over the prefix = 1
        acc = tl.load(O1 + rm[:, None] * s_o1 + h * HD + rd[None, :], mask=mrow[:, None], other=0.0).to(tl.float32)
        for n0 in range(0, N, BN):
            rn = n0 + tl.arange(0, BN); mcol = rn < N
            k = tl.load(K + rn[:, None] * s_k + h * HD + rd[None, :], mask=mcol[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k)) * scale
            vis = tl.load(MSK + rm[:, None] * N + rn[None, :], mask=mrow[:, None] & mcol[None, :], other=0)
            s = tl.where(vis != 0, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(V + rn[:, None] * s_v + h * HD + rd[None, :], mask=mcol[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        out = acc / l_i[:, None]
        tl.store(OUT + rm[:, None] * s_out + h * HD + rd[None, :], out.to(OUT.dtype.element_ty), mask=mrow[:, None])

    def run(q, k, v, o1, lse, msk, out, scale, NH, HD):
        N = q.shape[0]; BM, BN = 16, 64
        grid = (triton.cdiv(N, BM), NH)
        _win_merge[grid](q, k, v, o1, lse, msk, out, N, q.stride(0), k.stride(0), v.stride(0), o1.stride(0), lse.stride(0), lse.stride(1),
                         out.stride(0), scale, HD=HD, BM=BM, BN=BN, num_warps=4)
        return out
    _WIN_KERNEL = run
    return run


class Meta:
    """per-forward attention metadata (static buffers for CUDA graphs)"""
    pass


def attn_torch(q, kc, vc, meta, NH, HD):
    """reference: q [N, NH*HD] (roped), kc/vc [S, 1, NH, HD] already holding this forward's window K/V; page_table semantics"""
    N = q.shape[0]; acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    out = torch.empty(N, NH, HD, dtype=q.dtype, device=q.device)
    pt = meta.page_table.cpu(); sl = meta.seqlens.cpu().tolist(); cu = meta.cu_q.cpu().tolist()
    kf, vf = kc.view(-1, NH, HD), vc.view(-1, NH, HD); scale = HD ** -0.5
    for s in range(len(sl)):
        q0, q1, Lk = cu[s], cu[s + 1], sl[s]
        if q1 == q0: continue
        idx = pt[s, :Lk].long().to(q.device)
        K, V = kf[idx].to(acc), vf[idx].to(acc); Q = q[q0:q1].view(-1, NH, HD).to(acc)
        sc = torch.einsum("qhd,khd->hqk", Q, K) * scale
        Lq = q1 - q0; ii = torch.arange(Lq, device=q.device)[:, None]; jj = torch.arange(Lk, device=q.device)[None]
        sc = sc.masked_fill(~(jj <= Lk - Lq + ii)[None], float("-inf"))
        p = torch.softmax(sc, -1)
        out[q0:q1] = torch.einsum("hqk,khd->qhd", p, V).to(q.dtype)
    return out.view(N, NH * HD)


def attn_paged(q, kc, vc, meta, NH, HD):
    FA = get_fa3(); N = q.shape[0]
    o = FA(q.contiguous().view(N, NH, HD), kc, vc, cache_seqlens=meta.seqlens, page_table=meta.page_table, cu_seqlens_q=meta.cu_q,
           max_seqlen_q=meta.max_q, causal=True)
    o = o[0] if isinstance(o, (tuple, list)) else o
    return o.reshape(N, NH * HD)


def attn_split(q, k, v, kc, vc, meta, NH, HD):
    """prefix part: FA3 over cache slots [0, Lc) for all rows (non-causal, LSE); window part + merge: Triton (lattice mask)"""
    FA = get_fa3(); N = q.shape[0]; S = kc.shape[0]
    r = FA(q.contiguous().view(1, N, NH, HD), kc.view(1, S, NH, HD), vc.view(1, S, NH, HD), cache_seqlens=meta.lc32, causal=False,
           return_softmax_lse=True)
    o1, lse = r[0], r[1]
    o1 = o1.view(N, NH, HD); lse = lse.view(NH, N) if lse.dim() == 3 else lse      # [1, NH, N] -> [NH, N]
    out = torch.empty(N, NH, HD, device=q.device, dtype=q.dtype)
    get_win_kernel()(q, k, v, o1.view(N, NH * HD), lse, meta.winmask, out.view(N, NH * HD), HD ** -0.5, NH, HD)
    return out.view(N, NH * HD)


# ----------------------------------------------------------------------------------------------------------------- engine
class OuroEngine:
    def __init__(self, W, cfg, device, dtype=torch.bfloat16, S=2560, fused=True, log=print):
        self.c, self.dev, self.dt, self.S, self.log = cfg, device, dtype, S, log
        c = cfg; H = c.H
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
        self.ops.ensure_rope(S + 64)
        self.prefill_graphs = False; self.pg_bucket = 64; self.pgraphs = {}
        # per-loop KV caches, page_size 1 layout [S, 1, NH, HD]
        self.kc = [[torch.zeros(S, 1, c.NH, c.HD, device=device, dtype=dtype) for _ in range(c.NL)] for _ in range(c.T)]
        self.vc = [[torch.zeros(S, 1, c.NH, c.HD, device=device, dtype=dtype) for _ in range(c.NL)] for _ in range(c.T)]

    def loop_pass(self, X, t, meta, attn):
        """one 48-layer pass of rows X [N, H] reading/writing the pass-t caches; returns the normed state (next pass input)"""
        c, ops = self.c, self.ops; H, NH, HD = c.H, c.NH, c.HD; N = X.shape[0]
        res = X.clone()
        h = ops.rms(res, self.layers[0]["n1"])
        for li, Lw in enumerate(self.layers):
            qkv = F.linear(h, Lw["qkv"])
            q, k, v = qkv[:, :H], qkv[:, H : 2 * H], qkv[:, 2 * H :]
            q, k = ops.rope(q, k, meta.pos)
            kc, vc = self.kc[t][li], self.vc[t][li]
            kc.view(self.S, H).index_copy_(0, meta.slot, k); vc.view(self.S, H).index_copy_(0, meta.slot, v)
            if attn == "paged": o = attn_paged(q, kc, vc, meta, NH, HD)
            elif attn == "split": o = attn_split(q, k, v, kc, vc, meta, NH, HD)
            else: o = attn_torch(q, kc, vc, meta, NH, HD)
            ao = ops.rms(F.linear(o, Lw["o"]), Lw["n2"])
            h, res = ops.add_rms(ao, res, Lw["n3"])
            m = ops.rms(F.linear(ops.silu_mul(F.linear(h, Lw["gu"])), Lw["down"]), Lw["n4"])
            if li + 1 < c.NL: h, res = ops.add_rms(m, res, self.layers[li + 1]["n1"])
            else: res = res + m
        return ops.rms(res, self.norm)

    def gate(self, hn):
        return torch.sigmoid(F.linear(hn, self.gw, self.gb))[:, 0]

    def _prefill_body(self, ids_t, last, meta, nv, attn):
        X = F.embedding(ids_t, self.embed)
        rem = torch.ones(1, device=self.dev); exp = None
        for t in range(self.c.T):
            X = self.loop_pass(X, t, meta, attn)
            hn = X.index_select(0, last); lam = self.gate(hn)
            p = lam * rem if t < self.c.T - 1 else rem; rem = rem * (1 - lam)
            lg = F.linear(hn, self.lm_head)[:, :nv].float()
            exp = lg * p[:, None] if exp is None else exp + lg * p[:, None]
        return exp[0].topk(2)

    def _pgraph(self, Pb, nv):
        """CUDA graph of the whole T-pass prefill for prompts padded to Pb rows (pad rows sit AFTER the prompt: causal, so they never
        influence it; their K/V land in slots >= P, which every later forward overwrites before anything can read them)"""
        key = (Pb, nv)
        if key in self.pgraphs: return self.pgraphs[key]
        assert Pb + 8 <= self.S, (Pb, self.S)
        dev = self.dev; g = Meta()
        g.ids = torch.zeros(Pb, dtype=torch.int64, device=dev); g.last = torch.zeros(1, dtype=torch.int64, device=dev)
        g.v = torch.zeros(2, device=dev); g.i = torch.zeros(2, dtype=torch.int64, device=dev)
        m = Meta(); m.pos = torch.arange(Pb, device=dev); m.slot = m.pos
        m.page_table = torch.arange(Pb, device=dev, dtype=torch.int32)[None].contiguous(); m.seqlens = torch.tensor([Pb], device=dev, dtype=torch.int32)
        m.cu_q = torch.tensor([0, Pb], device=dev, dtype=torch.int32); m.max_q = Pb; g.meta = m
        def body():
            v = self._prefill_body(g.ids, g.last, m, nv, "paged"); g.v.copy_(v.values); g.i.copy_(v.indices)
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            body(); body()
        torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
        g.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g.graph, pool=torch.cuda.graph_pool_handle()): body()
        torch.cuda.synchronize()
        g.h_ids = torch.zeros(Pb, dtype=torch.int64).pin_memory(); g.h_last = torch.zeros(1, dtype=torch.int64).pin_memory()
        self.pgraphs[key] = g
        return g

    @torch.no_grad()
    def prefill(self, ids, nv, attn="auto"):
        """normal T-pass forward of the prompt (causal), fills the T per-loop caches at slots 0..P-1; returns the prompt's next-token
        prediction (gate-mixture readout of the last position, as ouro_stream.Stream.prefill) and its top-2 margin.
        With self.prefill_graphs on (CUDA, paged FA3 available): one CUDA graph per padded length bucket (multiple of self.pg_bucket)."""
        P = len(ids); assert P + 8 <= self.S, (P, self.S)
        dev = self.dev
        if self.prefill_graphs and attn != "torch" and str(dev).startswith("cuda"):
            Pb = -(-P // self.pg_bucket) * self.pg_bucket; g = self._pgraph(Pb, nv)
            g.h_ids.zero_(); g.h_ids[:P] = torch.as_tensor(ids, dtype=torch.int64); g.h_last[0] = P - 1
            g.ids.copy_(g.h_ids, non_blocking=True); g.last.copy_(g.h_last, non_blocking=True)
            g.graph.replay(); v = g.v.tolist(); i = g.i.tolist()
            return int(i[0]), float(v[0] - v[1])
        meta = Meta()
        meta.pos = torch.arange(P, device=dev); meta.slot = meta.pos
        meta.page_table = torch.arange(P, device=dev, dtype=torch.int32)[None]; meta.seqlens = torch.tensor([P], device=dev, dtype=torch.int32)
        meta.cu_q = torch.tensor([0, P], device=dev, dtype=torch.int32); meta.max_q = P
        v2 = self._prefill_body(torch.tensor(ids, device=dev), torch.tensor([P - 1], device=dev), meta, nv, "torch" if attn == "torch" else "paged")
        return int(v2.indices[0]), float(v2.values[0] - v2.values[1])


# ----------------------------------------------------------------------------------------------------------------- lattice decoder
class Lattice:
    """B >= 1: synchronous MARS lattice; B = 0: plain full-depth AR (one token slot). See module docstring."""

    def __init__(self, eng, B, nv, eos, mask_id, attn="paged", graphs=None, margins=False, switch_L=None):
        self.e, self.B, self.nv, self.eos, self.mask_id, self.margins = eng, B, nv, set(eos), mask_id, margins
        c = eng.c; dev = eng.dev; self.T = c.T
        D = B + 1; self.D = D; N = D + D * B; self.N = N
        self.attn = attn if attn != "auto" else "paged"
        # hybrid: both graph sets are captured; a step runs `split` once the prefix reaches switch_L (prefix read once), else `paged`
        self.modes = ["paged", "split"] if self.attn == "hybrid" else [self.attn]; self.cur = self.modes[0]
        self.switch_L = switch_L if switch_L is not None else 10 ** 9; self.gsets = {}
        self.graphs_on = (str(dev).startswith("cuda")) if graphs is None else graphs
        # static row structure
        relpos = list(range(D)) + [a + i for a in range(D) for i in range(1, B + 1)]
        tails = [list(range(D))] + [list(range(a + 1)) + [D + a * B + i for i in range(B)] for a in range(D)] if B > 0 else [[0]]
        cu = [0, D] + [D + (a + 1) * B for a in range(D)] if B > 0 else [0, 1]
        self.nseq = len(tails); TM = max(len(x) for x in tails)
        tp = torch.zeros(self.nseq, TM, dtype=torch.int64)
        for s, tl_ in enumerate(tails): tp[s, : len(tl_)] = torch.tensor(tl_)
        wm = torch.zeros(N, N, dtype=torch.int8)
        for j in range(D): wm[j, : j + 1] = 1
        for a in range(D):
            for i in range(1, B + 1):
                r = D + a * B + i - 1; wm[r, : a + 1] = 1; wm[r, D + a * B : r + 1] = 1
        # consistency: the page-table view and the window mask describe the same visibility
        for s, tl_ in enumerate(tails):
            for qi, r in enumerate(range(cu[s], cu[s + 1])):
                vis = set(tl_[: len(tl_) - (cu[s + 1] - cu[s]) + qi + 1]); assert vis == set(wm[r].nonzero().flatten().tolist()), (B, s, r)
        self.relpos = torch.tensor(relpos, device=dev); self.tailpad = tp.to(dev); self.lens = torch.tensor([len(x) for x in tails], device=dev)
        self.cu_q = torch.tensor(cu, device=dev, dtype=torch.int32); self.max_q = max(cu[i + 1] - cu[i] for i in range(self.nseq))
        self.winmask = wm.to(dev).contiguous(); self.ar_idx = torch.arange(N, device=dev)
        self.col = torch.arange(eng.S, device=dev)[None]
        # static buffers
        self.lcur = torch.zeros(1, dtype=torch.int64, device=dev)
        self.ids = torch.full((N,), mask_id, dtype=torch.int64, device=dev)
        self.X = torch.zeros(N, c.H, device=dev, dtype=eng.dt)
        self.rem = torch.ones(N, device=dev); self.exp = torch.zeros(N, nv, device=dev); self.top = torch.zeros(N, dtype=torch.int64, device=dev)
        self.top2v = torch.zeros(N, 2, device=dev)
        m = Meta(); m.cu_q = self.cu_q; m.max_q = self.max_q; m.winmask = self.winmask
        m.pos = torch.zeros(N, dtype=torch.int64, device=dev); m.slot = torch.zeros(N, dtype=torch.int64, device=dev)
        m.seqlens = torch.zeros(self.nseq, dtype=torch.int32, device=dev); m.page_table = torch.zeros(self.nseq, eng.S, dtype=torch.int32, device=dev)
        m.lc32 = torch.zeros(1, dtype=torch.int32, device=dev)
        self.meta = m
        if str(dev).startswith("cuda"):
            self.h_top = torch.zeros(N, dtype=torch.int64).pin_memory(); self.h_ids = torch.zeros(N, dtype=torch.int64).pin_memory()
            self.h_lc = torch.zeros(1, dtype=torch.int64).pin_memory(); self.h_m = torch.zeros(N, 2).pin_memory()
            self.ev = torch.cuda.Event()
        self.graphs = None

    # ---- device-side work of one pass (captured) ----
    def _prep(self):
        m, lc = self.meta, self.lcur
        m.pos.copy_(lc + self.relpos); m.slot.copy_(lc + self.ar_idx); m.lc32.copy_(lc.to(torch.int32))
        m.seqlens.copy_((lc + self.lens).to(torch.int32))
        if self.cur != "split":
            rel = self.col - lc
            tail = self.tailpad.gather(1, rel.clamp(0, self.tailpad.shape[1] - 1).expand(self.nseq, -1))
            pt = torch.where(rel < 0, self.col, lc + tail)
            pt = torch.where(rel >= self.lens[:, None], torch.zeros_like(pt), pt)
            m.page_table.copy_(pt.to(torch.int32))

    def _pass(self, t):
        e = self.e
        if t == 0:
            self._prep()
            self.X.copy_(F.embedding(self.ids, e.embed)); self.rem.fill_(1.0)
        hn = e.loop_pass(self.X, t, self.meta, self.cur)
        self.X.copy_(hn)
        lam = e.gate(hn)
        lg = F.linear(hn, e.lm_head)[:, : self.nv]
        if t < self.T - 1:
            pt = lam * self.rem; self.rem.mul_(1 - lam)
        else:
            pt = self.rem
        if t == 0: torch.mul(lg.float(), pt[:, None], out=self.exp)
        else: self.exp.add_(lg.float() * pt[:, None])
        if t == self.T - 1:
            self.top.copy_(self.exp.argmax(-1))
            if self.margins: self.top2v.copy_(self.exp.topk(2, -1).values)

    def capture(self):
        if not self.graphs_on: return
        # the warm-up passes write K/V at cache slots [lcur, lcur+N) = [0, N): harmless, every decode starts with a prefill
        for mode in self.modes:
            self.cur = mode
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    for t in range(self.T): self._pass(t)
            torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
            pool = torch.cuda.graph_pool_handle(); gs = []
            for t in range(self.T):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool): self._pass(t)
                gs.append(g)
            torch.cuda.synchronize(); self.gsets[mode] = gs
        self.cur = self.modes[0]; self.graphs = self.gsets[self.cur]

    def run_passes(self, lc=None):
        if len(self.modes) > 1 and lc is not None:
            self.cur = "split" if lc >= self.switch_L else "paged"
        if self.graphs is not None:
            for g in self.gsets[self.cur]: g.replay()
        else:
            for t in range(self.T): self._pass(t)

    # ---- one lattice step: H2D inputs, T passes, D2H argmax, ONE sync ----
    def step(self, lc, ids_host):
        if self.graphs is not None:
            self.h_lc[0] = lc; self.h_ids.copy_(torch.as_tensor(ids_host, dtype=torch.int64))
            self.lcur.copy_(self.h_lc, non_blocking=True); self.ids.copy_(self.h_ids, non_blocking=True)
            self.run_passes(lc)
            self.h_top.copy_(self.top, non_blocking=True)
            if self.margins: self.h_m.copy_(self.top2v, non_blocking=True)
            self.ev.record(); self.ev.synchronize()
            top = self.h_top.tolist(); mg = (self.h_m[:, 0] - self.h_m[:, 1]).tolist() if self.margins else None
        else:
            self.lcur.fill_(lc); self.ids.copy_(torch.as_tensor(ids_host, dtype=torch.int64, device=self.ids.device))
            self.run_passes(lc)
            top = self.top.tolist(); mg = (self.top2v[:, 0] - self.top2v[:, 1]).tolist() if self.margins else None
        return top, mg

    @torch.no_grad()
    def decode(self, prompt_ids, max_new, stop_fn=None, prefill_attn="auto"):
        """returns dict(gen, log=accepted tokens per step, steps, step_ms, gpu_ms, margins per committed token, prefill_ms)"""
        e, B, D = self.e, self.B, self.D; P0 = len(prompt_ids)
        assert P0 + max_new + self.N + B + 8 <= e.S, ("cache too small", P0, max_new, self.N, e.S)
        cuda = str(e.dev).startswith("cuda")
        t0 = time.perf_counter()
        first, m0 = e.prefill(list(prompt_ids), self.nv, attn="torch" if self.attn == "torch" else prefill_attn)
        if cuda: torch.cuda.synchronize()
        t_pref = time.perf_counter() - t0
        known = list(prompt_ids) + [first]; lc = P0; props = None; done = first in self.eos
        log, step_ms, gpu_ms, margins, prop_m = [], [], [], [m0], []
        while not done and len(known) - P0 < max_new:
            ts = time.perf_counter()
            ids = [known[-1]] + (props if props is not None else [self.mask_id] * B) + [self.mask_id] * (D * B)
            top, mg = self.step(lc, ids)
            tg = time.perf_counter()
            a = 0
            if props is not None:
                while a < B and top[a] == props[a]: a += 1
            commit = (props[:a] if props is not None else []) + [top[a]]
            if mg is not None:
                margins.extend(mg[: a + 1]); prop_m.append(mg[D + a * B : D + a * B + B])   # committed tokens' margins; next proposals' margins
            lc += a + 1                                           # token slots 0..a join the prefix: their KV is already at slots Lc..Lc+a
            props = [top[D + a * B + i] for i in range(B)] if B > 0 else None
            n0 = len(known)
            for tk in commit:
                known.append(tk)
                if tk in self.eos: done = True; break
            log.append(len(commit))                               # as the reference (len(commit) even if an EOS cut it short)
            if stop_fn is not None and not done and stop_fn(known[P0:]): done = True
            te = time.perf_counter(); step_ms.append(1000 * (te - ts)); gpu_ms.append(1000 * (tg - ts))
        gen = known[P0:]
        return dict(gen=[t for t in gen if t not in self.eos], gen_raw=gen, log=log, steps=len(log), step_ms=step_ms, gpu_ms=gpu_ms,
                    margins=margins if self.margins else None, prop_margins=prop_m if self.margins else None, prefill_ms=1000 * t_pref)


# ----------------------------------------------------------------------------------------------------------------- checks
@torch.no_grad()
def check_attention(eng, B, attn, Lc=137, seed=0, log=print):
    """one lattice pass with random cache/rows: attention backend `attn` vs torch reference (max abs error of the attention output)"""
    torch.manual_seed(seed); c = eng.c
    for t in range(1):
        for li in range(c.NL):
            eng.kc[t][li][:Lc].normal_(); eng.vc[t][li][:Lc].normal_()
    L = Lattice(eng, B, nv=8, eos=[0], mask_id=1, attn=attn, graphs=False)
    L.lcur.fill_(Lc); L._prep(); Lt = Lattice(eng, B, nv=8, eos=[0], mask_id=1, attn="torch", graphs=False); Lt.lcur.fill_(Lc); Lt._prep()
    N = L.N; H = c.H
    qkv = torch.randn(N, 3 * H, device=eng.dev, dtype=eng.dt)
    q, k, v = qkv[:, :H], qkv[:, H : 2 * H], qkv[:, 2 * H :]
    kc, vc = eng.kc[0][0], eng.vc[0][0]
    kc.view(eng.S, H).index_copy_(0, Lt.meta.slot, k); vc.view(eng.S, H).index_copy_(0, Lt.meta.slot, v)
    ref = attn_torch(q, kc, vc, Lt.meta, c.NH, c.HD)
    if attn == "paged": out = attn_paged(q, kc, vc, L.meta, c.NH, c.HD)
    else: out = attn_split(q, k, v, kc, vc, L.meta, c.NH, c.HD)
    err = (out.float() - ref.float()).abs().max().item()
    log(f"[check] attention {attn} B={B} N={N} Lc={Lc}: max abs err vs torch {err:.5f}")
    return err
