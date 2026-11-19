import os as _os, sys as _sys
_TRAIN_DIR = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', 'train'))
if _TRAIN_DIR not in _sys.path:
    _sys.path.insert(0, _TRAIN_DIR)
"""mech_core.py -- teacher-forced mechanism probes for LATE ENTRY in a looped LM (Ouro), 2026-10-23.

Claim under test: "a token does not need its own early passes; it needs the context's deep keys (after mid-pass fine-tuning)".
Everything is teacher forced on fixed text, so no decoding is involved.

Ouro facts used (modeling_ouro.py): the loop index only picks the KV-cache slot (current_ut * num_layers + layer_idx); weights, the
final norm and the exit gate are shared by all passes; pass 1's input is the raw token embedding; the NORMED state of pass k is the
input of pass k+1; a pass-k query reads pass-k keys/values (of positions <= p). Readout = gate mixture over the passes (stick
breaking: pass i gets lam_i * prod_{j<i}(1-lam_j), the last pass the remainder), logits mixed before the softmax.

Objects (positions p in S = the generated positions whose next token is also generated text):
  full forward    the normal T-pass forward over the whole sequence; it provides, for every pass k and layer l, the K/V of all positions
                  (K[k][l], V[k][l]: [Hkv, N, d], post-RoPE) and, at S, the per-pass normed state, gate logit and logits
  k-pass model    ar_k = mixture over the first k passes of the full forward, last pass taking the remainder (== total_ut_steps=k,
                  because pass j never reads a deeper pass)
  slot            a fresh row at p: input = embedding of x_p; its i-th pass reads the context K/V named by a SOURCE LIST plus its OWN
                  current-pass key/value (position p) and nothing else; readout = stick-breaking mixture restarted at the slot's first
                  pass (= the renormalised gate mixture of the late-entry decoders). Source = (pass index of K, pass index of V, bool
                  mask [nq, N] of visible context positions). One SDPA call per layer over [source_1 | source_2 | ... | self] keys.
  cell(D, t)      slot with t passes at pass indices D-t+1..D reading pass-k K/V of ALL positions < p (context run to D passes)
                  L_t(p) = cell(T, t);  cell(D, D) == ar_D;  cell(T, T) == full depth  (tested in test_mech_cpu.py)
All math is done in the model's dtype (fp32 in the probes, TF32 off); readouts and NLL in fp32.
"""
import math

import torch
import torch.nn.functional as F


def load_model(path, dev, dtype=torch.float32):
    """same loader as the campaign (ouro_stream.load): A2DOuroLMHeadModel = gate-mixture readout, also for stock Ouro checkpoints"""
    import transformers
    from dllm.pipelines.a2d import A2DOuroConfig, A2DOuroLMHeadModel
    cfg = A2DOuroConfig.from_pretrained(path); cfg.model_type = "a2d-ouro"; cfg.use_cache = False
    m = A2DOuroLMHeadModel.from_pretrained(path, config=cfg, torch_dtype=dtype, device_map={"": dev}, attn_implementation="sdpa").eval()
    tok = transformers.AutoTokenizer.from_pretrained(path)
    return m, tok


def strict_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), -1)


def mixture(lgs, gls):
    """stick-breaking gate mixture over the given passes (first pass starts with mass 1, the last takes the remainder)
    lgs: list of [n, V] fp32 logits, gls: list of [n] gate logits -> [n, V] fp32"""
    rem = None; exp = None; n = len(lgs)
    for i, (lg, g) in enumerate(zip(lgs, gls)):
        lam = torch.sigmoid(g.float()); r = torch.ones_like(lam) if rem is None else rem
        w = lam * r if i < n - 1 else r; rem = r * (1.0 - lam)
        exp = lg * w[:, None] if exp is None else exp + lg * w[:, None]
    return exp


def summarize(logits, tgt, extra=False):
    """argmax / NLL of the target (fp32); extra: top-1 prob and top1-top2 logit margin"""
    lsm = torch.log_softmax(logits.float(), -1)
    out = dict(am=lsm.argmax(-1).int().cpu().numpy(), nll=(-lsm.gather(1, tgt[:, None])[:, 0]).float().cpu().numpy())
    if extra:
        t2 = logits.float().topk(2, -1).values
        out.update(p1=lsm.max(-1).values.exp().float().cpu().numpy(), margin=(t2[:, 0] - t2[:, 1]).float().cpu().numpy())
    return out


class Probe:
    STATS = ("entropy", "last8", "sink", "self")   # per query & layer, head-averaged: attention entropy (nats), mass on positions p-7..p
                                                   # (the query's own key included), on position 0, on its own key

    def __init__(self, m, nv=None):
        mm = m.model; c = mm.config
        self.m, self.mm = m, mm
        self.layers = list(mm.layers[: c.num_hidden_layers]); self.L = len(self.layers); self.T = int(mm.total_ut_steps)
        self.H, self.Hkv = c.num_attention_heads, c.num_key_value_heads; self.rep = self.H // self.Hkv
        self.hd = self.layers[0].self_attn.head_dim; self.scale = self.layers[0].self_attn.scaling
        self.nv = int(nv or m.lm_head.weight.shape[0]); self.dev = m.lm_head.weight.device; self.dt = m.lm_head.weight.dtype
        assert all(t == "full_attention" for t in c.layer_types), "sliding-window layers not supported"

    # ------------------------------------------------------------------ building blocks (same math as OuroDecoderLayer)
    def _rope(self, N):
        pos = torch.arange(N, device=self.dev)
        cos, sin = self.mm.rotary_emb(torch.zeros(1, 1, 1, device=self.dev, dtype=self.dt), pos[None])
        return cos[0], sin[0]                                                    # [N, d]

    def _qkv(self, layer, h, cos, sin):
        a = layer.self_attn; n = h.shape[0]; x = layer.input_layernorm(h)
        q = a.q_proj(x).view(n, self.H, self.hd).transpose(0, 1)
        k = a.k_proj(x).view(n, self.Hkv, self.hd).transpose(0, 1)
        v = a.v_proj(x).view(n, self.Hkv, self.hd).transpose(0, 1)
        q = q * cos + _rot_half(q) * sin; k = k * cos + _rot_half(k) * sin
        return q, k, v                                                           # [H, n, d], [Hkv, n, d] x2

    def _post(self, layer, h, o):
        n = h.shape[0]; o = layer.self_attn.o_proj(o.transpose(0, 1).reshape(n, self.H * self.hd))
        h = h + layer.input_layernorm_2(o)
        return h + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(h)))

    def _r(self, x):
        return x if self.rep == 1 else x.repeat_interleave(self.rep, 0)

    def _attn(self, q, k, v, mask=None, causal=False, stats=None):
        """q [H, n, d]; k, v [Hkv, M, d]; mask [n, M] bool (True = visible). stats = (qpos [n], kpos [M]) -> also returns [n, 4] stats"""
        k, v = self._r(k), self._r(v)
        if stats is None:
            return F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=None if mask is None else mask[None, None],
                                                  is_causal=causal, scale=self.scale)[0], None
        qpos, kpos = stats
        if mask is None: mask = kpos[None] <= qpos[:, None]
        s = torch.matmul(q, k.transpose(1, 2)) * self.scale
        s = s.masked_fill(~mask[None], float("-inf")); P = torch.softmax(s.float(), -1)   # [H, n, M]
        o = torch.matmul(P.to(v.dtype), v)
        ent = -(P * torch.log(P.clamp_min(1e-30))).sum(-1)
        win = (kpos[None] >= qpos[:, None] - 7) & (kpos[None] <= qpos[:, None]) & mask
        last8 = (P * win[None]).sum(-1); sink = (P * ((kpos == 0)[None] & mask)[None]).sum(-1)
        selfm = (P * ((kpos[None] == qpos[:, None]) & mask)[None]).sum(-1)
        st = torch.stack([ent, last8, sink, selfm], -1).mean(0)                 # head mean -> [n, 4]
        return o, st

    def _lens(self, h):
        return self.m.lm_head(self.mm.norm(h))[:, : self.nv].argmax(-1).int()

    def _readout(self, h):
        return self.m.lm_head(h)[:, : self.nv].float()

    # ------------------------------------------------------------------ the normal T-pass forward
    @torch.no_grad()
    def full_forward(self, ids, S, stats_passes=(), lens_passes=(), passes=None):
        """ids [N] long, S [nq] long (query rows of interest). Returns dict K, V (per pass, per layer), hs / g / lg at S per pass,
        stats {pass: [L, nq, 4]}, lens {pass: [L, nq] argmax of lm_head(norm(residual after layer l))}"""
        T = passes or self.T; N = ids.shape[0]; cos, sin = self._rope(N); self.cos, self.sin = cos, sin
        kpos = torch.arange(N, device=self.dev); h = self.mm.embed_tokens(ids)
        K = [[None] * self.L for _ in range(T)]; V = [[None] * self.L for _ in range(T)]
        hs, gl, lg, stats, lens = [], [], [], {}, {}
        for k in range(T):
            st_k, ln_k = [], []
            for l, layer in enumerate(self.layers):
                q, kk, vv = self._qkv(layer, h, cos, sin); K[k][l], V[k][l] = kk, vv
                o, _ = self._attn(q, kk, vv, causal=True)
                if k in stats_passes:
                    oS, st = self._attn(q[:, S], kk, vv, stats=(S, kpos)); st_k.append(st)
                h = self._post(layer, h, o)
                if k in lens_passes: ln_k.append(self._lens(h[S]))
            h = self.mm.norm(h)
            hs.append(h[S].clone()); gl.append(self.mm.early_exit_gate(h[S])[:, 0].float()); lg.append(self._readout(h[S]))
            if st_k: stats[k] = torch.stack(st_k)
            if ln_k: lens[k] = torch.stack(ln_k)
        return dict(K=K, V=V, hs=hs, g=gl, lg=lg, stats=stats, lens=lens, N=N)

    # ------------------------------------------------------------------ late-entry slots
    @torch.no_grad()
    def slot(self, ids, S, FF, specs, stats=False, lens=False):
        """slot at every p in S. specs: one entry per slot pass, each a list of sources (kd, vd, mask [nq, N] bool): pass i of the slot reads
        K[kd][l] / V[vd][l] of the visible context positions plus its own current-pass key. Returns mixture logits [nq, nv] fp32, final
        normed state [nq, H], optional stats [L, nq, 4] / lens [L, nq] of the LAST pass (used for single-pass slots)."""
        nq = S.shape[0]; cos, sin = self.cos[S], self.sin[S]; N = FF["N"]
        h = self.mm.embed_tokens(ids[S]); eye = torch.eye(nq, dtype=torch.bool, device=self.dev)
        lgs, gls, st_l, ln_l = [], [], [], []
        for i, spec in enumerate(specs):
            last = i == len(specs) - 1
            M = torch.cat([s[2] for s in spec] + [eye], 1)
            kpos = torch.cat([torch.arange(N, device=self.dev)] * len(spec) + [S])
            for l, layer in enumerate(self.layers):
                q, ks, vs = self._qkv(layer, h, cos, sin)
                Kc = torch.cat([FF["K"][s[0]][l] for s in spec] + [ks], 1); Vc = torch.cat([FF["V"][s[1]][l] for s in spec] + [vs], 1)
                o, st = self._attn(q, Kc, Vc, mask=M, stats=(S, kpos) if (stats and last) else None)
                if st is not None: st_l.append(st)
                h = self._post(layer, h, o)
                if lens and last: ln_l.append(self._lens(h))
            h = self.mm.norm(h)
            gls.append(self.mm.early_exit_gate(h)[:, 0].float()); lgs.append(self._readout(h))
        return dict(mix=mixture(lgs, gls), h=h, stats=torch.stack(st_l) if st_l else None, lens=torch.stack(ln_l) if ln_l else None)


def causal_strict(S, N, dev):
    return torch.arange(N, device=dev)[None] < S[:, None]                         # [nq, N]: context position j visible to slot p iff j < p


def cell_specs(D, t, C):
    """cell(D, t): t slot passes at pass indices D-t+1..D (0-based D-t..D-1), each reading the same-index K/V of all positions < p"""
    return [[(D - t + i, D - t + i, C)] for i in range(t)]
