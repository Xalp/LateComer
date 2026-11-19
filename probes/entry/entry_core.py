import os as _os, sys as _sys
_TRAIN_DIR = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', 'train'))
if _TRAIN_DIR not in _sys.path:
    _sys.path.insert(0, _TRAIN_DIR)
"""entry_core.py -- ENTRY-SIGNAL probe for late entry in a looped LM (Ouro), 2026-10-27.

Question: a token that enters a T=4 Ouro stack at the LAST pass (1 own pass, reading ideal deep context K/V) fails without training.
Is that a COORDINATE gap (the raw embedding is not in the normed loop-input space) or an INFORMATION gap (the embedding carries no
context, while the true pass-4 input has three passes of context processing)?

Ouro facts (verified in dllm/pipelines/a2d/models/ouro/modeling_ouro.py, OuroModel.forward lines 585-608 and OuroForCausalLM.forward
lines 745-806): hidden = embed_tokens(x); for every pass: 48 decoder layers, then hidden = self.norm(hidden) -> this NORMED state is
(a) the input of the next pass and (b) exactly what lm_head reads (logits = lm_head(normed state), no further norm) and what the exit
gate reads. tie_word_embeddings = False (v2 / sft / stock26 config.json): embed_tokens.weight (input rows E[v]) and lm_head.weight
(output rows W_out[v]) are separate matrices. Hence the output rows W_out[v] live in the loop-input space (the space of self.norm(h)),
the input rows E[v] only in the pass-1 input space.

Setup (teacher forcing on fixed text; one slot per generated position q with token x_q and generated successor x_{q+1}):
  full forward  normal T-pass forward over the sequence; gives the pass-T K/V of every position, and at q-1 and q the per-pass normed
                states hs[k], gate logits and readouts (mixture = Ouro's gate mixture over the 4 passes = "full depth").
  late slot     ONE pass at pass index T (0-based T-1) for position q: input = an ENTRY state z_q (in place of the true pass-T input
                hs[T-2][q]); every layer's query reads the pass-T keys/values of positions < q from the full forward (ideal deep
                context) plus its own key/value; output = normed state; readout = its pass-T logits (= the gate mixture renormalised to
                the single pass the slot ran: one pass, stick-breaking gives it weight 1). Identical to mech_core cell(T, 1) when
                z_q = E[x_q], and to the full forward's pass-T computation at q when z_q = hs[T-2][q] (ORA).
Entries z_q (names used everywhere; '*' = additions beyond the brief, kept as controls):
  E     E[x_q] (raw input embedding; current late entry)          NE    c * n(E)/||n(E)||, n = the model's own self.norm (gain incl.),
  LE    W E + b (ridge, fit set)                                   TM*   per-token mean of the true pass-T input on the fit set
  OE    c * W_out[x_q]/||W_out[x_q]|| (tied, zero shot)            LOE   W W_out[x_q] + b (ridge)
  SE    sum_v p(v) E[v], p = softmax of the predecessor's full-depth mixture logits at q-1 (vocab cut to the tokenizer's nv tokens)
  SE10  same over the predecessor's top-10, renormalised           NSE   c * n(SE)/||n(SE)||        LSE   W SE + b (ridge)
  OSE   c * u/||u||, u = sum_v p(v) W_out[v] (= lm_head^T softmax(lm_head h)); OSE10 = top-10 version, rescaled
  MU*   mean true pass-T input over the fit set (constant: no token, no context)
  H     hs[T-1][q-1]: the predecessor's normed deep (pass-T) state, which produced p (context, no token identity)
  EH    E + H     NEH*  NE + H     SEH  SE + H     OEH  OE + H     LEH  W [E; H] + b (ridge)
  ORA   hs[T-2][q]: the true pass-T input (oracle; reproduces the full forward's pass T exactly)
c = median ||true pass-T input|| over the fit set (per model). Ridge maps: per-feature standardised, intercept unpenalised, lambda chosen
on held-out fit docs by MSE, then refit on all fit docs; model frozen.
All math in the model dtype (fp32 in the probes, TF32 off); readouts / NLL / statistics fp32 or fp64.
"""
import os, sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.environ.get("MECH_DIR", os.path.abspath(os.path.join(_HERE, "..", "mechanism")))):
    if _p not in sys.path: sys.path.append(_p)
import mech_core as mc  # noqa: E402

# (name, family) in display order; family: tok = token identity only, soft = predecessor-uncertainty token, ctx = carries context, ora
ENTRIES = [("E", "tok"), ("NE", "tok"), ("LE", "tok"), ("TM", "tok"), ("OE", "tok"), ("LOE", "tok"),
           ("SE", "soft"), ("SE10", "soft"), ("NSE", "soft"), ("LSE", "soft"), ("OSE", "soft"), ("OSE10", "soft"),
           ("MU", "ctx"), ("H", "ctx"), ("EH", "ctx"), ("NEH", "ctx"), ("SEH", "ctx"), ("OEH", "ctx"), ("LEH", "ctx"),
           ("ORA", "ora")]
EXTRA = {"TM", "MU", "NEH"}        # beyond the brief (controls)
MAPS = {"LE": "E", "LSE": "SE", "LOE": "O", "LEH": "EH"}   # ridge map -> its feature
TOPK = 10
LAMS = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0)


class EntryProbe(mc.Probe):
    @torch.no_grad()
    def forward_lean(self, ids, Sx, kv_pass=None):
        """the normal T-pass forward (same math as mc.Probe.full_forward). Returns per pass, at the rows Sx: normed state hs[k] [n, d],
        gate logit g[k] [n] fp32, readout lg[k] [n, nv] fp32; and the K/V of pass `kv_pass` (0-based) for all positions (None otherwise)."""
        T, N = self.T, ids.shape[0]; cos, sin = self._rope(N); self.cos, self.sin = cos, sin
        h = self.mm.embed_tokens(ids); K = [None] * self.L; V = [None] * self.L; hs, gl, lg = [], [], []
        for k in range(T):
            for l, layer in enumerate(self.layers):
                q, kk, vv = self._qkv(layer, h, cos, sin)
                if k == kv_pass: K[l], V[l] = kk, vv
                o, _ = self._attn(q, kk, vv, causal=True)
                h = self._post(layer, h, o)
            h = self.mm.norm(h)
            hs.append(h[Sx].clone()); gl.append(self.mm.early_exit_gate(h[Sx])[:, 0].float()); lg.append(self._readout(h[Sx]))
        return dict(K=K, V=V, hs=hs, g=gl, lg=lg, N=N, kv_pass=kv_pass)

    @torch.no_grad()
    def slot_from(self, z, pos, FF, tgt=None, chunk=2048, keep_logits=False):
        """late slot: ONE pass at pass index FF['kv_pass'] for each row r (entry state z[r] at position pos[r]); every layer's query reads
        the FF K/V of positions < pos[r] plus the row's own key/value (rows never see each other). Returns dict am [R] (argmax over nv),
        nll [R] of tgt (if given), h [R, d] normed output state, logits [R, nv] fp32 if keep_logits."""
        N = FF["N"]; ar = torch.arange(N, device=self.dev); am, nll, hh, lgs = [], [], [], []
        for a in range(0, z.shape[0], chunk):
            h = z[a:a + chunk].to(self.dt); p = pos[a:a + chunk]; r = h.shape[0]; cos, sin = self.cos[p], self.sin[p]
            M = torch.cat([ar[None] < p[:, None], torch.eye(r, dtype=torch.bool, device=self.dev)], 1)
            for l, layer in enumerate(self.layers):
                q, ks, vs = self._qkv(layer, h, cos, sin)
                o, _ = self._attn(q, torch.cat([FF["K"][l], ks], 1), torch.cat([FF["V"][l], vs], 1), mask=M)
                h = self._post(layer, h, o)
            h = self.mm.norm(h); lg = self._readout(h)
            am.append(lg.argmax(-1).int()); hh.append(h)
            if tgt is not None:
                nll.append(-torch.log_softmax(lg, -1).gather(1, tgt[a:a + chunk, None])[:, 0])
            if keep_logits: lgs.append(lg)
            del lg
        out = dict(am=torch.cat(am), h=torch.cat(hh))
        if tgt is not None: out["nll"] = torch.cat(nll)
        if keep_logits: out["logits"] = torch.cat(lgs)
        return out


# ---------------------------------------------------------------------------------------------------------------- features / entries
@torch.no_grad()
def features(pr, x, FF, topk=TOPK):
    """raw per-slot features. x [nq] = the slots' tokens x_q. FF must hold rows Sx = [P0-1 .. N-2] (the first slot's predecessor
    first); slots are Sx[1:]. Returns fp32 tensors [nq, d] (E, O, H, SE, SEk, OSE, OSEk, Y = true pass-T input) and per-slot stats."""
    T, nv = pr.T, pr.nv
    Ew, Ow = pr.mm.embed_tokens.weight, pr.m.lm_head.weight
    mixp = mc.mixture(FF["lg"], FF["g"])[:-1]                       # predecessor's full-depth mixture logits [nq, nv]
    P = torch.softmax(mixp.float(), -1)
    Ef, Of = Ew[:nv].float(), Ow[:nv].float()
    tv, ti = P.topk(topk, -1); tw = tv / tv.sum(-1, keepdim=True)
    f = dict(E=Ew[x].float(), O=Ow[x].float(), H=FF["hs"][T - 1][:-1].float(), Y=FF["hs"][T - 2][1:].float(),
             SE=P @ Ef, OSE=P @ Of, SEk=torch.einsum("nk,nkd->nd", tw, Ef[ti]), OSEk=torch.einsum("nk,nkd->nd", tw, Of[ti]))
    f["pxq"] = P.gather(1, x[:, None].long())[:, 0]                 # predecessor's probability of the token that follows
    f["pred_ok"] = P.argmax(-1) == x                                # predecessor's argmax == x_q
    f["se_e_cos"] = F.cosine_similarity(f["SE"], f["E"], dim=-1)
    return f


def rescale(v, c):
    return v * (c / v.norm(dim=-1, keepdim=True).clamp_min(1e-12))


@torch.no_grad()
def entries(pr, f, x, fit, names=None):
    """entry states [nq, d] fp32 for every name in `names` (default: all ENTRIES). fit = dict(c, maps {name: (W, b)}, tm (table [nv, d]),
    tm_seen [nv] bool, mu [d])."""
    names = names or [n for n, _ in ENTRIES]; c = fit["c"]; nrm = lambda v: pr.mm.norm(v.to(pr.dt)).float()
    lin = lambda name, X: X @ fit["maps"][name][0] + fit["maps"][name][1]
    z = {}
    need = lambda *ns: any(n in names for n in ns)
    if need("NE", "NEH"): z["NE"] = rescale(nrm(f["E"]), c)
    if need("LE", "TM"): z["LE"] = lin("LE", f["E"])
    if need("OE", "OEH"): z["OE"] = rescale(f["O"], c)
    for n in names:
        if n in z: continue
        if n == "E": z[n] = f["E"]
        elif n == "TM":
            seen = fit["tm_seen"][x.long()]; z[n] = torch.where(seen[:, None], fit["tm"][x.long()], z["LE"])
        elif n == "LOE": z[n] = lin("LOE", f["O"])
        elif n == "SE": z[n] = f["SE"]
        elif n == "SE10": z[n] = f["SEk"]
        elif n == "NSE": z[n] = rescale(nrm(f["SE"]), c)
        elif n == "LSE": z[n] = lin("LSE", f["SE"])
        elif n == "OSE": z[n] = rescale(f["OSE"], c)
        elif n == "OSE10": z[n] = rescale(f["OSEk"], c)
        elif n == "MU": z[n] = fit["mu"][None].expand_as(f["E"])
        elif n == "H": z[n] = f["H"]
        elif n == "EH": z[n] = f["E"] + f["H"]
        elif n == "NEH": z[n] = z["NE"] + f["H"]
        elif n == "SEH": z[n] = f["SE"] + f["H"]
        elif n == "OEH": z[n] = z["OE"] + f["H"]
        elif n == "LEH": z[n] = lin("LEH", torch.cat([f["E"], f["H"]], -1))
        elif n == "ORA": z[n] = f["Y"]
        else: raise KeyError(n)
    return {n: z[n] for n in names}


def map_feature(f, key):
    return torch.cat([f["E"], f["H"]], -1) if key == "EH" else f[key]


# ---------------------------------------------------------------------------------------------------------------- ridge (sufficient stats)
class RidgeAcc:
    """sufficient statistics of (X [n, D], Y [n, d]) in fp64: n, sum x, sum y, X^T X, X^T Y, sum ||y||^2"""

    def __init__(self, D, d, dev):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)
        self.n, self.sx, self.sy, self.Sxx, self.Sxy, self.syy = 0, z(D), z(d), z(D, D), z(D, d), z(())

    def add(self, X, Y):
        X, Y = X.double(), Y.double(); self.n += X.shape[0]
        self.sx += X.sum(0); self.sy += Y.sum(0); self.Sxx += X.T @ X; self.Sxy += X.T @ Y; self.syy += (Y * Y).sum()

    def merged(self, o):
        m = RidgeAcc.__new__(RidgeAcc); m.n = self.n + o.n
        for k in ("sx", "sy", "Sxx", "Sxy", "syy"): setattr(m, k, getattr(self, k) + getattr(o, k))
        return m


def ridge_solve(acc, lam):
    """standardised ridge with unpenalised intercept: minimise mean ||y - x W - b||^2 + lam ||W_z||^2 in z-scored feature coords.
    Returns (W [D, d], b [d]) in raw coords, fp64."""
    n = acc.n; mx, my = acc.sx / n, acc.sy / n
    Cxx = acc.Sxx / n - torch.outer(mx, mx); Cxy = acc.Sxy / n - torch.outer(mx, my)
    sd = Cxx.diagonal().clamp_min(0).sqrt(); sd = torch.where(sd > 1e-12 * sd.max().clamp_min(1e-30), sd, torch.ones_like(sd))
    Czz = Cxx / torch.outer(sd, sd); Czy = Cxy / sd[:, None]
    Wz = torch.linalg.solve(Czz + lam * torch.eye(Czz.shape[0], dtype=Czz.dtype, device=Czz.device), Czy)
    W = Wz / sd[:, None]; b = my - mx @ W
    return W, b


def ridge_mse(acc, W, b):
    """mean over rows of ||y - x W - b||^2 from sufficient statistics; also the total variance of y (for R^2)"""
    n = acc.n
    sse = acc.syy - 2 * (W * acc.Sxy).sum() - 2 * (b * acc.sy).sum() + ((acc.Sxx @ W) * W).sum() + 2 * ((acc.sx @ W) * b).sum() + n * (b * b).sum()
    var = acc.syy / n - ((acc.sy / n) ** 2).sum()
    return float(sse / n), float(var)


def ridge_fit(tr, va, lams=LAMS):
    """choose lambda by held-out MSE (va), refit on tr+va. Returns dict(W, b fp32, lam, val {lam: (mse, r2)}, train_r2)"""
    val = {}
    for lam in lams:
        W, b = ridge_solve(tr, lam); mse, var = ridge_mse(va, W, b); val[lam] = (mse, 1 - mse / var)
    best = min(val, key=lambda k: val[k][0]); al = tr.merged(va); W, b = ridge_solve(al, best); mse, var = ridge_mse(al, W, b)
    return dict(W=W.float(), b=b.float(), lam=best, val={str(k): v for k, v in val.items()}, val_r2=val[best][1], all_r2=1 - mse / var)


class TokMean:
    """per-token mean of Y (fp64 sums)"""

    def __init__(self, nv, d, dev):
        self.s = torch.zeros(nv, d, dtype=torch.float64, device=dev); self.c = torch.zeros(nv, dtype=torch.float64, device=dev)

    def add(self, x, Y):
        self.s.index_add_(0, x.long(), Y.double()); self.c.index_add_(0, x.long(), torch.ones_like(x, dtype=torch.float64))

    def table(self):
        seen = self.c > 0
        return (self.s / self.c.clamp_min(1)[:, None]).float(), seen


@torch.no_grad()
def centered_cos(A, B):
    """per-row cosine of A[v] and B[v] after subtracting each matrix's mean row"""
    A = A.float() - A.float().mean(0, keepdim=True); B = B.float() - B.float().mean(0, keepdim=True)
    return F.cosine_similarity(A, B, dim=-1)
