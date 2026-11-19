"""lattice_model.py -- two-stream training forward for the K-PASS LATE LATTICE (DESIGN.md).

Decoders trained for: decode/late2/late.py LateLattice (K=2, read lkv, ver am, dense latt:B) and its K-generalisation
decode/late_k/latek.py LateKLattice (lk:K:latt:B). Every lattice block row enters at pass E = T-K+1 and runs K passes;
context positions committed r steps ago with r*K < T are read from the K/V their own TOKEN SLOT wrote in the committing step (lateKV),
everything older (and the prompt) from the exact full-depth K/V (DESIGN.md section 1).

Training forward over x_0..x_{n-1} (B_s sequences, right-padded):
  stream 1   rows 0..n-1        plain full-depth Ouro (T passes, causal, gate mixture)
  stream 2   rows n..2n-1       CHAIN rows: row n+p = the joined token slot of position p (input E[x_p]); exactly the K/V later steps read
             rows 2n..2n+E-1    EXTRA rows of selected blocks: non-joined token slots (input E[x_{c+j}]) and placeholders (input E[<|mask|>])
  passes 1..E-1 : stream 1 alone (SDPA is_causal);  passes E..T : [stream 1 | stream 2] with the bool mask of build_lattice (one mask for
  all K passes: the late/full choice r*K < T does not depend on the pass). Stream-2 readout = mixture over its own K passes, restarted.
A saved checkpoint is an ordinary A2D-Ouro checkpoint (LatticeOuroLM only adds a code path taken when lat_K is given).
"""
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers.utils import ModelOutput

from dllm.pipelines.a2d.models.ouro.modeling_a2d_ouro import A2DOuroConfig, A2DOuroLMHeadModel


# ------------------------------------------------------------------------------------------------ layout
@dataclass
class LatBatch:
    K: int
    B: int
    T: int
    n: int
    E: int                                  # padded number of extra rows (multiple of `mult`, may be 0)
    ex_ids: torch.Tensor                    # [Bs, E] long  (pad: mask id)
    ex_pos: torch.Tensor                    # [Bs, E] long  (pad: 0)
    mask: torch.Tensor                      # [Bs, 1, R, R] bool, R = 2n + E
    rows1: torch.Tensor                     # [Bs, n] bool  stream-1 loss rows (label at p+1)
    tgt1: torch.Tensor                      # [Bs, n] long  x_{p+1} (-100 elsewhere)
    tok_sel: torch.Tensor                   # [Bs, n+E] bool  stream-2 rows with a token-slot loss
    tok_tgt: torch.Tensor                   # [Bs, n+E] long
    tok_pos: torch.Tensor                   # [Bs, n+E] long  position of the row (for the stream-1 teacher of KL)
    ph_sel: torch.Tensor                    # [Bs, n+E] bool  placeholder rows
    ph_tgt: torch.Tensor                    # [Bs, n+E] long
    info: list = field(default_factory=list)  # per sequence: dict(a, vend, anchors, ells, blocks={i: {lattice row r: stream-2 row}})

    def to(self, dev):
        for k in ("ex_ids", "ex_pos", "mask", "rows1", "tgt1", "tok_sel", "tok_tgt", "tok_pos", "ph_sel", "ph_tgt"):
            setattr(self, k, getattr(self, k).to(dev))
        return self


def sample_anchors(a, n, B, alpha, rng):
    """step boundaries c_1 = a (l_1 = 1: the first step after prefill has no proposals), then l = 1 + min(B, Geometric(alpha))"""
    anchors, ells = [a], [1]; c = a + 1
    while c < n:
        g = 0
        while g < B and rng.random() < alpha: g += 1
        anchors.append(c); ells.append(1 + g); c += 1 + g
    return anchors, ells


def block_rows(B):
    """late.py's block-local row order for the dense shape latt:B: token slots 0..B, then set a = 0..B, placeholder i = 1..B.
    returns list of (r, kind, j_or_a, i, relpos)"""
    D = B + 1; out = [(j, "tok", j, 0, j) for j in range(D)]
    for a in range(D):
        for i in range(1, B + 1): out.append((D + a * B + i - 1, "ph", a, i, a + i))
    return out


def build_lattice(input_ids, labels, K, B, T, stop_ids, mask_id, alpha=0.8, extra_max=1024, extra_frac=1.0, rng=None,
                  all_blocks=False, anchors=None, mult=16):
    """layout + mask of one micro-batch (CPU tensors; .to(dev) afterwards). anchors: optional list (per sequence) of anchor lists
    (tests: the same anchors as the teacher-forced decoder); all_blocks: extra rows for every block (tests)."""
    assert 1 <= K <= T and B >= 1, (K, B, T)
    ids_c, lab_c = input_ids.cpu(), labels.cpu()
    Bs, n = ids_c.shape
    rng = rng or random.Random(0)
    stop = torch.zeros(Bs, n, dtype=torch.bool)
    for s in stop_ids:
        if s is not None and s >= 0: stop |= ids_c == int(s)
    nxt = torch.zeros(Bs, n, dtype=torch.bool); nxt[:, :-1] = lab_c[:, 1:] != -100
    tgt1 = torch.full((Bs, n), -100, dtype=torch.long); tgt1[:, :-1] = lab_c[:, 1:]; tgt1[~nxt] = -100
    BR = block_rows(B); D = B + 1
    per = []
    for b in range(Bs):
        sup = (lab_c[b] != -100).nonzero().flatten()
        if len(sup) == 0:
            per.append(dict(a=None, vend=None, anchors=[], ells=[], st=torch.full((n,), -1, dtype=torch.long), extra=[], blocks={})); continue
        a = int(sup[0])
        vend = a                                                    # valid targets: a <= p < vend (label at p+1, no stop token at a..p)
        while vend < n and bool(nxt[b, vend]) and not bool(stop[b, vend]): vend += 1
        if anchors is not None:
            anc = list(anchors[b]); assert anc[0] == a and all(x < y for x, y in zip(anc, anc[1:])), (anc[:3], a)
            ells = [y - x for x, y in zip(anc, anc[1:])] + [max(1, n - anc[-1])]
        else:
            anc, ells = sample_anchors(a, n, B, alpha, rng)
        st = torch.full((n,), -1, dtype=torch.long)
        for i, (c, l) in enumerate(zip(anc, ells)): st[c: min(c + l, n)] = i
        # candidate blocks: slot 0 has a target; extra rows = non-joined token slots + placeholders with a valid target
        cand = []
        for i, (c, l) in enumerate(zip(anc, ells)):
            if not (a <= c < vend): continue
            rows = [(r, kind, ja, ii, c + rp) for (r, kind, ja, ii, rp) in BR if c + rp < vend and not (kind == "tok" and ja < l)]
            cand.append((i, rows))
        budget = None if all_blocks else int(min(extra_max, extra_frac * (vend - a)))
        if not all_blocks: rng.shuffle(cand)
        chosen, used = [], 0
        for i, rows in cand:
            if budget is not None and used + len(rows) > budget: continue
            chosen.append((i, rows)); used += len(rows)
        chosen.sort()
        per.append(dict(a=a, vend=vend, anchors=anc, ells=ells, st=st, chosen=chosen, extra_n=used))
    E = max([p.get("extra_n", 0) for p in per] + [0]); E = (E + mult - 1) // mult * mult
    R = 2 * n + E; Q = n + E
    mask = torch.zeros(Bs, 1, R, R, dtype=torch.bool)
    ex_ids = torch.full((Bs, E), int(mask_id), dtype=torch.long); ex_pos = torch.zeros(Bs, E, dtype=torch.long)
    tok_sel = torch.zeros(Bs, Q, dtype=torch.bool); tok_tgt = torch.full((Bs, Q), -100, dtype=torch.long); tok_pos = torch.zeros(Bs, Q, dtype=torch.long)
    ph_sel = torch.zeros(Bs, Q, dtype=torch.bool); ph_tgt = torch.full((Bs, Q), -100, dtype=torch.long)
    tri = torch.tril(torch.ones(n, n, dtype=torch.bool)); jj = torch.arange(n)
    info = []
    for b, P in enumerate(per):
        M = mask[b, 0]
        M[:n, :n] = tri                                                                   # stream 1: causal, never reads stream 2
        a = P["a"]
        if a is None:                                                                     # no chain: stream-2 rows see self + stream 1 j < p
            M[n:2 * n, :n] = torch.tril(torch.ones(n, n, dtype=torch.bool), -1); M[n:2 * n, n:2 * n] = torch.eye(n, dtype=torch.bool)
            if E: M[2 * n:, 2 * n:] = torch.eye(E, dtype=torch.bool)
            info.append(dict(a=None, blocks={})); continue
        st, anc, vend = P["st"], P["anchors"], P["vend"]
        # query meta (chain rows p >= a; extra rows); unused rows handled after
        q_step = torch.full((Q,), -1, dtype=torch.long); q_bound = torch.full((Q,), -1, dtype=torch.long); q_c = torch.zeros(Q, dtype=torch.long)
        pos_all = torch.arange(n)
        g = pos_all >= a
        q_step[:n][g] = st[g]; q_bound[:n][g] = pos_all[g]; q_c[:n][g] = torch.tensor(anc, dtype=torch.long)[st[g]]
        blocks = {}; e = 0; ex_meta = []                                                  # (e, block i, kind, j_or_a, i)
        for i, rows in P["chosen"]:
            c, l = anc[i], P["ells"][i]
            bm = {}
            for j in range(min(l, D)):
                if c + j < n: bm[j] = c + j                                               # joined slots = chain rows (stream-2 row index p)
            for (r, kind, ja, ii, pp) in rows:
                ex_ids[b, e] = int(ids_c[b, pp]) if kind == "tok" else int(mask_id); ex_pos[b, e] = pp
                q_step[n + e] = i; q_c[n + e] = c; q_bound[n + e] = c + ja
                ex_meta.append((e, i, kind, ja, ii, l)); bm[r] = n + e
                e += 1
            blocks[i] = bm
        used = q_step >= 0
        stj = st[None, :]; isgen = (jj >= a)[None, :]
        qs, qb, qc = q_step[:, None], q_bound[:, None], q_c[:, None]
        late_ok = isgen & (jj[None] <= qb) & (stj <= qs) & ((qs - stj) * K < T)
        full_ok = (jj[None] < qc) & (~isgen | ((qs - stj) * K >= T))
        late_ok &= used[:, None]; full_ok &= used[:, None]
        # unused chain rows (p < a): self + stream 1 j < p;  unused extra rows (padding): self only
        un = (~used[:n]).nonzero().flatten()
        late_ok[un, un] = True; full_ok[un] = (jj[None] < un[:, None])
        M[n:, :n] = full_ok; M[n:, n:2 * n] = late_ok
        if E:
            XX = torch.eye(E, dtype=torch.bool)                                            # padding rows: self only
            if ex_meta:
                m_ = len(ex_meta)
                eb = torch.tensor([x[1] for x in ex_meta]); ek = torch.tensor([x[2] == "ph" for x in ex_meta])
                ej = torch.tensor([x[3] for x in ex_meta]); ei = torch.tensor([x[4] for x in ex_meta])
                same = eb[:, None] == eb[None, :]
                # query token slot j1 / placeholder (a=j1, i1): extra token slots j2 <= j1 (all extra slots are >= l);
                # placeholders only within the own set, i2 <= i1
                XX[:m_, :m_] = same & ((~ek[None, :] & (ej[None, :] <= ej[:, None])) |
                                       (ek[:, None] & ek[None, :] & (ej[None, :] == ej[:, None]) & (ei[None, :] <= ei[:, None])))
            M[2 * n:, 2 * n:] = XX
        # loss rows
        cr = torch.arange(a, vend)
        tok_sel[b, cr] = True; tok_tgt[b, cr] = ids_c[b, cr + 1]; tok_pos[b, :n] = pos_all
        for (e1, i1, k1, j1, ii1, l1) in ex_meta:
            pp = int(ex_pos[b, e1]); tgt = int(ids_c[b, pp + 1])
            if k1 == "tok": tok_sel[b, n + e1] = True; tok_tgt[b, n + e1] = tgt
            else: ph_sel[b, n + e1] = True; ph_tgt[b, n + e1] = tgt
            tok_pos[b, n + e1] = pp
        info.append(dict(a=a, vend=vend, anchors=anc, ells=P["ells"], blocks=blocks, n_extra=len(ex_meta)))
    return LatBatch(K=K, B=B, T=T, n=n, E=E, ex_ids=ex_ids, ex_pos=ex_pos, mask=mask, rows1=nxt, tgt1=tgt1, tok_sel=tok_sel, tok_tgt=tok_tgt,
                    tok_pos=tok_pos, ph_sel=ph_sel, ph_tgt=ph_tgt, info=info)


def mixture_weights(gates):
    """Ouro's stick-breaking exit mixture over gate logits (one per pass, in order); the last pass takes the remainder (fp32)"""
    w = []; rem = None
    for i, g in enumerate(gates):
        lam = torch.sigmoid(g.squeeze(-1).float())
        r = torch.ones_like(lam) if rem is None else rem
        w.append(lam * r if i < len(gates) - 1 else r); rem = r * (1.0 - lam)
    return w


@dataclass
class LatOutput(ModelOutput):
    logits1: Optional[torch.Tensor] = None      # stream 1: [Bs,n,V] or [N1,V] (rows1)
    logits2: Optional[torch.Tensor] = None      # stream 2 (full): [Bs,n+E,V]
    logits_tok: Optional[torch.Tensor] = None   # [Ntok,V] (tok_sel rows)
    logits_ph: Optional[torch.Tensor] = None    # [Nph,V]  (ph_sel rows)


class LatticeOuroLM(A2DOuroLMHeadModel):
    """A2DOuroLMHeadModel + the two-stream lattice forward (taken only when lat is given)."""
    config_class = A2DOuroConfig

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, cache_position=None, logits_to_keep=0, use_weighted_exit=None, exit_at_step=None, exit_threshold=None,
                lat=None, lat_select=True, late_embeds=None, **kwargs):
        if lat is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values,
                                   inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, cache_position=cache_position,
                                   logits_to_keep=logits_to_keep, use_weighted_exit=use_weighted_exit, exit_at_step=exit_at_step,
                                   exit_threshold=exit_threshold, **kwargs)
        return self.lattice(input_ids, lat, select=lat_select, inputs_embeds=inputs_embeds, late_embeds=late_embeds)

    def _pass(self, h, mask, pos, pe, k):
        for layer in self.model.layers[: self.config.num_hidden_layers]:
            h = layer(h, attention_mask=mask, position_ids=pos, past_key_value=None, use_cache=False, cache_position=None,
                      position_embeddings=pe, current_ut=k)
        return self.model.norm(h)

    def _readout(self, hs, rows):
        if rows is not None: hs = [h[rows] for h in hs]
        if len(hs) == 1: return self.lm_head(hs[0])
        w = mixture_weights([self.model.early_exit_gate(h) for h in hs])
        out = None
        for h, wi in zip(hs, w):
            s = self.lm_head(h) * wi.unsqueeze(-1).to(h.dtype)
            out = s if out is None else out + s
        return out

    def lattice(self, input_ids, lat, select=True, inputs_embeds=None, late_embeds=None):
        """select=True: logits1 at rows1, logits_tok / logits_ph at the loss rows; select=False: full logits1 / logits2 (tests).
        inputs_embeds / late_embeds: optional explicit inputs of stream 1 [Bs,n,H] / stream 2 [Bs,n+E,H] (gradient tests)."""
        T = int(self.model.total_ut_steps); K = int(lat.K); assert 1 <= K <= T and lat.T == T, (K, T, lat.T)
        assert self.config._attn_implementation == "sdpa", f"lattice forward needs sdpa, got {self.config._attn_implementation!r}"
        e1 = self.model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if late_embeds is None:
            e2 = torch.cat([e1, self.model.embed_tokens(lat.ex_ids.to(e1.device))], 1) if lat.E else e1
        else: e2 = late_embeds
        Bs, n, _ = e1.shape; dev = e1.device
        pos1 = torch.arange(n, device=dev)[None].expand(Bs, n)
        pos2 = torch.cat([pos1, pos1, lat.ex_pos.to(dev)], 1)
        pe1 = self.model.rotary_emb(e1, pos1); pe2 = self.model.rotary_emb(e1, pos2)
        M = lat.mask.to(dev); entry = T - K
        h1 = e1; h2 = None; hs1 = []; hs2 = []
        for k in range(T):
            if k < entry:
                h1 = self._pass(h1, None, pos1, pe1, k)
            else:
                if k == entry: h2 = e2
                H = self._pass(torch.cat([h1, h2], 1), M, pos2, pe2, k)
                h1, h2 = H[:, :n], H[:, n:]
                hs2.append(h2)
            hs1.append(h1)
        if not select:
            return LatOutput(logits1=self._readout(hs1, None), logits2=self._readout(hs2, None))
        return LatOutput(logits1=self._readout(hs1, lat.rows1.to(dev)), logits_tok=self._readout(hs2, lat.tok_sel.to(dev)),
                         logits_ph=self._readout(hs2, lat.ph_sel.to(dev)))
