"""sparse.py -- SPARSE synchronous MARS lattice on the decode substrate (engine.py here is a byte copy of ../engine.py).

Dense lattice (engine.Lattice, B proposals): D = B+1 token slots + a full set of B placeholders after EVERY token slot -> (B+1)^2 rows.
Sparse lattice: the same D = B+1 token slots, but placeholder sets only after the token slots listed in `sets` = [(a, s), ...]
(a in 0..B, 0 must be present, s >= 1 placeholders each; s may exceed B).

  rows 0..D-1                 token slots (slot 0 = last committed token, slots 1..B = proposals)       position Lc + j
  rows off_k .. off_k+s_k-1   placeholder i = 1..s_k of set k, which follows token slot a_k             position Lc + a_k + i
  visibility (every pass t, through the prefix's pass-t KV + window rows' pass-t KV):
    token slot j       : token slots 0..j
    placeholder (a, i) : token slots 0..a and placeholders (a, 1..i)          (= exactly the dense lattice's rule)
Accept/commit = dense: c = #leading proposals equal to the preceding token slot's full-depth prediction; commit proposals 1..c +
token slot c's prediction; token slots 0..c's KV is their full-depth KV -> Lc += c + 1. Every committed token is a full-depth
prediction from committed context, so the output equals AR whatever the proposals are (proposals only decide how many tokens a
step commits).

Next proposals after accepting c (the only place the sparsity shows):
  a* = max{a in sets : a <= c}  (the deepest set whose conditioning tokens are all committed; a* = c when slot c has a set)
  placeholder (a*, i) sits at Lc+a*+i and predicts position Lc+a*+i+1; the next window's proposal j sits at Lc' + j = Lc+c+1+j
  -> proposal j = argmax of placeholder (a*, c - a* + j), for j = 1.. while c - a* + j <= s_{a*} (and j <= B).
  Missing proposals (a short tail) are filled with <|mask|> and acceptance is capped at the number of real proposals, so a filler is
  never accepted (accepting one would still be exact: acceptance only ever compares with the full-depth prediction).
  The first step after prefill has no proposals (n_real = 0 -> c = 0), as the dense engine's forced c = 0.
Fixed shapes per configuration: one CUDA graph per (config, pass), one host sync per step (inherited from engine.Lattice).

Config string: "sp:B:a.s+a.s+..." e.g. sp:8:0.8+1.8+8.8 ; shorthand "sp:B:a+a+..." means s = B for each set.
"""
import time
import torch
import engine as E


def parse_sets(spec, B):
    sets = []
    for tok in spec.split("+"):
        if "." in tok: a, s = tok.split("."); sets.append((int(a), int(s)))
        else: sets.append((int(tok), B))
    sets = sorted(sets)
    assert sets[0][0] == 0, f"set after token slot 0 required: {spec}"
    assert len({a for a, _ in sets}) == len(sets) and all(0 <= a <= B and s >= 1 for a, s in sets), spec
    return sets


def parse_mode(mode):
    """'ar' -> ('ar', 0, None); 'latt:B' -> ('dense', B, None); 'sp:B:spec' -> ('sparse', B, sets); 'log:B:S' -> ('sparse', B, all sets size S)"""
    if mode == "ar": return "ar", 0, None
    p = mode.split(":")
    if p[0] == "latt": return "dense", int(p[1]), None
    if p[0] == "sp": B = int(p[1]); return "sparse", B, parse_sets(p[2], B)
    if p[0] == "log": B, S = int(p[1]), int(p[2]); return "sparse", B, [(a, S) for a in range(B + 1)]
    raise ValueError(mode)


def rows_of(mode):
    kind, B, sets = parse_mode(mode)
    if kind == "ar": return 1
    if kind == "dense": return (B + 1) ** 2
    return B + 1 + sum(s for _, s in sets)


class SparseLattice(E.Lattice):
    def __init__(self, eng, B, sets, nv, eos, mask_id, attn="paged", graphs=None, margins=False, switch_L=None, dump=False):
        assert B >= 1
        self.e, self.B, self.nv, self.eos, self.mask_id, self.margins, self.dump = eng, B, nv, set(eos), mask_id, margins, dump
        c = eng.c; dev = eng.dev; self.T = c.T
        self.sets = list(sets); self.set_a = [a for a, _ in sets]; self.set_s = [s for _, s in sets]
        D = B + 1; self.D = D
        offs = []; o = D
        for _, s in sets: offs.append(o); o += s
        self.set_off = offs; N = o; self.N = N
        self.astar = [max(a for a in self.set_a if a <= cc) for cc in range(B + 1)]
        self.kidx = {a: k for k, a in enumerate(self.set_a)}
        self.attn = attn if attn != "auto" else "paged"
        self.modes = ["paged", "split"] if self.attn == "hybrid" else [self.attn]; self.cur = self.modes[0]
        self.switch_L = switch_L if switch_L is not None else 10 ** 9; self.gsets = {}
        self.graphs_on = (str(dev).startswith("cuda")) if graphs is None else graphs
        # static row structure (same conventions as engine.Lattice)
        relpos = list(range(D)) + [a + i for a, s in sets for i in range(1, s + 1)]
        tails = [list(range(D))] + [list(range(a + 1)) + [offs[k] + i for i in range(s)] for k, (a, s) in enumerate(sets)]
        cu = [0, D] + [offs[k] + s for k, (a, s) in enumerate(sets)]
        self.nseq = len(tails); TM = max(len(x) for x in tails)
        tp = torch.zeros(self.nseq, TM, dtype=torch.int64)
        for sq, tl_ in enumerate(tails): tp[sq, : len(tl_)] = torch.tensor(tl_)
        wm = torch.zeros(N, N, dtype=torch.int8)
        for j in range(D): wm[j, : j + 1] = 1
        for k, (a, s) in enumerate(sets):
            for i in range(1, s + 1):
                r = offs[k] + i - 1; wm[r, : a + 1] = 1; wm[r, offs[k] : r + 1] = 1
        for sq, tl_ in enumerate(tails):
            for qi, r in enumerate(range(cu[sq], cu[sq + 1])):
                vis = set(tl_[: len(tl_) - (cu[sq + 1] - cu[sq]) + qi + 1]); assert vis == set(wm[r].nonzero().flatten().tolist()), (B, sets, sq, r)
        assert len(relpos) == N and cu[-1] == N
        self.relpos = torch.tensor(relpos, device=dev); self.tailpad = tp.to(dev); self.lens = torch.tensor([len(x) for x in tails], device=dev)
        self.cu_q = torch.tensor(cu, device=dev, dtype=torch.int32); self.max_q = max(cu[i + 1] - cu[i] for i in range(self.nseq))
        self.winmask = wm.to(dev).contiguous(); self.ar_idx = torch.arange(N, device=dev)
        self.col = torch.arange(eng.S, device=dev)[None]
        self.lcur = torch.zeros(1, dtype=torch.int64, device=dev)
        self.ids = torch.full((N,), mask_id, dtype=torch.int64, device=dev)
        self.X = torch.zeros(N, c.H, device=dev, dtype=eng.dt)
        self.rem = torch.ones(N, device=dev); self.exp = torch.zeros(N, nv, device=dev); self.top = torch.zeros(N, dtype=torch.int64, device=dev)
        self.top2v = torch.zeros(N, 2, device=dev)
        m = E.Meta(); m.cu_q = self.cu_q; m.max_q = self.max_q; m.winmask = self.winmask
        m.pos = torch.zeros(N, dtype=torch.int64, device=dev); m.slot = torch.zeros(N, dtype=torch.int64, device=dev)
        m.seqlens = torch.zeros(self.nseq, dtype=torch.int32, device=dev); m.page_table = torch.zeros(self.nseq, eng.S, dtype=torch.int32, device=dev)
        m.lc32 = torch.zeros(1, dtype=torch.int32, device=dev)
        self.meta = m
        if str(dev).startswith("cuda"):
            self.h_top = torch.zeros(N, dtype=torch.int64).pin_memory(); self.h_ids = torch.zeros(N, dtype=torch.int64).pin_memory()
            self.h_lc = torch.zeros(1, dtype=torch.int64).pin_memory(); self.h_m = torch.zeros(N, 2).pin_memory()
            self.ev = torch.cuda.Event()
        self.graphs = None

    def next_props(self, top, c):
        """proposals for the next window after accepting c (see module docstring): list of real proposals (<= B)"""
        a = self.astar[c]; k = self.kidx[a]; off, s = self.set_off[k], self.set_s[k]
        i0 = c - a                                   # proposal j <- placeholder (a, i0 + j)
        n = max(0, min(self.B, s - i0))
        return [top[off + i0 + j - 1] for j in range(1, n + 1)], (off + i0, n)

    @torch.no_grad()
    def decode(self, prompt_ids, max_new, stop_fn=None, prefill_attn="auto"):
        e, B, D = self.e, self.B, self.D; P0 = len(prompt_ids)
        assert P0 + max_new + self.N + B + 8 <= e.S, ("cache too small", P0, max_new, self.N, e.S)
        cuda = str(e.dev).startswith("cuda")
        t0 = time.perf_counter()
        first, m0 = e.prefill(list(prompt_ids), self.nv, attn="torch" if self.attn == "torch" else prefill_attn)
        if cuda: torch.cuda.synchronize()
        t_pref = time.perf_counter() - t0
        known = list(prompt_ids) + [first]; lc = P0; props = []; done = first in self.eos
        log, nprop, step_ms, gpu_ms, margins, prop_m, dumps = [], [], [], [], [m0], [], []
        filler = [self.mask_id] * B
        while not done and len(known) - P0 < max_new:
            ts = time.perf_counter()
            ids = [known[-1]] + props + filler[len(props):] + [self.mask_id] * (self.N - D)
            top, mg = self.step(lc, ids)
            tg = time.perf_counter()
            c = 0
            while c < len(props) and top[c] == props[c]: c += 1
            commit = props[:c] + [top[c]]
            if self.dump:   # mask-chain predictions of every set whose conditioning slots are all committed (a <= c)
                dumps.append([lc - P0, c] + [[a, top[self.set_off[k]: self.set_off[k] + self.set_s[k]]] for k, a in enumerate(self.set_a) if a <= c])
            nprop.append(len(props))
            nxt, (r0, n) = self.next_props(top, c)
            if mg is not None:
                margins.extend(mg[: c + 1]); prop_m.append(mg[r0: r0 + n])
            lc += c + 1
            props = nxt
            for tk in commit:
                known.append(tk)
                if tk in self.eos: done = True; break
            log.append(len(commit))
            if stop_fn is not None and not done and stop_fn(known[P0:]): done = True
            te = time.perf_counter(); step_ms.append(1000 * (te - ts)); gpu_ms.append(1000 * (tg - ts))
        gen = known[P0:]
        return dict(gen=[t for t in gen if t not in self.eos], gen_raw=gen, log=log, nprop=nprop, steps=len(log), step_ms=step_ms, gpu_ms=gpu_ms,
                    margins=margins if self.margins else None, prop_margins=prop_m if self.margins else None, prefill_ms=1000 * t_pref,
                    dump=dumps if self.dump else None)


def make_decoder(eng, mode, nv, eos, mask_id, **kw):
    kind, B, sets = parse_mode(mode)
    dump = kw.pop("dump", False)
    if kind in ("ar", "dense"): return E.Lattice(eng, B, nv, eos, mask_id, **kw)
    return SparseLattice(eng, B, sets, nv, eos, mask_id, dump=dump or mode.startswith("log:"), **kw)


@torch.no_grad()
def check_attention_mode(eng, mode, attn, Lc=137, seed=0, log=print):
    """one pass of `mode`'s rows with random cache/qkv: backend `attn` vs the torch reference (as engine.check_attention)"""
    torch.manual_seed(seed); c = eng.c
    for li in range(c.NL):
        eng.kc[0][li][:Lc].normal_(); eng.vc[0][li][:Lc].normal_()
    kind, B, sets = parse_mode(mode)
    mk = (lambda at: E.Lattice(eng, B, nv=8, eos=[0], mask_id=8, attn=at, graphs=False)) if kind != "sparse" else \
         (lambda at: SparseLattice(eng, B, sets, nv=8, eos=[0], mask_id=8, attn=at, graphs=False))
    L = mk(attn); L.lcur.fill_(Lc); L._prep(); Lt = mk("torch"); Lt.lcur.fill_(Lc); Lt._prep()
    N = L.N; H = c.H
    qkv = torch.randn(N, 3 * H, device=eng.dev, dtype=eng.dt)
    q, k, v = qkv[:, :H], qkv[:, H : 2 * H], qkv[:, 2 * H :]
    kc, vc = eng.kc[0][0], eng.vc[0][0]
    kc.view(eng.S, H).index_copy_(0, Lt.meta.slot, k); vc.view(eng.S, H).index_copy_(0, Lt.meta.slot, v)
    ref = E.attn_torch(q, kc, vc, Lt.meta, c.NH, c.HD)
    out = E.attn_paged(q, kc, vc, L.meta, c.NH, c.HD) if attn == "paged" else E.attn_split(q, k, v, kc, vc, L.meta, c.NH, c.HD)
    err = (out.float() - ref.float()).abs().max().item()
    log(f"[check] attention {attn} {mode} N={N} Lc={Lc}: max abs err vs torch {err:.5f}")
    return err
