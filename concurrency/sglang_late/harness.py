"""harness.py -- route-A batch-1 driver of the executors without SGLang (CPU tests, in-process GPU equivalence vs late.py).

The KV store is a FlatKV with SGLang's layout (base slot 0 = dummy, req_to_token row rpi maps positions to base slots; slots may be a
shuffled permutation), the prefill is the same full-depth forward the SGLang model runs for an EXTEND batch (core_ops.full_depth via
prefill_meta), and the decode loop is late.LateLattice.decode (host bookkeeping in late_step.LateReqState).
"""
import random, time
import torch
from .core_ops import FlatKV, extend_metas
from .late_step import LateStepExecutor, ARStepExecutor, LateReqState
from .shapes import LateShape, parse_mode


def prefill_rows(P, base_row, pad, device):
    """one request, no prefix: rows = the P prompt positions + (pad > 0) pad rows up to a multiple of `pad` AFTER the prompt (late.py's
    prefill_graphs GEMM shapes); pad rows write the dummy base slot 0 and their key columns point at it (causal: prompt rows never see them).
    Returns (pos, wbase, base_rows [1, Pb], Pb)."""
    Pb = -(-P // pad) * pad if pad else P
    pos = torch.arange(Pb, device=device)
    wbase = torch.zeros(Pb, dtype=torch.int64, device=device); wbase[:P] = base_row[:P].long()
    base_rows = torch.zeros(1, Pb, dtype=torch.int64, device=device); base_rows[0, :P] = base_row[:P].long()
    return pos, wbase, base_rows, Pb


@torch.no_grad()
def prefill_one(core, kv, ids, base_row, nv, pad=0, attn="paged"):
    """T-pass causal forward of one prompt writing all T loop regions; returns (first token, top-2 margin) of the gate mixture readout
    of the last position (engine.OuroEngine.prefill)"""
    dev = core.dev; P = len(ids)
    pos, wbase, base_rows, Pb = prefill_rows(P, base_row, pad, dev)
    ids_t = torch.zeros(Pb, dtype=torch.int64, device=dev); ids_t[:P] = torch.as_tensor(ids, dtype=torch.int64, device=dev)
    sl = torch.tensor([Pb], dtype=torch.int32, device=dev); cu = torch.tensor([0, Pb], dtype=torch.int32, device=dev)
    metas = extend_metas(core.c.T, kv, pos, wbase, base_rows, sl, cu, Pb, Pb)
    exp = core.full_depth(ids_t, metas, kv, torch.tensor([P - 1], device=dev), nv, attn)
    v = exp[0].topk(2)
    return int(v.indices[0]), float(v.values[0] - v.values[1])


class Harness:
    """mode: 'ar' or 'l2:<read>:am:<shape>'. size = base slots of the KV store; perm_seed: None = identity position -> slot p+1,
    else a random permutation of 1..size-1 (tests that nothing depends on slot = position)."""

    def __init__(self, core, mode, W, nv, eos, mask_id, size, attn="paged", graphs=None, perm_seed=None, prefill_pad=0, log=print):
        self.core, self.nv, self.eos, self.mask_id, self.W, self.attn, self.pad = core, nv, set(eos), mask_id, W, attn, prefill_pad
        p = parse_mode(mode); self.p = p; dev = core.dev
        self.sh = LateShape(p["B"], p["sets"]) if p["kind"] == "l2" else None
        scr = self.sh.PH if self.sh else 0
        self.kv = FlatKV.alloc(core.c, size, max(scr, 1), dev, core.dt)
        slots = list(range(1, size))
        if perm_seed is not None: random.Random(perm_seed).shuffle(slots)
        assert len(slots) >= W
        self.rtt = torch.zeros(2, W + 8, dtype=torch.int32, device=dev); self.rtt[1, :W + 8] = torch.tensor(slots[: W + 8], dtype=torch.int32)
        self.rpi = 1
        if self.sh is not None:
            self.ex = LateStepExecutor(core, self.sh, self.kv, self.rtt, W, nv, mask_id, R=1, read=p["read"], attn=attn, graphs=graphs, log=log)
        else:
            self.ex = ARStepExecutor(core, self.kv, self.rtt, W, nv, mask_id, R=1, attn=attn, graphs=graphs, log=log)
        self.ex.rpi.fill_(self.rpi)
        self.ex.capture()

    @torch.no_grad()
    def decode(self, prompt_ids, max_new, stop_fn=None, hook=None):
        """late.LateLattice.decode (and engine.Lattice.decode for ar): same return fields (gen, gen_raw, log, steps, c, nprop, step_ms, ...)"""
        P0 = len(prompt_ids); cuda = self.ex.cuda
        assert P0 + max_new + 2 * (self.sh.N if self.sh else 1) + 16 <= self.W, ("page-table width too small", P0, max_new, self.W)
        t0 = time.perf_counter()
        first, m0 = prefill_one(self.core, self.kv, prompt_ids, self.rtt[self.rpi], self.nv, self.pad, self.attn)
        if cuda: torch.cuda.synchronize()
        t_pref = time.perf_counter() - t0
        known = list(prompt_ids) + [first]; done = first in self.eos
        log, cs, nprop, step_ms = [], [], [], []
        if self.sh is None:
            lc = P0
            while not done and len(known) - P0 < max_new:
                ts = time.perf_counter()
                top = self.ex.step([lc], [self.rpi], [known[-1]])
                lc += 1; tk = top[0]; known.append(tk); log.append(1)
                if tk in self.eos: done = True
                if stop_fn is not None and not done and stop_fn(known[P0:]): done = True
                step_ms.append(1000 * (time.perf_counter() - ts))
        else:
            st = LateReqState(P0); sh = self.sh
            while not done and len(known) - P0 < max_new:
                ts = time.perf_counter()
                ids = st.build_ids(lambda p: known[p], known[-1], sh, self.mask_id)
                top = self.ex.step([(st.lc, st.p1, st.p2)], [self.rpi], [ids])[0]
                npr = len(st.props)
                commit, c = st.advance(top, sh)
                for tk in commit:
                    known.append(tk)
                    if tk in self.eos: done = True; break
                log.append(len(commit)); cs.append(c); nprop.append(npr)
                if hook is not None: hook(self, len(log), st.lc, st.p1, st.p2, known)
                if stop_fn is not None and not done and stop_fn(known[P0:]): done = True
                step_ms.append(1000 * (time.perf_counter() - ts))
        gen = known[P0:]
        return dict(gen=[t for t in gen if t not in self.eos], gen_raw=gen, log=log, c=cs, nprop=nprop, steps=len(log), step_ms=step_ms,
                    prefill_ms=1000 * t_pref, first_margin=m0)

    def kv_at(self, li, region, pos):
        """(K, V) [NH, HD] that this harness holds for position `pos` in loop region `region`, layer li"""
        s = int(self.rtt[self.rpi, pos]); i = region * self.kv.STRIDE + s
        return self.kv.kflat[li][i, 0], self.kv.vflat[li][i, 0]
