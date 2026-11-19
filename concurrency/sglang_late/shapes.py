"""shapes.py -- static row structure of the two-pass late lattice.

COPIED from decode/late.py (parse_shape, parse_late, LateLattice.__init__ row structure and
its explicit-mask consistency check, LateLattice.next_props) and sparse.py (parse_mode for the sp:B:a.s+... syntax). No numerics here.
"""
import torch


def parse_sparse(shape):
    """sparse.parse_mode for 'sp:B:a.s+a.s+...' (a set after token slot a with s placeholders; 'a' alone = s = B)"""
    parts = shape.split(":")
    assert parts[0] == "sp" and len(parts) == 3, shape
    B = int(parts[1]); sets = []
    for x in parts[2].split("+"):
        if "." in x: a, s = x.split("."); sets.append((int(a), int(s)))
        else: sets.append((int(x), B))
    sets = sorted(sets)
    assert sets[0][0] == 0 and all(0 <= a <= B and s >= 1 for a, s in sets) and len({a for a, _ in sets}) == len(sets), shape
    return B, sets


def parse_shape(shape):
    if shape == "b0": return 0, []
    if shape.startswith("latt:"):
        B = int(shape.split(":")[1]); return B, [(a, B) for a in range(B + 1)]
    return parse_sparse(shape)


def parse_mode(mode):
    """'l2:<read>:<ver>:<shape>' (late.parse_late; only ver 'am' is served) or 'ar' -> dict"""
    if mode == "ar":
        return dict(kind="ar", read=None, ver=None, B=0, sets=[], shape="ar")
    assert mode.startswith("l2:"), mode
    _, read, ver, shape = mode.split(":", 3)
    assert read in ("old", "lkv"), mode
    assert ver == "am", "only argmax verification is served (gate variants are M3)"
    B, sets = parse_shape(shape)
    return dict(kind="l2", read=read, ver=ver, B=B, sets=sets, shape=shape)


class LateShape:
    """row structure of one request's lattice step (late.LateLattice.__init__): N = 2D + NB rows = [group A (D) | group B (D) | token
    slots (D) | placeholder sets]; nseq = 2 + nblk attention sequences [A | B | token slots | set_0 | ...]."""

    def __init__(self, B, sets):
        D = B + 1; self.B, self.D = B, D
        assert (B == 0 and not sets) or (B >= 1 and sets and sets[0][0] == 0), (B, sets)
        self.sets = list(sets); self.set_a = [a for a, _ in sets]; self.set_s = [s for _, s in sets]
        offs = []; o = D
        for _, s in sets: offs.append(o); o += s
        self.set_off = offs; NB = o; self.NB = NB; self.G = 2 * D; self.N = self.G + NB; self.PH = NB - D
        self.astar = [max(a for a in self.set_a if a <= cc) for cc in range(B + 1)] if sets else [None]
        self.kidx = {a: k for k, a in enumerate(self.set_a)}
        relpos = list(range(D)) + [a + i for a, s in sets for i in range(1, s + 1)]
        tails = [list(range(D))] + [list(range(a + 1)) + [offs[k] + i for i in range(s)] for k, (a, s) in enumerate(sets)]
        cu = [0, D, 2 * D] + [2 * D + D] + [2 * D + offs[k] + s for k, (a, s) in enumerate(sets)]
        self.nblk = len(tails); self.nseq = 2 + self.nblk; TM = max(len(x) for x in tails); self.TM = TM
        tp = torch.zeros(self.nblk, TM, dtype=torch.int64)
        for sq, tl_ in enumerate(tails): tp[sq, : len(tl_)] = torch.tensor(tl_)
        wm = torch.zeros(NB, NB, dtype=torch.int8)
        for j in range(D): wm[j, : j + 1] = 1
        for k, (a, s) in enumerate(sets):
            for i in range(1, s + 1):
                r = offs[k] + i - 1; wm[r, : a + 1] = 1; wm[r, offs[k] : r + 1] = 1
        cub = [0, D] + [offs[k] + s for k, (a, s) in enumerate(sets)]
        for sq, tl_ in enumerate(tails):
            for qi, r in enumerate(range(cub[sq], cub[sq + 1])):
                vis = set(tl_[: len(tl_) - (cub[sq + 1] - cub[sq]) + qi + 1]); assert vis == set(wm[r].nonzero().flatten().tolist()), (B, sets, sq, r)
        assert len(relpos) == NB and cu[-1] == self.N, (len(relpos), NB, cu, self.N)
        self.wm = wm; self.relpos = relpos; self.tails = tails; self.tailpad = tp; self.lens = [len(x) for x in tails]
        self.cu = cu; self.max_q = max(cu[i + 1] - cu[i] for i in range(self.nseq))

    def next_props(self, top, c):
        """late.LateLattice.next_props: proposals for the next step from the deepest set a* <= c (top = this request's NB block argmaxes)"""
        if not self.sets: return []
        a = self.astar[c]; k = self.kidx[a]; off, s = self.set_off[k], self.set_s[k]
        i0 = c - a
        n = max(0, min(self.B, s - i0))
        return [top[off + i0 + j - 1] for j in range(1, n + 1)]
