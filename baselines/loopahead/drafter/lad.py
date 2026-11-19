"""lad.py -- OUR reimplementation of LoopAhead's LAD drafter (their Section 3.1 / Appendix A.5).

Pinned by the paper: two causal Transformer layers, hidden 512, 8 heads, SiLU-gated FFN 1536, RMSNorm, input = a causal
window of w=64 cached early-loop (pass-1) target states + the anchor token's frozen embedding; one query per future
position j, built from (anchor embedding, latest projected context state, position embedding), all queries built before
any token is selected; output projected back to the target hidden size, ADDED to the latest early-loop state, and read
through the FROZEN target output head (their Eq. 3). Trained to predict K_train=7 future positions (their Eq. 5).

Our recorded choices where the paper is silent (PLAN.md section 4): the three input projections are separate bias-free
linears; the query combination is a SUM; position embeddings are LEARNED, one per window slot (0..63, right-aligned so
the latest state is always slot 63) and one per query slot; an RMSNorm(2048) is applied to the target states on input;
pre-norm blocks; a final RMSNorm(512) before the output projection. With these choices the parameter count is
11,051,008 == the paper's stated Raven-Llama base-drafter count ("~11.1M" for Ouro), which is the evidence for them.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (self.weight.float() * xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)


class Block(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.h = heads; self.hd = d // heads
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, ffn, bias=False); self.up = nn.Linear(d, ffn, bias=False)
        self.down = nn.Linear(ffn, d, bias=False)

    def forward(self, x, amask):
        # amask [B, 1, L, L] additive (0 / -inf)
        B, L, d = x.shape
        h = self.n1(x)
        q = self.q(h).view(B, L, self.h, self.hd).transpose(1, 2)
        k = self.k(h).view(B, L, self.h, self.hd).transpose(1, 2)
        v = self.v(h).view(B, L, self.h, self.hd).transpose(1, 2)
        s = (q.float() @ k.float().transpose(-1, -2)) * (self.hd ** -0.5) + amask
        a = torch.softmax(s, -1).to(v.dtype) @ v
        x = x + self.o(a.transpose(1, 2).reshape(B, L, d))
        h = self.n2(x)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x


class LAD(nn.Module):
    def __init__(self, h_tgt=2048, d=512, heads=8, ffn=1536, layers=2, w=64, k_train=7):
        super().__init__()
        self.w, self.k_train, self.h_tgt, self.d = w, k_train, h_tgt, d
        self.norm_in = RMSNorm(h_tgt)
        self.ctx_proj = nn.Linear(h_tgt, d, bias=False)
        self.last_proj = nn.Linear(h_tgt, d, bias=False)
        self.anc_proj = nn.Linear(h_tgt, d, bias=False)
        self.pos = nn.Embedding(w + k_train, d)
        self.blocks = nn.ModuleList([Block(d, heads, ffn) for _ in range(layers)])
        self.norm_f = RMSNorm(d)
        self.out_proj = nn.Linear(d, h_tgt, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, H, hmask, E_a, K=None):
        """H [B, w, h_tgt] pass-1 target states, RIGHT-aligned (latest at slot w-1, left-pad invalid);
        hmask [B, w] bool (True = valid); E_a [B, h_tgt] anchor embedding; returns delta [B, K, h_tgt].
        Caller computes h_tilde = H[:, -1] + delta and logits = h_tilde @ W_out^T (frozen head)."""
        K = K or self.k_train
        B, w, _ = H.shape
        assert w == self.w
        Hn = self.norm_in(H)
        ctx = self.ctx_proj(Hn) + self.pos.weight[:w].to(H.dtype)
        q = (self.anc_proj(E_a) + self.last_proj(Hn[:, -1]))[:, None, :] + self.pos.weight[w:w + K][None].to(H.dtype)
        x = torch.cat([ctx, q], 1)
        L = w + K
        causal = torch.ones(L, L, dtype=torch.bool, device=H.device).tril()
        vis = causal[None].expand(B, -1, -1) & torch.cat([hmask, hmask.new_ones(B, K)], 1)[:, None, :]
        vis = vis | torch.eye(L, dtype=torch.bool, device=H.device)[None]   # a fully-masked (padded) row would softmax to NaN and poison layer 2; let every row see itself
        amask = torch.zeros(B, 1, L, L, dtype=torch.float32, device=H.device).masked_fill_(~vis[:, None], float("-inf"))
        for blk in self.blocks: x = blk(x, amask)
        return self.out_proj(self.norm_f(x[:, w:]))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    m = LAD()
    print("params:", m.n_params())        # must print 11051008
    H = torch.randn(3, 64, 2048); mk = torch.ones(3, 64, dtype=torch.bool); mk[0, :10] = False
    d = m(H, mk, torch.randn(3, 2048), K=3)
    print("out:", tuple(d.shape))
