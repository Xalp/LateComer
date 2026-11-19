"""Correct (per-start anchored) chunk-extension segmentation + dual-stream loss.

This is the VALIDATED segmentation: chunk boundaries are decided by simulating the
model's own tau-acceptance on the ground-truth response (teacher-forced), with each
chunk's confidence conditioned on the committed clean prefix + the in-chunk masks --
exactly what inference does. (The cheap fixed-block proxy underestimates this badly:
1.11 vs 1.66 tok/fwd at tau=0.9 on Dolci, which is why it never pushed the frontier.)

Ported from the proven a2d chunkext implementation; right-shift convention
(logits[p-1] predicts token[p]).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def chunk_csbd_v2_mask(block_ids: torch.Tensor) -> torch.Tensor:
    """[2L,2L] bool mask for one sequence. Layout [noisy(L)|clean(L)]; block_ids[i]
    = chunk index of logical position i (prompt=0, chunks=1.., pad=large)."""
    L = block_ids.shape[0]
    dev = block_ids.device
    q = torch.arange(2 * L, device=dev)
    kv = torch.arange(2 * L, device=dev)
    x0_q = (q >= L).long()[:, None]
    x0_kv = (kv >= L).long()[None, :]
    pos_q = torch.where(q >= L, q - L, q)[:, None]
    pos_kv = torch.where(kv >= L, kv - L, kv)[None, :]
    bq = block_ids[pos_q.squeeze(1)][:, None]
    bk = block_ids[pos_kv.squeeze(0)][None, :]
    clean_causal = (x0_q == 1) & (x0_kv == 1) & (pos_q >= pos_kv)
    noisy_block_causal = (x0_q == 0) & (x0_kv == 0) & (bq == bk) & (pos_q >= pos_kv)
    noisy_sees_clean = (x0_q == 0) & (x0_kv == 1) & (bq > bk)
    return clean_causal | noisy_block_causal | noisy_sees_clean


@torch.no_grad()
def segment_sequence_fast(model, ids_1d, P, end, mask_id, *, block_size=16,
                          tau=0.9, pad_block=10**6):
    """Teacher-forced tau-segmentation via ONE batched forward over all start
    positions (each correctly anchored: clean prefix [0,p) + a fresh mask block).
    Returns block_ids[L]: prompt=0, chunks=1.., padding-> pad_block."""
    dev = ids_1d.device
    L = ids_1d.shape[0]
    block_ids = torch.full((L,), pad_block, dtype=torch.long, device=dev)
    block_ids[:end] = 0
    starts = list(range(P, end))
    if not starts:
        return block_ids
    B = len(starts)
    Lr = (end - 1) + block_size
    inp = torch.full((B, Lr), mask_id, dtype=ids_1d.dtype, device=dev)
    for r, p in enumerate(starts):
        inp[r, :p] = ids_1d[:p]
        inp[r, p:p + block_size] = mask_id
    cm = torch.ones(Lr, Lr, dtype=torch.bool, device=dev).tril().view(1, 1, Lr, Lr)
    hs = model.model(input_ids=inp, attention_mask=cm,
                     position_ids=torch.arange(Lr, device=dev).unsqueeze(0)).last_hidden_state
    rows_idx = torch.arange(B, device=dev).unsqueeze(1)
    base = torch.tensor([p - 1 for p in starts], device=dev).unsqueeze(1)
    cols = base + torch.arange(block_size, device=dev).unsqueeze(0)
    sl = hs[rows_idx, cols]
    logits = model.lm_head(sl).float()
    conf = F.softmax(logits, dim=-1).max(dim=-1).values  # [B, block_size]
    idx = {p: r for r, p in enumerate(starts)}
    p = P
    blk = 1
    while p < end:
        r = idx[p]
        c = 1
        for j in range(1, block_size):
            if p + j >= end:
                break
            if conf[r, j].item() >= tau:
                c += 1
            else:
                break
        block_ids[p:p + c] = blk
        blk += 1
        p += c
    return block_ids


def chunkext_loss(model, input_ids, labels, block_ids, mask_id, *, clean_ar_weight=1.0):
    """Dual-stream MARS loss with on-policy chunk blocks. Right-shift loss over all
    response tokens covers within-chunk offsets AND each chunk's +1 boundary token.
    Returns (loss, dict)."""
    B, L = input_ids.shape
    dev = input_ids.device
    resp = labels != -100
    noisy_ids = torch.where(resp, mask_id, input_ids)
    concat_ids = torch.cat([noisy_ids, input_ids], dim=1)
    pos = torch.arange(L, device=dev).unsqueeze(0).expand(B, -1)
    concat_pos = torch.cat([pos, pos], dim=1)
    masks = torch.stack([chunk_csbd_v2_mask(block_ids[b]) for b in range(B)]).unsqueeze(1)
    out = model(input_ids=concat_ids, attention_mask=masks, position_ids=concat_pos)
    logits = out.logits
    noisy_logits, clean_logits = logits[:, :L], logits[:, L:]
    tgt = input_ids[:, 1:]
    sup = resp[:, 1:]
    nl = noisy_logits[:, :-1]
    cl = clean_logits[:, :-1]
    loss_noisy = F.cross_entropy(nl[sup], tgt[sup], reduction="mean")
    loss_clean = F.cross_entropy(cl[sup], tgt[sup], reduction="mean")
    return loss_noisy + clean_ar_weight * loss_clean, {
        "loss_noisy": loss_noisy.detach(), "loss_clean": loss_clean.detach(),
    }
