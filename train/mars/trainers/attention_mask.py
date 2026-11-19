"""
MARS training attention mask.

During MARS training, the input is constructed as [noisy | clean] where:
  - noisy (positions 0..n-1): fully masked tokens to be predicted
  - clean (positions n..2n-1): original unmasked tokens providing context

The attention mask enforces three visibility rules:
  1. Clean context (clean-to-clean): strict causal -- each clean token sees
     only previous clean tokens.
  2. Noisy block (noisy-to-noisy): strict causal within the same block --
     each noisy token sees only previous noisy tokens that belong to the
     same block.
  3. Noisy sees clean (noisy-to-clean): inter-block causal -- each noisy
     token in block k can attend to all clean tokens in blocks < k.

These rules prevent information leakage while allowing the model to learn
block-parallel generation from clean context.
"""

import torch


def mars_attention_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the MARS training attention mask.

    Args:
        b: Batch dimension (unused, for flex_attention compatibility).
        h: Head dimension (unused, for flex_attention compatibility).
        q_idx: Query position indices.
        kv_idx: Key/value position indices.
        block_size: Number of tokens per block.
        n: Length of each stream (noisy or clean). The full sequence
           has length 2*n with layout [noisy | clean].

    Returns:
        Boolean mask of shape (Q_LEN, KV_LEN) where True means "allowed
        to attend".

    Structure:
        - Clean Context: Strict Causal
        - Noisy Block: Strict Causal (within-block only)
        - Noisy sees Clean: Inter-Block Causal
    """
    # Indicate whether token belongs to Noisy (0) or Clean (1)
    x0_flag_q = (q_idx >= n).int()
    x0_flag_kv = (kv_idx >= n).int()

    # Logical positions (aligned to 0..n-1)
    pos_q = torch.where(x0_flag_q == 1, q_idx - n, q_idx)
    pos_kv = torch.where(x0_flag_kv == 1, kv_idx - n, kv_idx)

    block_q = pos_q // block_size
    block_kv = pos_kv // block_size

    # 1. Clean Context Self-Attention -- Strict Causal
    clean_causal = (x0_flag_q == 1) & (x0_flag_kv == 1) & (pos_q >= pos_kv)

    # 2. Noisy Block Self-Attention -- Causal AND same block
    noisy_block_causal = (x0_flag_q == 0) & (x0_flag_kv == 0) & (block_q == block_kv) & (pos_q >= pos_kv)

    # 3. Noisy sees Clean Context -- Inter-Block Causal (no leakage)
    noisy_sees_clean = (x0_flag_q == 0) & (x0_flag_kv == 1) & (block_q > block_kv)

    # 4. Combine Masks
    return clean_causal | noisy_block_causal | noisy_sees_clean


def mars_variable_block_mask(block_ids: torch.Tensor) -> torch.Tensor:
    """MARS attention mask with ARBITRARY per-position block ids (one per batch row).

    Identical visibility rules to ``mars_attention_mask`` (clean=strict causal;
    noisy=same-block causal; noisy-sees-clean=strictly-earlier-block), but the
    block of a logical position comes from ``block_ids`` instead of
    ``pos // block_size``. This is what lets MARS use the model's own
    tau-acceptance spans (chunk-extension) as the blocks instead of a fixed size.

    Requirement: ``block_ids`` must be NON-DECREASING along position within each
    row, so "block_q > block_kv" == "kv's chunk ends before q's chunk starts"
    (i.e. noisy chunk k sees exactly the committed clean prefix [0, start_k)).
    Convention used by the trainer: prompt -> 0, response chunks -> 1,2,..., pad
    -> a large sentinel (so chunks never attend padding).

    Args:
        block_ids: LongTensor [b, l] of per-position chunk indices.
    Returns:
        Bool mask [b, 1, 2l, 2l] for the ``[noisy(l) | clean(l)]`` layout;
        True = attend allowed.
    """
    b, l = block_ids.shape
    dev = block_ids.device
    bid2 = torch.cat([block_ids, block_ids], dim=1)            # [b, 2l]
    pos = torch.arange(l, device=dev)
    pos2 = torch.cat([pos, pos])                               # [2l]
    is_clean = torch.arange(2 * l, device=dev) >= l            # [2l]

    bq = bid2[:, :, None]                                      # [b, 2l, 1]
    bk = bid2[:, None, :]                                      # [b, 1, 2l]
    pq = pos2[None, :, None]                                   # [1, 2l, 1]
    pk = pos2[None, None, :]                                   # [1, 1, 2l]
    cq = is_clean[None, :, None]                               # [1, 2l, 1]
    ck = is_clean[None, None, :]                               # [1, 1, 2l]

    clean_causal = cq & ck & (pq >= pk)
    noisy_block_causal = (~cq) & (~ck) & (bq == bk) & (pq >= pk)
    noisy_sees_clean = (~cq) & ck & (bq > bk)
    mask = clean_causal | noisy_block_causal | noisy_sees_clean
    return mask.unsqueeze(1)                                   # [b, 1, 2l, 2l]


def mars_chunk_mask(noisy_logical, clean_len, block_size):
    """MARS mask for a chunked layout ``[noisy_chunk | clean_full]``.

    The full noisy stream is split into chunks of whole blocks; each chunk is
    forwarded as ``[noisy_chunk (m positions) | clean (clean_len positions)]``.
    Because a noisy token only attends (rule 2) to same-block noisy and (rule 3)
    to clean blocks strictly before it, processing the noisy stream in
    block-aligned chunks against the full clean stream is mathematically
    identical to the single 2L forward (verified to fp32 machine precision).

    Args:
        noisy_logical: 1D LongTensor [m] of the chunk's noisy tokens' *logical*
            positions (their original index in 0..clean_len-1).
        clean_len: length of the (full) clean stream.
        block_size: MARS block size.
    Returns:
        Bool mask [m+clean_len, m+clean_len]; True = attend allowed.
    """
    import torch

    device = noisy_logical.device
    m = noisy_logical.shape[0]
    total = m + clean_len
    idx = torch.arange(total, device=device)
    is_noisy = idx < m  # True for the first m positions
    logical = torch.cat(
        [noisy_logical, torch.arange(clean_len, device=device)]
    )
    block = logical // block_size
    nq, nkv = is_noisy[:, None], is_noisy[None, :]
    pq, pkv = logical[:, None], logical[None, :]
    bq, bkv = block[:, None], block[None, :]
    clean_causal = (~nq) & (~nkv) & (pq >= pkv)
    noisy_block_causal = nq & nkv & (bq == bkv) & (pq >= pkv)
    noisy_sees_clean = nq & (~nkv) & (bq > bkv)
    return clean_causal | noisy_block_causal | noisy_sees_clean


def mars_plan_block_mask(block_ids: torch.Tensor) -> torch.Tensor:
    """MARS mask with per-row block ids where the noisy stream is BIDIRECTIONAL inside a block ("plan" slots see each
    other and the block head); clean stays strictly causal; noisy sees clean only in strictly-earlier blocks.
    Same requirements/layout as ``mars_variable_block_mask``; returns bool [b, 1, 2l, 2l]."""
    b, l = block_ids.shape
    dev = block_ids.device
    bid2 = torch.cat([block_ids, block_ids], dim=1)
    pos = torch.arange(l, device=dev)
    pos2 = torch.cat([pos, pos])
    is_clean = torch.arange(2 * l, device=dev) >= l
    bq, bk = bid2[:, :, None], bid2[:, None, :]
    pq, pk = pos2[None, :, None], pos2[None, None, :]
    cq, ck = is_clean[None, :, None], is_clean[None, None, :]
    clean_causal = cq & ck & (pq >= pk)
    noisy_block = (~cq) & (~ck) & (bq == bk)                     # bidirectional within the block
    noisy_sees_clean = (~cq) & ck & (bq > bk)
    return (clean_causal | noisy_block | noisy_sees_clean).unsqueeze(1)
