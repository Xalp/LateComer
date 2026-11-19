"""
MARS single-sequence multi-token generator for agentic RL rollouts (Option-4).

This is the correctness-first, HuggingFace-`transformers` generator used by the
RL rollout loop (Phase 2 of the agentic-RL integration). It is intentionally
**not** batched and does **not** use a KV cache: it runs one full causal forward
per acceptance step. The goal is a reference implementation whose behavior is
easy to reason about and to assert against standard AR decoding. A block-cached /
batched / vLLM version is a separate, later optimization (see TODOs).

------------------------------------------------------------------------------
Mechanics (MARS left-to-right sliding window)
------------------------------------------------------------------------------
At each step we append ``B`` ``[MASK]`` tokens after the committed prefix and run
ONE causal forward over ``[prefix | B masks]``. We then accept the leftmost
predicted tokens while their confidence (top-1 probability) >= tau, accepting at
least one. The accepted tokens are committed to the prefix and the window slides.

With ``right_shift_logits=True`` (the convention these Qwen3 MARS checkpoints are
trained with), the raw model logit at absolute position ``p`` predicts the token
at position ``p+1`` (i.e. standard next-token AR). For the current window:

    shifted[0] = raw logit at (last committed token)   -> predicts window pos 0
    shifted[j] = raw logit at (mask j-1)                -> predicts window pos j

So **window position 0 is conditioned only on the clean committed prefix** (zero
masks before it). Its distribution is *exactly* the standard clean-AR
distribution. Window positions j>0 are conditioned on j mask tokens and therefore
come from a masked-context distribution that differs from clean AR.

------------------------------------------------------------------------------
Why Option-4 only trains on the window-start token (read this before the loss)
------------------------------------------------------------------------------
Because the window-start token was sampled from exactly the clean-AR
distribution, behavior policy == target policy on that token, at ANY tau. The RL
trainer recomputes its logprob with a plain clean causal AR forward over the
fixed response and gets the same distribution it was sampled from -> standard
on-policy policy gradient, no importance-sampling correction. Tokens at window
positions j>0 were sampled from masked-context distributions and MUST be excluded
from the loss. ``is_window_start`` returned here is precisely that loss mask.

Consequence at tau=1.0: every forward accepts exactly one token, so every token
is a window start, ``accept_log`` is all 1s, and the generated sequence is
token-identical to standard AR greedy decoding. That is the primary correctness
test (``tests/test_mars_generate.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .mars_sampler import build_pure_causal_attention_mask
from .mars_cached_sampler import (
    build_cached_attention_mask,
    compute_cached_position_ids,
    trim_kv_cache,
)


def window_starts_from_accept_log(accept_log: list[int]) -> torch.Tensor:
    """Boolean mask (length == total accepted tokens) that is True at the first
    token of every acceptance window.

    The window-start positions are the cumulative-sum boundaries of ``accept_log``:
    ``0, N_1, N_1+N_2, ...``. This is the Option-4 per-token policy-gradient loss
    mask within a single generation segment.
    """
    total = int(sum(accept_log))
    is_start = torch.zeros(total, dtype=torch.bool)
    cursor = 0
    for n in accept_log:
        if n <= 0:
            continue
        is_start[cursor] = True
        cursor += n
    return is_start


@dataclass
class MarsGenerateOutput:
    token_ids: torch.Tensor       # [num_generated] generated token ids (no prefix)
    accept_log: list[int]         # tokens accepted per forward pass, e.g. [1, 3, 2, ...]
    is_window_start: torch.Tensor # [num_generated] bool, True at each window's first token
    num_forwards: int             # == len(accept_log)


def _accept_from_block(block_logits, tau, temperature, one_token_mode, stop_token_ids, budget):
    """Left-to-right acceptance on one block's (already right-shifted) logits.

    block_logits: [n, V]. Returns (accepted_token_ids: list[int], stop: bool).
    Gate = intrinsic top-1 prob (>= tau); emitted token = argmax (temp=0) or sampled.
    Always accepts >=1; stops at the first stop token; trims to `budget`.
    """
    probs = F.softmax(block_logits.float(), dim=-1)
    top1_prob, argmax_tok = probs.max(dim=-1)
    if temperature and temperature > 0:
        emitted = torch.distributions.Categorical(logits=block_logits / temperature).sample()
    else:
        emitted = argmax_tok
    accepted = []
    for j in range(block_logits.shape[0]):
        if j > 0:
            if one_token_mode:
                break
            if float(top1_prob[j].item()) < tau:
                break
        tok = int(emitted[j].item())
        accepted.append(tok)
        if tok in stop_token_ids:
            break
    if len(accepted) > budget:
        accepted = accepted[:budget]
    stop = False
    if any(t in stop_token_ids for t in accepted):
        stop = True
        cut = next(i for i, t in enumerate(accepted) if t in stop_token_ids) + 1
        accepted = accepted[:cut]
    return accepted, stop


@torch.no_grad()
def generate_mars(
    model,
    tokenizer,
    prefix,
    tau: float | None = None,
    B: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    right_shift_logits: bool = True,
    stop_token_ids=None,
    use_cache: bool = True,
    return_dict: bool = False,
    force_ar_in_code: bool = False,
    code_open_id: int = 151657,   # <tool_call>
    code_close_id: int = 151658,  # </tool_call>
):
    """Generate from a MARS model with the left-to-right sliding window.

    Args:
        model: A MARS ``A2DQwen*LMHeadModel`` (accepts a 4D bool attention mask).
        tokenizer: tokenizer with ``mask_token_id`` / ``eos_token_id`` / ``pad_token_id``.
        prefix: prompt token ids (list[int] or 1D LongTensor).
        tau: confidence threshold in (0, 1]. ``None`` or ``>=1.0`` means one-token
            mode (pure AR; every forward accepts exactly one token).
        B: block / window size (number of ``[MASK]`` tokens per forward).
        max_new_tokens: hard cap on generated tokens.
        temperature: sampling temperature for the *emitted* token at accepted
            positions. ``0`` => greedy (argmax). The acceptance GATE always uses
            the intrinsic top-1 probability, independent of ``temperature``.
        right_shift_logits: use the standard-AR logit alignment (True for the
            Qwen3 MARS checkpoints). Leave True unless you know otherwise.
        stop_token_ids: iterable of ids that end generation (defaults to
            ``{eos_token_id}``).
        return_dict: if True return a ``MarsGenerateOutput``; else return the
            tuple ``(token_ids, accept_log)``.

    Returns:
        ``(token_ids, accept_log)`` or ``MarsGenerateOutput``. ``is_window_start``
        is always derivable via ``window_starts_from_accept_log(accept_log)`` and
        is included in the dict form.
    """
    device = next(model.parameters()).device

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if stop_token_ids is None:
        stop_token_ids = {tokenizer.eos_token_id}
    stop_token_ids = set(int(t) for t in stop_token_ids if t is not None)

    assert mask_id is not None, "tokenizer has no mask_token_id; MARS needs a <|mask|> token"
    assert B >= 1

    one_token_mode = (tau is None) or (tau >= 1.0)

    if isinstance(prefix, torch.Tensor):
        prefix_ids = prefix.to(device=device, dtype=torch.long).flatten()
    else:
        prefix_ids = torch.as_tensor(list(prefix), dtype=torch.long, device=device)

    generated: list[int] = []
    accept_log: list[int] = []
    in_code = False   # inside a <tool_call>...</tool_call> span -> force AR (1 tok/fwd)
    def _cur_nmask():
        return 1 if (one_token_mode or (force_ar_in_code and in_code)) else B
    def _update_in_code(accepted):
        nonlocal in_code
        for t in accepted:
            if t == code_open_id: in_code = True
            elif t == code_close_id: in_code = False
    stop = False

    if not use_cache:
        # ---- correctness-first: full causal forward over the whole sequence each step ----
        committed_len = prefix_ids.shape[0]
        x = prefix_ids.view(1, -1)
        while len(generated) < max_new_tokens and not stop:
            n_mask = _cur_nmask()
            masks = torch.full((1, n_mask), mask_id, dtype=torch.long, device=device)
            x_in = torch.cat([x, masks], dim=1)
            T_total = x_in.shape[1]
            T_prefix = committed_len
            attn, pos = build_pure_causal_attention_mask(x_in, pad_token_id=pad_id)
            logits = model(x_in, attention_mask=attn, position_ids=pos).logits  # [1, T_total, V]
            block_logits = logits[:, T_prefix:T_total, :]
            if right_shift_logits:
                shifted = torch.empty_like(block_logits)
                shifted[:, 0:1, :] = logits[:, T_prefix - 1:T_prefix, :]
                shifted[:, 1:, :] = block_logits[:, :-1, :]
                block_logits = shifted
            accepted, stop = _accept_from_block(
                block_logits[0], tau, temperature, one_token_mode or n_mask == 1,
                stop_token_ids, max_new_tokens - len(generated))
            if not accepted:
                break
            accept_log.append(len(accepted))
            generated.extend(accepted)
            _update_in_code(accepted)
            acc = torch.tensor(accepted, dtype=torch.long, device=device).view(1, -1)
            x = torch.cat([x, acc], dim=1)
            committed_len += len(accepted)
    else:
        # ---- KV-cached: forward only [prev_accepted | mask_block] each step (O(L) total) ----
        # The cache holds KV for committed (clean) tokens only. Each step forwards the
        # previously-accepted tokens (so their REAL KV enters the cache) followed by the
        # mask block; afterwards the mask-block KV is cropped off.
        x_prefix = prefix_ids.view(1, -1)
        kv_cache = None
        accepted_ids = None
        while len(generated) < max_new_tokens and not stop:
            n_mask = _cur_nmask()
            T_prefix = x_prefix.shape[1]
            mask_block = torch.full((1, n_mask), mask_id, dtype=torch.long, device=device)
            if kv_cache is None:
                x_full = torch.cat([x_prefix, mask_block], dim=1)
                attn, pos = build_pure_causal_attention_mask(x_full, pad_token_id=pad_id)
                out = model(x_full, attention_mask=attn, position_ids=pos, use_cache=True)
                kv_cache = out.past_key_values
                if right_shift_logits:
                    block_logits = out.logits[:, T_prefix - 1:T_prefix - 1 + n_mask, :]
                else:
                    block_logits = out.logits[:, T_prefix:T_prefix + n_mask, :]
            else:
                cache_len = kv_cache.get_seq_length()
                new_input = torch.cat([accepted_ids, mask_block], dim=1)
                N_prev = accepted_ids.shape[1]
                cached_prefix = x_prefix[:, :cache_len]
                attn = build_cached_attention_mask(cached_prefix, new_input, pad_id)
                pos = compute_cached_position_ids(cached_prefix, new_input, pad_id)
                out = model(new_input, attention_mask=attn, position_ids=pos,
                            past_key_values=kv_cache, use_cache=True)
                kv_cache = out.past_key_values
                if right_shift_logits:
                    block_logits = out.logits[:, N_prev - 1:N_prev - 1 + n_mask, :]
                else:
                    block_logits = out.logits[:, N_prev:N_prev + n_mask, :]
            accepted, stop = _accept_from_block(
                block_logits[0], tau, temperature, one_token_mode or n_mask == 1,
                stop_token_ids, max_new_tokens - len(generated))
            if not accepted:
                break
            accept_log.append(len(accepted))
            generated.extend(accepted)
            _update_in_code(accepted)
            accepted_ids = torch.tensor(accepted, dtype=torch.long, device=device).view(1, -1)
            # crop the mask-block KV; the just-forwarded accepted tokens keep their real KV
            trim_kv_cache(kv_cache, kv_cache.get_seq_length() - n_mask)
            x_prefix = torch.cat([x_prefix, accepted_ids], dim=1)

    token_ids = torch.tensor(generated, dtype=torch.long, device=device)
    is_window_start = window_starts_from_accept_log(accept_log).to(device)

    if return_dict:
        return MarsGenerateOutput(
            token_ids=token_ids,
            accept_log=accept_log,
            is_window_start=is_window_start,
            num_forwards=len(accept_log),
        )
    return token_ids, accept_log
