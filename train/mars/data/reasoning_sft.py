"""
Reasoning SFT mapping for MARS training on chat data that carries a separate
``reasoning_content`` field (e.g. ``stepfun-ai/Step-3.5-Flash-SFT``).

Each example has ``conversations`` = ordered turns; assistant turns may carry
``reasoning_content`` (the chain-of-thought) in addition to ``content`` (the
final answer). The Qwen3 chat template renders an assistant turn as
``<think>\\n{reasoning_content}\\n</think>\\n\\n{content}``.

This map fn only builds ``input_ids`` / ``labels`` (assistant-only loss). The
*reasoning region* (tokens between ``<think>`` and ``</think>``) is recovered at
train time directly from the token ids by ``MARSReasoningTrainer`` (single-token
tags 151667 / 151668 for Qwen3), so nothing extra needs to be threaded through
the data collator.

Loss masking reuses ``agentic_sft.assistant_spans``: supervise the assistant
content (reasoning + answer + closing ``<|im_end|>``), mask everything else.
"""

from __future__ import annotations

from .agentic_sft import assistant_spans


def _to_messages(conversations):
    """Map dataset turns to chat-template messages, carrying reasoning_content
    so the Qwen3 template emits the <think>...</think> block."""
    msgs = []
    for t in conversations:
        role = t.get("role")
        content = t.get("content") or ""
        m = {"role": role, "content": content}
        rc = t.get("reasoning_content")
        if rc:
            m["reasoning_content"] = rc
        msgs.append(m)
    return msgs


def reasoning_sft_map_fn(row, *, tokenizer, mask_prompt_loss: bool = True) -> dict:
    """Build ``input_ids`` / ``labels`` for a reasoning chat row.

    Expects ``row["conversations"]`` (list of turns with role/content and an
    optional assistant ``reasoning_content``). Returns ``input_ids``, ``labels``,
    ``prompt_len``.
    """
    conversations = row.get("conversations")
    if not conversations:
        return {"input_ids": [], "labels": [], "prompt_len": 0}

    # Drop rows with a turn that has no usable content.
    for t in conversations:
        if (t.get("content") in (None, "")) and not t.get("reasoning_content"):
            return {"input_ids": [], "labels": [], "prompt_len": 0}

    messages = _to_messages(conversations)
    try:
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
    except Exception:
        return {"input_ids": [], "labels": [], "prompt_len": 0}

    if not mask_prompt_loss:
        return {"input_ids": input_ids, "labels": list(input_ids), "prompt_len": 0}

    labels = [-100] * len(input_ids)
    spans = assistant_spans(input_ids, tokenizer)
    for start, end in spans:
        for t in range(start, min(end + 1, len(input_ids))):
            labels[t] = input_ids[t]

    if not spans:
        return {"input_ids": [], "labels": [], "prompt_len": 0}

    prompt_len = spans[0][0]
    return {"input_ids": input_ids, "labels": labels, "prompt_len": prompt_len}
