"""
Agentic multi-turn SFT mapping for MARS training on tool-call trajectories
(e.g. ``Gen-Verse/Open-AgentRL-SFT-3K``).

Differences from ``dllm.utils.default_mdlm_sft_map_fn``:

  1. **Passes ``tools``** to ``apply_chat_template`` so the tool definitions are
     rendered into the system prompt -- the model must see the same tool-call
     format it will face at RL time.

  2. **Assistant-only masking across *all* turns.** The default map fn supervises
     only the final message; for a multi-turn agentic trajectory that would train
     on just the last assistant turn and mask every earlier one. Here we set
     ``labels = -100`` everywhere except the tokens the *assistant* generated
     (its content + any inline ``<tool_call>`` / ``<think>`` blocks + the closing
     ``<|im_end|>``). User turns, the system/tools prompt, and tool-response
     (observation) turns are all masked -- matching how the RL trainer masks
     tool/observation tokens.

Robustness note: we tokenize the full conversation ONCE with
``add_generation_prompt=False`` and locate assistant spans by scanning the
rendered token stream for ``<|im_start|>assistant ... <|im_end|>``. We deliberately
do NOT re-render prefixes per turn, because the Qwen3 chat template injects an
empty ``<think>`` scaffold on ``add_generation_prompt=True`` and strips reasoning
from non-final assistant turns, which makes prefix re-rendering inconsistent.
Scanning the single canonical render avoids that entirely.

Validate the masking on real data with ``scripts/validate_agentic_sft.py`` once
the env + tokenizer are available; unit-test the *logic* (tokenizer-independent)
with ``tests/test_agentic_sft_map.py``.
"""

from __future__ import annotations


def _token_text(tokenizer, token_id):
    return tokenizer.decode([int(token_id)])


def assistant_spans(input_ids, tokenizer):
    """Return a list of ``(start, end_inclusive)`` token-index spans that the
    assistant generated, given a fully-rendered chat token sequence.

    Each span runs from the first content token after an ``<|im_start|>assistant\\n``
    header up to and including the next ``<|im_end|>``.
    """
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    n = len(input_ids)
    spans = []
    i = 0
    while i < n:
        if input_ids[i] != im_start:
            i += 1
            continue
        # Locate the end of the role header line ("<role>\n") by finding the first
        # following token whose decoded text contains a newline.
        role_end = None
        j = i + 1
        while j < n and (j - i) <= 8:
            if "\n" in _token_text(tokenizer, input_ids[j]):
                role_end = j
                break
            j += 1
        if role_end is None:
            i += 1
            continue
        role_str = tokenizer.decode(
            [int(t) for t in input_ids[i + 1:role_end + 1]]
        ).strip()
        # Find the closing <|im_end|> for this turn.
        k = role_end + 1
        while k < n and input_ids[k] != im_end:
            k += 1
        if role_str.startswith("assistant"):
            content_start = role_end + 1
            content_end = k if k < n else n - 1  # include <|im_end|> when present
            if content_end >= content_start:
                spans.append((content_start, content_end))
        i = k + 1
    return spans


def agentic_sft_map_fn(row, *, tokenizer, mask_prompt_loss: bool = True) -> dict:
    """Build ``input_ids`` / ``labels`` for an agentic tool-call trajectory row.

    Expects ``row["messages"]`` (chat list) and optionally ``row["tools"]``
    (function definitions). Returns ``input_ids``, ``labels``, ``prompt_len``.
    """
    messages = row["messages"]

    # Skip rows with a message that has neither content nor tool_calls.
    for msg in messages:
        if msg.get("content") is None and not msg.get("tool_calls"):
            return {"input_ids": [], "labels": [], "prompt_len": 0}

    tools = row.get("tools", None)
    kwargs = {}
    if tools:
        kwargs["tools"] = tools

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, **kwargs
    )

    if not mask_prompt_loss:
        return {"input_ids": input_ids, "labels": list(input_ids), "prompt_len": 0}

    labels = [-100] * len(input_ids)
    spans = assistant_spans(input_ids, tokenizer)
    for start, end in spans:
        for t in range(start, min(end + 1, len(input_ids))):
            labels[t] = input_ids[t]

    # prompt_len = number of leading masked (non-supervised) tokens. Used by the
    # length filter; with assistant-only masking the "prompt" is everything before
    # the first assistant span.
    prompt_len = spans[0][0] if spans else len(input_ids)

    # Degenerate: nothing supervised -> drop the row.
    if not spans:
        return {"input_ids": [], "labels": [], "prompt_len": 0}

    return {"input_ids": input_ids, "labels": labels, "prompt_len": prompt_len}
