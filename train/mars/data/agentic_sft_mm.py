"""
Multimodal (Qwen2.5-VL) agentic SFT mapping + collator for MARS training on
DeepEyesV2-style trajectories (``honglyhly/DeepEyesV2_SFT``).

Data format (sharegpt): ``row["messages"]`` with roles system/user/assistant and
literal ``<image>`` placeholders in content (tool observations are *user* turns
wrapped in ``<tool_response>``, and may themselves contain images, e.g. image
search results); ``row["images"]`` is a list of image paths consumed in order.

Mapping (tokenization-time, images NOT materialized):
  * content is split on ``<image>`` into Qwen2.5-VL content items, rendered with
    the stock processor chat template, and tokenized with image-pad expansion —
    this requires reading the images for their grid sizes, but only
    ``input_ids``/``labels``/image *paths* are stored.
  * labels follow ``agentic_sft.assistant_spans``: supervise every assistant
    turn (content + <think>/<tool_call> + closing <|im_end|>), mask system/user/
    tool-response tokens — matching both the cold-start SFT and the RL action
    mask.

Collation (``MMAgenticCollator``): right-pad ids/labels, re-process the images
to ``pixel_values``/``image_grid_thw`` (deterministic, so the grids match the
map-time expansion), and let ``AppendEOSBlockWrapper`` (applied by the caller,
before padding) round each sample to a block multiple.
"""

from __future__ import annotations

import os

import torch

from .agentic_sft import assistant_spans

IMAGE_PLACEHOLDER = "<image>"


def _to_mm_messages(messages):
    """sharegpt messages with inline ``<image>`` -> Qwen2.5-VL content lists."""
    converted = []
    for msg in messages:
        content = msg["content"]
        if IMAGE_PLACEHOLDER not in content:
            converted.append({"role": msg["role"], "content": [{"type": "text", "text": content}]})
            continue
        items = []
        chunks = content.split(IMAGE_PLACEHOLDER)
        for i, chunk in enumerate(chunks):
            if i > 0:
                items.append({"type": "image"})
            if chunk:
                items.append({"type": "text", "text": chunk})
        converted.append({"role": msg["role"], "content": items})
    return converted


def _load_images(paths, image_root):
    from PIL import Image

    images = []
    for p in paths:
        full = p if os.path.isabs(p) else os.path.join(image_root, p)
        with Image.open(full) as im:
            images.append(im.convert("RGB"))
    return images


def agentic_sft_mm_map_fn(
    row, *, processor, image_root, max_length: int | None = None,
    mask_prompt_loss: bool = True,
) -> dict:
    """Build ``input_ids``/``labels``/``image_paths`` for one trajectory row.

    If ``max_length`` is set, over-long trajectories are TURN-truncated (kept up
    to the last ``<|im_end|>`` that fits) rather than dropped -- this keeps the
    sample (just fewer trailing turns) without ever cutting through an image span
    (which would desync image tokens vs pixel_values). image_paths are trimmed to
    the images that remain.
    """
    messages = row["messages"]
    image_paths = list(row.get("images") or [])

    n_placeholders = sum(m["content"].count(IMAGE_PLACEHOLDER) for m in messages)
    drop = {"input_ids": [], "labels": [], "image_paths": [], "prompt_len": 0}
    if n_placeholders != len(image_paths):
        return drop
    for msg in messages:
        if msg.get("content") is None:
            return drop

    text = processor.apply_chat_template(
        _to_mm_messages(messages), tokenize=False, add_generation_prompt=False
    )
    try:
        images = _load_images(image_paths, image_root) if image_paths else None
    except Exception:
        return drop
    enc = processor(text=[text], images=images, return_tensors="pt")
    input_ids = enc["input_ids"][0].tolist()

    tokenizer = processor.tokenizer

    # Hard truncation at max_length (KEEP every trajectory, including the long
    # single-turn reasoning ones). We cut mid-turn if needed -- the partial
    # assistant turn is still supervised content (assistant_spans ends an unclosed
    # turn at the truncation point), so the model still learns MARS generation on
    # it. The ONLY constraint is to never split an image's vision block (that would
    # desync image tokens vs pixel_values), so if the cut lands inside an open
    # <|vision_start|>..<|vision_end|> block we back it up to before that block.
    if max_length is not None and len(input_ids) > max_length:
        vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        cut = max_length
        i = cut - 1
        while i >= 0:
            if input_ids[i] == vision_end:
                break               # last block before cut is closed -> safe
            if input_ids[i] == vision_start:
                cut = i              # cut lands inside an open block -> drop it
                break
            i -= 1
        input_ids = input_ids[:cut]
        n_imgs = sum(1 for t in input_ids if t == vision_start)
        image_paths = image_paths[:n_imgs]

    labels = [-100] * len(input_ids)
    spans = assistant_spans(input_ids, tokenizer)
    if mask_prompt_loss:
        for start, end in spans:
            for t in range(start, min(end + 1, len(input_ids))):
                labels[t] = input_ids[t]
    else:
        labels = list(input_ids)

    if not spans:
        return drop
    return {
        "input_ids": input_ids,
        "labels": labels,
        "image_paths": image_paths,
        "prompt_len": spans[0][0],
    }


class MMAgenticCollator:
    """Pad ids/labels and materialize pixel_values at batch time.

    Padded sequence lengths stay block-multiples because every sample was
    rounded up by ``AppendEOSBlockWrapper`` (the longest sample sets the pad
    target). Image grids are deterministic for a given processor config, so
    they match the image-pad expansion done at map time.
    """

    def __init__(self, processor, image_root, label_pad_token_id: int = -100):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_root = image_root
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features, return_tensors=None):
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids, labels, attention_mask = [], [], []
        all_images = []
        for f in features:
            ids, labs = list(f["input_ids"]), list(f["labels"])
            pad = max_len - len(ids)
            attention_mask.append([1] * len(ids) + [0] * pad)
            input_ids.append(ids + [pad_id] * pad)
            labels.append(labs + [self.label_pad_token_id] * pad)
            all_images.extend(_load_images(f.get("image_paths") or [], self.image_root))

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        if all_images:
            enc = self.processor.image_processor(images=all_images, return_tensors="pt")
            batch["pixel_values"] = enc["pixel_values"]
            batch["image_grid_thw"] = enc["image_grid_thw"]
        # Pass through per-row noisy-chunk tags (set when the dataset is expanded
        # x k for memory-bounded chunked-noisy training).
        if "mars_chunk_id" in features[0]:
            batch["mars_chunk_id"] = torch.tensor(
                [int(f["mars_chunk_id"]) for f in features], dtype=torch.long
            )
            batch["mars_chunk_k"] = torch.tensor(
                [int(f["mars_chunk_k"]) for f in features], dtype=torch.long
            )
        return batch
