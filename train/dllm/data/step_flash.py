"""Loader for ``stepfun-ai/Step-3.5-Flash-SFT``.

The repo ships three views of the same data (see its README):

  * ``json/<domain>/chunk_*.json`` -- canonical RAW training data (a JSON array
    of examples). This is the origin and the only tokenizer-agnostic form.
  * ``tokenizers/`` -- tokenizer snapshots (Step-3.5-Flash, Qwen3).
  * ``compiled/`` -- tokenizer-specific pre-tokenized shards for StepTronOSS.

We load the **raw JSON** so MARS can tokenize with its own (A2D) tokenizer and
chat template. Each raw example has a ``conversations`` list whose turns carry
``role`` / ``content`` / ``loss_mask`` and, on assistant turns, an optional
``reasoning_content`` (the chain-of-thought). We normalize every turn to a
uniform schema (so Arrow doesn't choke on the heterogeneous ``meta`` struct and
the sometimes-absent ``reasoning_content``) and expose it as ``messages`` --
the same field the rest of the pipeline expects.

The Qwen3 chat template renders an assistant turn that has ``reasoning_content``
as ``<think>\n{reasoning_content}\n</think>\n\n{content}``; that ``<think>``
span is exactly what :func:`mars.data.reasoning_sft_map_fn` marks as the
reasoning region for reasoning-only MARS training.
"""

import glob
import json
import os
import re

from datasets import Dataset, DatasetDict

from dllm.utils.utils import get_default_logger

logger = get_default_logger(__name__)


def _normalize_conversations(conversations):
    """Return a uniform-schema ``messages`` list, or ``None`` to drop the row.

    Every message gets the same keys (``role``, ``content``, ``reasoning_content``,
    ``loss_mask``) so the resulting Arrow column has a consistent struct type.
    ``reasoning_content`` is "" when absent; the chat template only reads it on
    assistant turns and ignores empty strings.
    """
    if not conversations:
        return None
    messages = []
    for turn in conversations:
        content = turn.get("content")
        if content is None:
            return None
        messages.append(
            {
                "role": turn.get("role", ""),
                "content": content,
                "reasoning_content": turn.get("reasoning_content") or "",
                "loss_mask": int(turn.get("loss_mask", 1)),
            }
        )
    return messages


def _chunk_sort_key(path: str) -> int:
    m = re.search(r"chunk_(\d+)\.json$", os.path.basename(path))
    return int(m.group(1)) if m else 0


def _resolve_json_dir(name_or_path: str, domain: str) -> str:
    """Resolve to a local directory holding ``chunk_*.json``.

    Accepts either a local path (a repo snapshot dir, the ``json/`` dir, or a
    ``json/<domain>/`` dir) or a HF Hub ``repo_id`` (downloaded with
    ``snapshot_download``, fetching only the requested domain's raw JSON so the
    large ``compiled/`` shards are skipped).
    """
    candidates = [
        name_or_path,
        os.path.join(name_or_path, "json", domain),
        os.path.join(name_or_path, domain),
    ]
    for cand in candidates:
        if os.path.isdir(cand) and glob.glob(os.path.join(cand, "chunk_*.json")):
            return cand

    # Not a local dir with chunks -> treat as a Hub repo id and download the
    # raw JSON for this domain only.
    from huggingface_hub import snapshot_download

    logger.info(
        f"Downloading raw JSON for {name_or_path} (domain={domain}) via snapshot_download..."
    )
    local = snapshot_download(
        repo_id=name_or_path,
        repo_type="dataset",
        allow_patterns=[f"json/{domain}/chunk_*.json"],
    )
    json_dir = os.path.join(local, "json", domain)
    if not glob.glob(os.path.join(json_dir, "chunk_*.json")):
        raise FileNotFoundError(
            f"No chunk_*.json found for domain={domain!r} under {json_dir}"
        )
    return json_dir


def load_dataset_step_flash(
    name_or_path: str,
    chunks: int | None = None,
    domain: str = "general",
) -> DatasetDict:
    """Load Step-3.5-Flash-SFT raw JSON into ``DatasetDict({"train": ...})``.

    Args:
        name_or_path: Hub repo id (``stepfun-ai/Step-3.5-Flash-SFT``) or a local
            snapshot / json directory.
        chunks: if set, only load the first ``chunks`` ``chunk_*.json`` files
            (handy for quick runs; full ``general`` is ~100 chunks).
        domain: which ``json/<domain>`` subset to load (default ``general``).

    Each row has a single ``messages`` field.
    """
    json_dir = _resolve_json_dir(name_or_path, domain)
    files = sorted(
        glob.glob(os.path.join(json_dir, "chunk_*.json")), key=_chunk_sort_key
    )
    if chunks is not None:
        files = files[: int(chunks)]
    if not files:
        raise FileNotFoundError(f"No chunk_*.json found under {json_dir}")
    logger.info(f"Step-3.5-Flash-SFT: loading {len(files)} raw chunk(s) from {json_dir}")

    def _gen():
        for path in files:
            with open(path, "r", encoding="utf-8") as fh:
                examples = json.load(fh)
            for ex in examples:
                messages = _normalize_conversations(ex.get("conversations"))
                if messages:
                    yield {"messages": messages}

    # from_generator streams to an on-disk Arrow cache (memory-light) and is
    # reused across runs via its content hash.
    ds = Dataset.from_generator(
        _gen,
        gen_kwargs={},
    )
    return DatasetDict({"train": ds})
