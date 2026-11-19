"""Model loading, deterministic s_0 noise draws, and GSM8K scoring helpers for Huginn SFT evals."""

import json
import os
import torch
from transformers import AutoTokenizer

from huginn.model.modeling_a2d_huginn import A2DHuginnConfig, A2DHuginnForCausalLM
from huginn.model.raven_modeling_minimal import apply_rotary_emb_complex_like, precompute_freqs_cis
from huginn.eval.score_gsm8k import load_gold, paired_compare, score_one, score_records

NV_BASE = 65536
STOP_IDS = (65505, 65508)  # <|end_text|>, <|end_turn|>
DEFAULT_TASKS = "data/gsm8k_cot.jsonl"
DEFAULT_GOLD = "data/gsm8k_gold.json"


def read_jsonl(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_hg_model_and_tok(model_dir: str, dtype=torch.bfloat16, device: str = "cuda"):
    """Load Base Huginn-0125, Stage-1 ar_sft, or Stage-2 mars_* cleanly without auto_map hazards."""
    cfg = A2DHuginnConfig.from_pretrained(model_dir)
    cfg.model_type = A2DHuginnConfig.model_type
    cfg._attn_implementation = "sdpa"
    cfg.use_cache = False

    model = A2DHuginnForCausalLM.from_pretrained(
        model_dir,
        config=cfg,
        torch_dtype=dtype,
    ).to(device).eval()

    tok = AutoTokenizer.from_pretrained(model_dir, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mask_id = tok.convert_tokens_to_ids("<|mask|>") if "<|mask|>" in tok.get_vocab() else None
    return model, tok, mask_id


def make_noise(m, L: int, seed: int, device, dtype, scale: float = 1.0) -> torch.Tensor:
    """Deterministic position-indexed s_0 draw matching raven_modeling_minimal.initialize_state."""
    E = m.config.n_embd
    std = float(m.config.init_values["std"]) * float(scale)
    if std <= 0.0:
        return torch.zeros((L, E), device=device, dtype=dtype)
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    x = torch.empty((L, E), device=device, dtype=torch.float32)
    torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std, generator=g)
    return (x * float(m.emb_scale)).to(dtype)


def doc_seed(base: int, doc_id: int) -> int:
    return (int(base) * 1000003 + int(doc_id) * 7919 + 17) % (2 ** 62)


def ensure_freqs_cis(m, min_len: int, device) -> torch.Tensor:
    """Return `m.freqs_cis` directly if long enough, else extend via exact `precompute_freqs_cis`."""
    fc = m.freqs_cis
    if fc.shape[1] >= int(min_len):
        return fc
    c = m.config
    hd = c.n_embd // c.num_attention_heads
    new_len = max(int(min_len), fc.shape[1] * 2)
    ext = precompute_freqs_cis(hd, new_len, float(c.rope_base), 1).to(device=device, dtype=fc.dtype)
    return ext


__all__ = [
    "NV_BASE",
    "STOP_IDS",
    "DEFAULT_TASKS",
    "DEFAULT_GOLD",
    "apply_rotary_emb_complex_like",
    "ensure_freqs_cis",
    "read_jsonl",
    "load_hg_model_and_tok",
    "make_noise",
    "doc_seed",
    "load_gold",
    "score_one",
    "score_records",
    "paired_compare",
]
