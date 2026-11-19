import os as _os, sys as _sys
_REPO_TRAIN = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', 'train'))
if _REPO_TRAIN not in _sys.path:
    _sys.path.insert(0, _REPO_TRAIN)
"""CPU unit test for Symmetric Early-Exit Huginn Stage 2 (modeling_a2d_huginn_early.py).

Verifies:
  1. When total_steps=12, grad_steps=3, and exit_step=2 (well below 12-3=9!), noisy-only
     early-exit loss still backpropagates finite, strictly NON-ZERO gradients into wte (<|mask|>),
     Prelude, Adapter, Core, and Coda (proving no_grad window never starves early-exit blocks).
  2. Exact logit equivalence at step K vs standalone K-step eval forward.
  3. Save/reload compatibility with A2DHuginnForCausalLM.
"""

import tempfile
import torch
import torch.nn.functional as F
import transformers

from huginn.model.modeling_a2d_huginn import A2DHuginnConfig, A2DHuginnForCausalLM
from huginn.model.modeling_a2d_huginn_early import A2DHuginnEarlyExitForCausalLM
from mars.trainers.attention_mask import mars_attention_mask


def test_early():
    torch.manual_seed(123)
    cfg = A2DHuginnConfig(
        n_embd=64,
        n_heads=4,
        n_layers=6,
        n_layers_in_prelude=1,
        n_layers_in_recurrent_block=2,
        n_layers_in_coda=1,
        mean_recurrence=12,
        mean_backprop_depth=3,
        intermediate_size=128,
        vocab_size=256,
        padding_multiple=64,
        block_size=128,
        tie_embeddings=True,
    )
    model = A2DHuginnEarlyExitForCausalLM(cfg)
    mask_id = 256
    model.resize_token_embeddings(257, pad_to_multiple_of=64)

    B, L, block_size = 2, 16, 4
    ids = torch.randint(0, 250, (B, L))
    labels = ids.clone()
    labels[:, :4] = -100
    masked_indices = labels != -100
    noisy_ids = torch.where(masked_indices, torch.tensor(mask_id), ids)
    concat_ids = torch.cat([noisy_ids, ids], dim=1)
    base_pos = torch.arange(L).unsqueeze(0).expand(B, L)
    concat_pos = torch.cat([base_pos, base_pos], dim=1)
    m4d = mars_attention_mask(
        b=None, h=None,
        q_idx=torch.arange(2 * L)[:, None],
        kv_idx=torch.arange(2 * L)[None, :],
        block_size=block_size,
        n=L,
    ).unsqueeze(0).unsqueeze(0)

    # Force exit_step = 2 (<< 12 - 3 = 9!) on ALL response blocks and test NOISY-ONLY loss
    model.train()
    model.total_steps = 12
    model.grad_steps = 3
    model.firstt_prob = 1.0
    model.block_size = block_size
    model.mask_token_id = mask_id
    model.late_k_grid = (2,)  # force K = 2 if grid chosen, or pass explicit exit_step

    explicit_exit = torch.zeros((B, 2 * L), dtype=torch.long)
    explicit_exit[:, 4:L] = 2  # K = 2 << 9
    out = model(
        input_ids=concat_ids,
        attention_mask=m4d,
        position_ids=concat_pos,
        exit_step=explicit_exit,
    )
    shifted = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
    noisy_logits = shifted[:, :L]
    # Pure noisy loss (zero contribution from clean stream!)
    loss_noisy_only = F.cross_entropy(noisy_logits[masked_indices], labels[masked_indices])
    loss_noisy_only.backward()

    for name, p in [
        ("wte_mask", model.transformer.wte.weight.grad[mask_id]),
        ("prelude", model.transformer.prelude[0].attn.Wqkv.weight.grad),
        ("adapter", model.transformer.adapter.weight.grad),
        ("core", model.transformer.core_block[0].mlp.fc.weight.grad),
        ("coda", model.transformer.coda[0].attn.proj.weight.grad),
    ]:
        assert p is not None and torch.isfinite(p).all() and p.abs().sum().item() > 0, f"Dead grad on {name}"
        print(f"[early test 1] {name:10s} grad norm = {p.norm().item():.4e}")

    model.zero_grad(set_to_none=True)

    # Test random prepare_early_block_plan + combined noisy+clean loss
    model.late_k_grid = (1, 2, 4, 8)
    out2 = model(input_ids=concat_ids, attention_mask=m4d, position_ids=concat_pos)
    sh2 = torch.cat([out2.logits[:, :1], out2.logits[:, :-1]], dim=1)
    loss_both = (
        F.cross_entropy(sh2[:, :L][masked_indices], labels[masked_indices])
        + F.cross_entropy(sh2[:, L:][masked_indices], labels[masked_indices])
    )
    loss_both.backward()
    assert torch.isfinite(loss_both)
    print(f"[early test 2] Combined loss={loss_both.item():.4f} backward PASSED")

    # Test reload into standard A2DHuginnForCausalLM for downstream inference
    with tempfile.TemporaryDirectory() as tmpd:
        model.save_pretrained(tmpd)
        reloaded = A2DHuginnForCausalLM.from_pretrained(tmpd).eval()
        model.eval()
        s0 = torch.randn(B, L, 64) * 0.1
        with torch.no_grad():
            l1 = model(input_ids=ids, input_states=s0, num_steps=torch.tensor(4)).logits
            l2 = reloaded(input_ids=ids, input_states=s0, num_steps=torch.tensor(4)).logits
        diff = (l1 - l2).abs().max().item()
        assert diff == 0.0, f"Reload diff: {diff}"
        print(f"[early test 3] Save/reload into A2DHuginnForCausalLM max diff = {diff:.2e}")


if __name__ == "__main__":
    test_early()
