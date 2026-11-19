import os as _os, sys as _sys
_REPO_TRAIN = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', 'train'))
if _REPO_TRAIN not in _sys.path:
    _sys.path.insert(0, _REPO_TRAIN)
"""CPU unit & mathematical exactness test for Huginn Stage 1 + Stage 2 SFT.

Run from the repository root:
  OMP_NUM_THREADS=1 PYTHONPATH=.:/path/to/mars python huginn/tests/test_hg_sft_cpu.py
"""

import os
import tempfile
import torch
import torch.nn.functional as F
import transformers

from huginn.model.raven_config_minimal import RavenConfig
from huginn.model.raven_modeling_minimal import RavenForCausalLM
from huginn.model.modeling_a2d_huginn import A2DHuginnConfig, A2DHuginnForCausalLM
from mars.trainers.attention_mask import mars_attention_mask


def test_all():
    torch.manual_seed(42)

    # 1. Exactness against original RavenForCausalLM in eval mode
    base_cfg = RavenConfig(
        n_embd=64,
        n_heads=4,
        n_layers=6,
        n_layers_in_prelude=1,
        n_layers_in_recurrent_block=2,
        n_layers_in_coda=1,
        mean_recurrence=6,
        mean_backprop_depth=3,
        intermediate_size=128,
        vocab_size=256,
        padding_multiple=64,
        block_size=128,
        tie_embeddings=True,
    )
    a2d_cfg = A2DHuginnConfig(**base_cfg.to_dict())

    ref_model = RavenForCausalLM(base_cfg).eval()
    a2d_model = A2DHuginnForCausalLM(a2d_cfg).eval()
    a2d_model.load_state_dict(ref_model.state_dict())

    B, L = 2, 16
    ids = torch.randint(0, 250, (B, L))
    init_s = torch.randn(B, L, 64) * 0.1
    with torch.no_grad():
        out_ref = ref_model(input_ids=ids, input_states=init_s, num_steps=6)
        out_a2d = a2d_model(input_ids=ids, input_states=init_s, num_steps=6)
    max_diff = (out_ref.logits - out_a2d.logits).abs().max().item()
    print(f"[test 1] Raven vs A2DHuginn eval max logit diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Logit mismatch: {max_diff}"

    # 2. Stage 1 CausalLM shifted next-token loss & gradients
    a2d_model.train()
    a2d_model.total_steps = 6
    a2d_model.grad_steps = 3
    labels = ids.clone()
    labels[:, :4] = -100  # prompt masked
    out_s1 = a2d_model(input_ids=ids, labels=labels)
    assert out_s1.loss is not None and torch.isfinite(out_s1.loss), "Stage 1 loss non-finite"
    # Verify shifted next-token semantics manually
    expected_loss = F.cross_entropy(
        out_s1.logits[:, :-1, :].reshape(-1, 256),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    assert abs(out_s1.loss.item() - expected_loss.item()) < 1e-6, "Stage 1 shift loss mismatch"
    out_s1.loss.backward()
    for name, p in [
        ("wte", a2d_model.transformer.wte.weight),
        ("prelude", a2d_model.transformer.prelude[0].attn.Wqkv.weight),
        ("adapter", a2d_model.transformer.adapter.weight),
        ("core", a2d_model.transformer.core_block[0].mlp.fc.weight),
        ("coda", a2d_model.transformer.coda[0].attn.proj.weight),
    ]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum().item() > 0, f"Zero/NaN grad on {name}"
    print(f"[test 2] Stage 1 loss={out_s1.loss.item():.4f}, all module grads finite & non-zero")
    a2d_model.zero_grad(set_to_none=True)

    # 3. Stage 2 [noisy | clean] clean-half invariance + multi-batch 2D position_ids + zero leakage
    # Resize embeddings to add <|mask|> = 256 -> padded to 320 (multiple of 64)
    mask_id = 256
    a2d_model.resize_token_embeddings(257, pad_to_multiple_of=64)
    assert a2d_model.config.vocab_size == 320 and a2d_model.config.padded_vocab_size == 320
    assert a2d_model.lm_head.weight.data_ptr() == a2d_model.transformer.wte.weight.data_ptr()

    a2d_model.eval()
    block_size = 4
    masked_indices = labels != -100
    noisy_ids = torch.where(masked_indices, torch.tensor(mask_id), ids)
    concat_ids = torch.cat([noisy_ids, ids], dim=1)  # (B=2, 2L=32)
    base_pos = torch.arange(L).unsqueeze(0).expand(B, L)
    concat_pos = torch.cat([base_pos, base_pos], dim=1)  # (B=2, 2L=32) -- tests B > 1!
    m4d = mars_attention_mask(
        b=None, h=None,
        q_idx=torch.arange(2 * L)[:, None],
        kv_idx=torch.arange(2 * L)[None, :],
        block_size=block_size,
        n=L,
    ).unsqueeze(0).unsqueeze(0)  # (1, 1, 2L, 2L)

    init_clean = torch.randn(B, L, 64) * 0.1
    init_noisy = torch.randn(B, L, 64) * 0.1
    init_concat = torch.cat([init_noisy, init_clean], dim=1)
    with torch.no_grad():
        solo_clean = a2d_model(input_ids=ids, input_states=init_clean, num_steps=torch.tensor(6)).logits
        dual_out = a2d_model(
            input_ids=concat_ids,
            input_states=init_concat,
            attention_mask=m4d,
            position_ids=concat_pos,
            num_steps=torch.tensor(6),
        ).logits
    clean_diff = (solo_clean - dual_out[:, L:, :]).abs().max().item()
    print(f"[test 3] Stage 2 clean-half vs standalone AR max diff (B=2): {clean_diff:.2e}")
    assert clean_diff < 1e-5, f"Clean stream broken by RoPE or 4D mask: {clean_diff}"

    # 4. Stage 2 Late-Entry training step + mask-token gradient + save/reload roundtrip
    a2d_model.train()
    a2d_model.ttt_prob = 1.0  # force Late-Entry block injection on every response block
    a2d_model.block_size = block_size
    a2d_model.mask_token_id = mask_id
    a2d_model.late_k_grid = (1, 2, 3, 4)

    out_s2 = a2d_model(
        input_ids=concat_ids,
        attention_mask=m4d,
        position_ids=concat_pos,
    )
    # Apply right_shift_logits just like MDLMTrainer._postprocess_outputs
    shifted = torch.cat([out_s2.logits[:, :1], out_s2.logits[:, :-1]], dim=1)
    noisy_logits, clean_logits = shifted[:, :L], shifted[:, L:]
    loss_s2 = (
        F.cross_entropy(noisy_logits[masked_indices], labels[masked_indices])
        + F.cross_entropy(clean_logits[masked_indices], labels[masked_indices])
    )
    assert torch.isfinite(loss_s2), "Stage 2 loss non-finite"
    loss_s2.backward()
    wte_grad = a2d_model.transformer.wte.weight.grad
    assert wte_grad is not None and torch.isfinite(wte_grad).all()
    print(
        f"[test 4] Stage 2 Late-Entry loss={loss_s2.item():.4f}, "
        f"<|mask|> row grad norm={wte_grad[mask_id].norm().item():.4e}"
    )

    with tempfile.TemporaryDirectory() as tmpd:
        a2d_model.save_pretrained(tmpd)
        reloaded = transformers.AutoModelForCausalLM.from_pretrained(tmpd)
        assert isinstance(reloaded, A2DHuginnForCausalLM)
        assert reloaded.config.vocab_size == 320 and reloaded.config.padded_vocab_size == 320
        assert reloaded.lm_head.weight.data_ptr() == reloaded.transformer.wte.weight.data_ptr()
    print("[test 5] Checkpoint save_pretrained -> AutoModelForCausalLM.from_pretrained roundtrip PASSED!")

    # 6. End-to-end MARSTrainer + AppendEOSBlockWrapper batch collation & compute_loss
    import dllm
    from mars.trainers import MARSTrainer
    tok_src = os.environ.get("HUGINN_TOKENIZER_DIR", "tomg-group-umd/huginn-0125")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(tok_src, padding_side="right", local_files_only=os.path.isdir(tok_src))
        tok.add_special_tokens({"mask_token": "<|mask|>"})
        vsz = 65536
        sample_feats = [
            {"input_ids": [65504, 100, 101, 200, 201, 202, 203, 65505], "labels": [-100, -100, -100, 200, 201, 202, 203, 65505]},
            {"input_ids": [65504, 105, 210, 211, 212, 65505], "labels": [-100, -100, 210, 211, 212, 65505]},
        ]
    except Exception:
        from tokenizers import Tokenizer, models, pre_tokenizers
        vocab = {f"t{i}": i for i in range(256)}
        vocab.update({"<pad>": 253, "<bos>": 254, "<eos>": 255})
        tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="t0"))
        tk.pre_tokenizer = pre_tokenizers.Whitespace()
        tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", padding_side="right")
        tok.add_special_tokens({"mask_token": "<|mask|>"})
        vsz = 256
        sample_feats = [
            {"input_ids": [254, 10, 11, 20, 21, 22, 23, 255], "labels": [-100, -100, -100, 20, 21, 22, 23, 255]},
            {"input_ids": [254, 15, 30, 31, 32, 255], "labels": [-100, -100, 30, 31, 32, 255]},
        ]

    # Tiny model matching tokenizer vocab for MARSTrainer smoke test
    tiny_cfg = A2DHuginnConfig(
        n_embd=32,
        n_heads=2,
        n_layers=4,
        n_layers_in_prelude=1,
        n_layers_in_recurrent_block=1,
        n_layers_in_coda=1,
        mean_recurrence=4,
        mean_backprop_depth=2,
        intermediate_size=64,
        vocab_size=vsz,
        padding_multiple=64,
        block_size=64,
        tie_embeddings=True,
    )
    tiny_model = A2DHuginnForCausalLM(tiny_cfg)
    tiny_model.resize_token_embeddings(len(tok), pad_to_multiple_of=64)
    tiny_model.ttt_prob = 0.5
    tiny_model.block_size = 8
    tiny_model.mask_token_id = int(tok.mask_token_id)

    collator = dllm.core.trainers.bd3lm.AppendEOSBlockWrapper(
        transformers.DataCollatorForSeq2Seq(tok, return_tensors="pt", padding=True),
        block_size=8,
    )
    batch = collator(sample_feats)
    assert batch["input_ids"].shape[1] % 8 == 0, f"Expected multiple of 8, got {batch['input_ids'].shape}"

    with tempfile.TemporaryDirectory() as tmpd2:
        targs = dllm.utils.TrainingArguments(
            output_dir=tmpd2,
            per_device_train_batch_size=2,
            eval_strategy="no",
            use_cpu=True,
            report_to="none",
        )
        mtrainer = MARSTrainer(
            model=tiny_model,
            processing_class=tok,
            args=targs,
            block_size=8,
            right_shift_logits=True,
            ar_weight=1.0,
            noisy_weight=1.0,
            data_collator=collator,
        )
        loss_m = mtrainer.compute_loss(tiny_model, batch)
        loss_m.backward()
        assert torch.isfinite(loss_m), "MARSTrainer loss non-finite"
        print(f"[test 6] End-to-end MARSTrainer.compute_loss + backward PASSED! loss={loss_m.item():.4f}")


if __name__ == "__main__":
    test_all()
