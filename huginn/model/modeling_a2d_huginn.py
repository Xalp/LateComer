"""A2D / MARS + Late-Entry wrapper for Huginn-0125 (RavenForCausalLM).

Supports both:
  1. Stage 1 Standard AR SFT (causal SDPA, fixed T=32 with 24 no-grad + 8 grad steps,
     proper HuggingFace shifted CausalLM next-token cross-entropy loss).
  2. Stage 2 MARS + Late-Entry SFT ([noisy | clean] dual-stream 4D boolean SDPA mask,
     multi-batch 2D position_ids RoPE indexing, <|mask|> vocabulary padding with
     synchronized vocab_size & padded_vocab_size, and Prelude+Core Late-Entry block
     injection matching Huginn's inference-time block decoder bit-for-bit).
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import bias as attn_bias
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from torch.utils.checkpoint import checkpoint
import transformers

from .raven_config_minimal import RavenConfig
from .raven_modeling_minimal import (
    RavenForCausalLM,
    CausalLMOutputRecurrentLatents,
    HuginnDynamicCache,
    ValidCache,
    apply_rotary_emb_complex_like,
)


class A2DHuginnConfig(RavenConfig):
    model_type = "a2d-huginn"

    def __init__(self, **kwargs):
        # Prevent TypeError when reloading a saved config.json that contains tie_word_embeddings
        tie_emb = kwargs.get("tie_embeddings", kwargs.get("tie_word_embeddings", True))
        kwargs["tie_embeddings"] = tie_emb
        kwargs.pop("tie_word_embeddings", None)

        # Keep vocab_size and padded_vocab_size strictly synchronized across resize_token_embeddings
        padded_v = kwargs.pop("padded_vocab_size", None)
        if padded_v is not None:
            kwargs["vocab_size"] = max(int(kwargs.get("vocab_size", padded_v)), int(padded_v))

        super().__init__(**kwargs)
        self.padded_vocab_size = self.vocab_size
        if not getattr(self, "_attn_implementation", None) or self._attn_implementation == "eager":
            self._attn_implementation = "sdpa"


class A2DHuginnForCausalLM(RavenForCausalLM):
    config_class = A2DHuginnConfig

    def __init__(self, config: A2DHuginnConfig):
        config.padded_vocab_size = config.vocab_size
        super().__init__(config)
        self.total_steps: int = int(config.mean_recurrence)
        self.grad_steps: int = int(config.mean_backprop_depth)
        self.ttt_prob: float = 0.0
        self.firstt_prob: float = 0.0
        self.block_size: int = 16
        self.mask_token_id: Optional[int] = None
        self.late_k_grid: tuple[int, ...] = (2, 4, 6, 8, 12, 16, 24)
        self.grad_ckpt_core: bool = False

    def set_input_embeddings(self, value: nn.Module):
        self.transformer.wte = value

    def set_output_embeddings(self, new_embeddings: nn.Module):
        self.lm_head = new_embeddings

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        old_emb = self.get_input_embeddings()
        try:
            resized = self._get_resized_embeddings(
                old_emb,
                new_num_tokens=new_num_tokens,
                pad_to_multiple_of=pad_to_multiple_of,
                mean_resizing=mean_resizing,
            )
        except TypeError:
            resized = self._get_resized_embeddings(
                old_emb,
                new_num_tokens=new_num_tokens,
                pad_to_multiple_of=pad_to_multiple_of,
            )
        w_clean = resized.weight.detach().clone()
        new_v, emb_dim = int(w_clean.shape[0]), int(w_clean.shape[1])
        new_emb = nn.Embedding(
            new_v,
            emb_dim,
            padding_idx=old_emb.padding_idx,
            device=w_clean.device,
            dtype=w_clean.dtype,
        )
        with torch.no_grad():
            new_emb.weight.copy_(w_clean)
        self.set_input_embeddings(new_emb)
        new_head = nn.Linear(
            self.config.n_embd,
            new_v,
            bias=False,
            device=new_emb.weight.device,
            dtype=new_emb.weight.dtype,
        )
        if getattr(self.config, "tie_embeddings", True):
            new_head.weight = new_emb.weight
        else:
            old_head = self.get_output_embeddings()
            with torch.no_grad():
                n_copy = min(old_head.weight.shape[0], new_v)
                new_head.weight[:n_copy].copy_(old_head.weight[:n_copy])
        self.set_output_embeddings(new_head)
        self.config.vocab_size = new_v
        self.config.padded_vocab_size = new_v
        return new_emb

    def _select_freqs_cis(
        self,
        seq_len: int,
        position_ids: Optional[torch.Tensor],
        cache_position: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if position_ids is None and cache_position is None:
            return self.freqs_cis[:, :seq_len]
        if position_ids is not None:
            if position_ids.ndim == 1:
                return self.freqs_cis[:, position_ids]
            if position_ids.ndim == 2:
                # (B, S) -> (B, S, 1, hd/2, 2); broadcasts cleanly with (B, S, H, hd/2, 2) for any B >= 1
                return self.freqs_cis[0, position_ids]
            raise ValueError(f"Unsupported position_ids shape: {position_ids.shape}")
        return self.freqs_cis[:, cache_position]

    def _attn_forward(
        self,
        attn_mod,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        block_idx: torch.Tensor,
        attn_mask,
        past_key_values: Optional[ValidCache] = None,
    ) -> torch.Tensor:
        B, S, E = x.shape
        q, k, v = attn_mod.Wqkv(x).split(attn_mod.chunks, dim=2)
        q = q.view(B, S, attn_mod.n_head, attn_mod.head_dim)
        k = k.view(B, S, attn_mod.n_kv_heads, attn_mod.head_dim)
        v = v.view(B, S, attn_mod.n_kv_heads, attn_mod.head_dim)
        if attn_mod.config.qk_bias:
            q_bias, k_bias = attn_mod.qk_bias.split(1, dim=0)
            q, k = (q + q_bias).to(q.dtype), (k + k_bias).to(k.dtype)
        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)

        q = q.transpose(1, 2)  # (B, nh, S, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, block_idx)

        if isinstance(attn_mask, BlockMask):
            y = flex_attention(q, k, v, block_mask=attn_mask)
        elif isinstance(attn_mask, torch.Tensor):
            # 4D boolean SDPA mask (True = attend allowed), e.g. MARS [1, 1, 2L, 2L]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        else:
            if q.shape[2] < k.shape[2]:
                if q.shape[2] > 1:
                    bias = attn_bias.causal_lower_right(q.shape[2], k.shape[2])
                    y = F.scaled_dot_product_attention(q, k, v, bias, dropout_p=0.0)
                else:
                    y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            else:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)

        y = y.transpose(1, 2).reshape(B, S, E).contiguous()
        return attn_mod.proj(y)

    def _sandwich_forward(
        self,
        block,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        step_idx: torch.Tensor,
        attn_mask,
        past_key_values: Optional[ValidCache] = None,
    ) -> torch.Tensor:
        attn_out = self._attn_forward(block.attn, block.norm_1(x), freqs_cis, step_idx, attn_mask, past_key_values)
        x = block.norm_2(attn_out + x)
        x = block.norm_4(block.mlp(block.norm_3(x)) + x)
        return x

    def _core_step_forward(
        self,
        x: torch.Tensor,
        input_embeds: torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask,
        past_key_values: Optional[ValidCache],
        block_idx: torch.Tensor,
        current_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_idx = block_idx.detach().clone()
        x = self._maybe_inject_noise(x, current_step)
        x = self.transformer.adapter(torch.cat([x, input_embeds.to(device=x.device, dtype=x.dtype)], dim=-1))
        for block in self.transformer.core_block:
            block_idx = block_idx + 1
            x = self._sandwich_forward(block, x, freqs_cis, block_idx, attn_mask, past_key_values)
        return x, block_idx

    def _sample_budget_k(self, T: int) -> int:
        if T <= 1:
            return 1
        valid_grid = [int(k) for k in self.late_k_grid if 1 <= int(k) < T]
        if valid_grid and float(torch.rand(1).item()) < 0.7:
            idx = int(torch.randint(0, len(valid_grid), (1,)).item())
            return valid_grid[idx]
        return int(torch.randint(1, T, (1,)).item())

    def prepare_block_plan(
        self,
        concat_input_ids: torch.Tensor,
        T: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Build per-block Late-Entry (or symmetric Early-Exit) schedule on [noisy | clean].

        In MARS Stage 2, concat_input_ids has shape (B, 2L) = [noisy(0..L-1) | clean(L..2L-1)].
        For each noisy block [s0..e) containing supervised (<|mask|>) positions:
          - Late Entry (prob = self.ttt_prob):
              1) Sample step budget K < T and set entry_step = T - K (0-indexed in 1..T-1).
              2) Sample verified candidate prefix length j in 0..(e - s0) and fill noisy[s0..s0+j)
                 with clean[s0..s0+j) BEFORE Prelude runs, matching inference where a Late-Entry
                 block runs Prelude on [anchor + proposals | <|mask|> placeholders], skips Core
                 steps 0..T-K-1, initializes s_0 at Core step T-K, and executes the final K Core
                 steps over full T-step clean context KV.
          - Symmetric Early Exit (prob = self.firstt_prob, default 0.0):
              Sample step budget K < T and capture latent state after K steps (steps 0..K-1).
        """
        B, T2 = concat_input_ids.shape
        if T2 % 2 != 0 or T <= 1:
            return concat_input_ids, None, None
        L = T2 // 2
        bs = max(int(getattr(self, "block_size", 16)), 1)
        mask_id = getattr(self, "mask_token_id", None)
        p_late = float(getattr(self, "ttt_prob", 0.0) or 0.0)
        p_early = float(getattr(self, "firstt_prob", 0.0) or 0.0)
        if p_late <= 0.0 and p_early <= 0.0:
            return concat_input_ids, None, None

        noisy = concat_input_ids[:, :L]
        clean = concat_input_ids[:, L:]
        mod_ids = concat_input_ids.clone()
        entry_step = torch.zeros((B, T2), dtype=torch.long, device=concat_input_ids.device)
        exit_step = torch.zeros((B, T2), dtype=torch.long, device=concat_input_ids.device)
        any_late = False
        any_early = False

        for b in range(B):
            for s0 in range(0, L, bs):
                e = min(s0 + bs, L)
                if mask_id is not None and not bool((noisy[b, s0:e] == mask_id).any()):
                    continue
                u = float(torch.rand(1).item())
                if u < p_late:
                    K = self._sample_budget_k(T)
                    t_entry = T - K  # 1 <= t_entry <= T - 1
                    j = int(torch.randint(0, (e - s0) + 1, (1,)).item())
                    if j > 0:
                        mod_ids[b, s0 : s0 + j] = clean[b, s0 : s0 + j]
                    entry_step[b, s0:e] = t_entry
                    any_late = True
                elif u < p_late + p_early:
                    K = self._sample_budget_k(T)
                    exit_step[b, s0:e] = K  # 1 <= K <= T - 1
                    any_early = True

        return (
            mod_ids,
            entry_step if any_late else None,
            exit_step if any_early else None,
        )

    # Explicit forward signature required so HuggingFace Trainer keeps input_ids/labels/attention_mask
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        input_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_steps: Optional[torch.Tensor] = None,
        past_key_values: Optional[ValidCache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        init_scale: float = 1.0,
        **kwargs,
    ) -> CausalLMOutputRecurrentLatents:
        if input_embeds is None and inputs_embeds is not None:
            input_embeds = inputs_embeds

        # Determine recurrence step counts (deterministic T=32 with 24 no-grad + 8 grad in training)
        if num_steps is not None:
            is_pair = (
                isinstance(num_steps, (tuple, list))
                or (isinstance(num_steps, torch.Tensor) and num_steps.ndim >= 1)
            ) and len(num_steps) > 1
            if is_pair:
                num_steps_no_grad = int(num_steps[0])
                num_steps_with_grad = int(num_steps[1])
            else:
                tot = int(num_steps.item() if isinstance(num_steps, torch.Tensor) else num_steps)
                if self.training:
                    g = min(tot, int(getattr(self, "grad_steps", self.config.mean_backprop_depth)))
                    num_steps_no_grad = max(tot - g, 0)
                    num_steps_with_grad = g
                else:
                    num_steps_no_grad = tot
                    num_steps_with_grad = 0
        elif self.training:
            tot = int(getattr(self, "total_steps", self.config.mean_recurrence))
            g = min(tot, int(getattr(self, "grad_steps", self.config.mean_backprop_depth)))
            num_steps_no_grad = max(tot - g, 0)
            num_steps_with_grad = g
        else:
            num_steps_no_grad = int(self.config.mean_recurrence)
            num_steps_with_grad = 0

        total_T = num_steps_no_grad + num_steps_with_grad

        prepared_attn_mask = None
        if isinstance(attention_mask, BlockMask):
            prepared_attn_mask = attention_mask
        elif isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4:
            prepared_attn_mask = attention_mask

        entry_step = kwargs.get("entry_step", None)
        exit_step = kwargs.get("exit_step", None)
        effective_ids = input_ids
        if (
            self.training
            and effective_ids is not None
            and input_embeds is None
            and isinstance(prepared_attn_mask, torch.Tensor)
            and entry_step is None
            and exit_step is None
            and (float(getattr(self, "ttt_prob", 0.0) or 0.0) > 0.0 or float(getattr(self, "firstt_prob", 0.0) or 0.0) > 0.0)
        ):
            effective_ids, entry_step, exit_step = self.prepare_block_plan(effective_ids, total_T)

        seq_len = effective_ids.shape[1] if effective_ids is not None else input_embeds.shape[1]
        freqs_cis = self._select_freqs_cis(seq_len, position_ids, cache_position)

        if input_embeds is None:
            input_embeds = self.transformer.wte(effective_ids)
            if self.emb_scale != 1:
                input_embeds = input_embeds * self.emb_scale

        if use_cache and past_key_values is None:
            past_key_values = HuginnDynamicCache()

        # 1. Non-recurrent Prelude (2 SandwichBlocks)
        block_idx = torch.tensor(-1, device=torch.device("cpu"), dtype=torch.long)
        for block in self.transformer.prelude:
            block_idx = block_idx + 1
            input_embeds = self._sandwich_forward(block, input_embeds, freqs_cis, block_idx, prepared_attn_mask, past_key_values)

        # 2. Recurrent Core (total_T steps = num_steps_no_grad + num_steps_with_grad)
        x = self.initialize_state(input_embeds, scale=init_scale) if input_states is None else input_states.clone()
        exit_saved: Optional[torch.Tensor] = torch.zeros_like(x) if exit_step is not None else None

        if num_steps_no_grad > 0:
            with torch.no_grad():
                emb_ng = input_embeds.detach()
                for step in range(num_steps_no_grad):
                    if entry_step is not None:
                        sel = entry_step == step
                        if bool(sel.any()):
                            x = torch.where(sel.unsqueeze(-1), self.initialize_state(emb_ng, scale=init_scale), x)
                    x, block_idx = self._core_step_forward(
                        x, emb_ng, freqs_cis, prepared_attn_mask, past_key_values, block_idx, step
                    )
                    if exit_step is not None and exit_saved is not None:
                        esel = exit_step == (step + 1)
                        if bool(esel.any()):
                            exit_saved = torch.where(esel.unsqueeze(-1), x, exit_saved)

        if num_steps_no_grad > 0 and num_steps_with_grad > 0:
            x = x.detach()
            if exit_saved is not None:
                exit_saved = exit_saved.detach()

        use_ckpt = bool(self.gradient_checkpointing or getattr(self, "grad_ckpt_core", False))
        for g_idx in range(num_steps_with_grad):
            step = num_steps_no_grad + g_idx
            if entry_step is not None:
                sel = entry_step == step
                if bool(sel.any()):
                    x0_fresh = self.initialize_state(input_embeds, scale=init_scale).detach()
                    x = torch.where(sel.unsqueeze(-1), x0_fresh, x)
            if use_ckpt:
                x, block_idx = checkpoint(
                    self._core_step_forward,
                    x,
                    input_embeds,
                    freqs_cis,
                    prepared_attn_mask,
                    past_key_values,
                    block_idx,
                    step,
                    use_reentrant=False,
                    preserve_rng_state=False,
                    determinism_check="none",
                )
            else:
                x, block_idx = self._core_step_forward(
                    x, input_embeds, freqs_cis, prepared_attn_mask, past_key_values, block_idx, step
                )
            if exit_step is not None and exit_saved is not None:
                esel = exit_step == (step + 1)
                if bool(esel.any()):
                    exit_saved = torch.where(esel.unsqueeze(-1), x, exit_saved)

        if exit_step is not None and exit_saved is not None:
            has_exit = exit_step > 0
            if bool(has_exit.any()):
                x = torch.where(has_exit.unsqueeze(-1), exit_saved, x)

        latent_states = x.detach()
        x = self.transformer.ln_f(x)

        # 3. Non-recurrent Coda (2 SandwichBlocks) + Final RMSNorm + LM Head
        block_idx = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in self.transformer.coda:
            block_idx = block_idx - 1
            x = self._sandwich_forward(block, x, freqs_cis, block_idx, prepared_attn_mask, past_key_values)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x).float()
        loss = None
        log_ppl = None
        if labels is not None:
            # Standard HuggingFace CausalLM next-token shift (logits[..., :-1, :] vs labels[..., 1:])
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            log_ppl = loss.detach()

        return CausalLMOutputRecurrentLatents(
            loss=loss,
            log_ppl=log_ppl,
            logits=logits,
            past_key_values=past_key_values,
            latent_states=latent_states,
        )


transformers.AutoConfig.register("a2d-huginn", A2DHuginnConfig)
transformers.AutoModel.register(A2DHuginnConfig, A2DHuginnForCausalLM)
transformers.AutoModelForCausalLM.register(A2DHuginnConfig, A2DHuginnForCausalLM)
transformers.AutoModelForMaskedLM.register(A2DHuginnConfig, A2DHuginnForCausalLM)
