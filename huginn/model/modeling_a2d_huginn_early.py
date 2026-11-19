"""Strictly symmetric Early-Exit Stage-2 wrapper for Huginn-0125 (RavenForCausalLM).

Shares the exact parameter state_dict and A2DHuginnConfig as A2DHuginnForCausalLM, while:
  1. Sampling early-exit block budgets K < T, verified candidate prefix lengths j, and
     block positions with the exact same RNG calls and distribution as Late Entry.
  2. Giving every Early-Exit block (even K <= 24 when T=32, grad_steps=8) its full
     min(K, grad_steps)-step Core backprop window [max(0, K - grad_steps), K) plus
     end-to-end Prelude, wte (<|mask|>), and Coda gradients via per-step activation
     checkpointing.
"""

from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask
from torch.utils.checkpoint import checkpoint

from .modeling_a2d_huginn import A2DHuginnConfig, A2DHuginnForCausalLM
from .raven_modeling_minimal import CausalLMOutputRecurrentLatents, HuginnDynamicCache, ValidCache


class A2DHuginnEarlyExitForCausalLM(A2DHuginnForCausalLM):
    config_class = A2DHuginnConfig

    def prepare_early_block_plan(
        self,
        concat_input_ids: torch.Tensor,
        T: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Symmetric counterpart to Late-Entry prepare_block_plan.

        On each response block [s0..e) containing <|mask|>, with probability firstt_prob:
          - Samples K = _sample_budget_k(T) using identical grid/distribution.
          - Samples verified candidate prefix length j in 0..(e - s0) and fills
            noisy[s0..s0+j) with clean[s0..s0+j) before Prelude (identical to Late Entry).
          - Sets exit_step[b, s0:e] = K so tokens execute the FIRST K Core steps (0..K-1).
        """
        B, T2 = concat_input_ids.shape
        if T2 % 2 != 0 or T <= 1:
            return concat_input_ids, None
        L = T2 // 2
        bs = max(int(getattr(self, "block_size", 16)), 1)
        mask_id = getattr(self, "mask_token_id", None)
        p_early = float(getattr(self, "firstt_prob", 0.5) or 0.0)
        if p_early <= 0.0:
            return concat_input_ids, None

        noisy = concat_input_ids[:, :L]
        clean = concat_input_ids[:, L:]
        mod_ids = concat_input_ids.clone()
        exit_step = torch.zeros((B, T2), dtype=torch.long, device=concat_input_ids.device)
        any_early = False

        for b in range(B):
            for s0 in range(0, L, bs):
                e = min(s0 + bs, L)
                if mask_id is not None and not bool((noisy[b, s0:e] == mask_id).any()):
                    continue
                u = float(torch.rand(1).item())
                if u < p_early:
                    K = self._sample_budget_k(T)
                    j = int(torch.randint(0, (e - s0) + 1, (1,)).item())
                    if j > 0:
                        mod_ids[b, s0 : s0 + j] = clean[b, s0 : s0 + j]
                    exit_step[b, s0:e] = K  # 1 <= K <= T - 1
                    any_early = True

        return mod_ids, (exit_step if any_early else None)

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
        if not self.training:
            return super().forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                input_embeds=input_embeds,
                input_states=input_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                num_steps=num_steps,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                init_scale=init_scale,
                **kwargs,
            )

        if input_embeds is None and inputs_embeds is not None:
            input_embeds = inputs_embeds

        tot = int(getattr(self, "total_steps", self.config.mean_recurrence))
        g = min(tot, int(getattr(self, "grad_steps", self.config.mean_backprop_depth)))
        total_T = tot

        prepared_attn_mask = None
        if isinstance(attention_mask, BlockMask):
            prepared_attn_mask = attention_mask
        elif isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4:
            prepared_attn_mask = attention_mask

        exit_step = kwargs.get("exit_step", None)
        effective_ids = input_ids
        if (
            effective_ids is not None
            and input_embeds is None
            and isinstance(prepared_attn_mask, torch.Tensor)
            and exit_step is None
            and float(getattr(self, "firstt_prob", 0.0) or 0.0) > 0.0
        ):
            effective_ids, exit_step = self.prepare_early_block_plan(effective_ids, total_T)

        seq_len = effective_ids.shape[1] if effective_ids is not None else input_embeds.shape[1]
        freqs_cis = self._select_freqs_cis(seq_len, position_ids, cache_position)

        if input_embeds is None:
            input_embeds = self.transformer.wte(effective_ids)
            if self.emb_scale != 1:
                input_embeds = input_embeds * self.emb_scale

        if use_cache and past_key_values is None:
            past_key_values = HuginnDynamicCache()

        # 1. Non-recurrent Prelude (runs with grad so wte + Prelude adapt to K-step readout)
        block_idx = torch.tensor(-1, device=torch.device("cpu"), dtype=torch.long)
        for block in self.transformer.prelude:
            block_idx = block_idx + 1
            input_embeds = self._sandwich_forward(
                block, input_embeds, freqs_cis, block_idx, prepared_attn_mask, past_key_values
            )

        # 2. Recurrent Core with per-position g-step truncated-BPTT window [target_end - g, target_end)
        x = self.initialize_state(input_embeds, scale=init_scale) if input_states is None else input_states.clone()
        B_sz = x.shape[0]
        if exit_step is not None:
            target_end = torch.where(exit_step > 0, exit_step, torch.full_like(exit_step, total_T))
            exit_saved: Optional[torch.Tensor] = torch.zeros_like(x)
        else:
            target_end = torch.full((B_sz, seq_len), total_T, dtype=torch.long, device=x.device)
            exit_saved = None
        grad_start = (target_end - g).clamp(min=0)

        for step in range(total_T):
            in_win = (step >= grad_start) & (step < target_end)
            if not bool(in_win.any()):
                with torch.no_grad():
                    x, block_idx = self._core_step_forward(
                        x.detach(),
                        input_embeds.detach(),
                        freqs_cis,
                        prepared_attn_mask,
                        past_key_values,
                        block_idx,
                        step,
                    )
            else:
                w3 = in_win.unsqueeze(-1)
                x_in = torch.where(w3, x, x.detach())
                emb_in = torch.where(w3, input_embeds, input_embeds.detach())
                x, block_idx = checkpoint(
                    self._core_step_forward,
                    x_in,
                    emb_in,
                    freqs_cis,
                    prepared_attn_mask,
                    past_key_values,
                    block_idx,
                    step,
                    use_reentrant=False,
                    preserve_rng_state=False,
                    determinism_check="none",
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

        # 3. Non-recurrent Coda + Final RMSNorm + LM Head
        block_idx = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in self.transformer.coda:
            block_idx = block_idx - 1
            x = self._sandwich_forward(block, x, freqs_cis, block_idx, prepared_attn_mask, past_key_values)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x).float()
        loss = None
        log_ppl = None
        if labels is not None:
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
