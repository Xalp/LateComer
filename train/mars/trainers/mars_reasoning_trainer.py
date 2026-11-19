"""
MARSReasoningTrainer: MARS dual-stream training where the multi-token (noisy)
denoising objective is applied ONLY to the *reasoning* region (tokens inside
``<think>...</think>``), while the auxiliary AR (clean) loss covers ALL
supervised tokens.

Rationale (reasoning/output split): MARS multi-token decoding accelerates the
monotonous reasoning trace well, but the final answer/output should be produced
one token at a time (pure AR) to preserve correctness. We therefore train the
model so that:

  * reasoning tokens (<think>...</think>): noised + MARS noisy-CE + AR clean-CE
  * output/answer tokens:                  never noised, AR clean-CE only

At decode time the policy is: multi-token (tau<1) while inside <think>, switch
to AR (tau=1) after </think>.

The reasoning region is recovered from the token ids (single-token Qwen3 tags
``<think>``=151667, ``</think>``=151668), so no extra collated field is needed.

Loss = reasoning_mask * CE(noisy_logits, labels) + ar_weight * CE(clean_logits, labels)
"""

import torch
import torch.nn.functional as F
import transformers
from functools import partial
from torch.nn.attention.flex_attention import create_block_mask

from .mars_trainer import MARSTrainer
from .attention_mask import mars_attention_mask


# Qwen3 single-token thinking tags.
THINK_OPEN_ID = 151667
THINK_CLOSE_ID = 151668


def reasoning_token_mask(input_ids: torch.Tensor,
                         open_id: int = THINK_OPEN_ID,
                         close_id: int = THINK_CLOSE_ID) -> torch.Tensor:
    """(b, l) bool mask: True for tokens strictly between <think> and </think>
    (the reasoning *content*; the tags themselves are excluded). Handles multiple
    think blocks per sequence via cumulative counts."""
    opened = (input_ids == open_id).cumsum(dim=1)
    closed = (input_ids == close_id).cumsum(dim=1)
    inside = (opened - closed) > 0          # includes the <think> tag position
    inside = inside & (input_ids != open_id)  # exclude the <think> tag itself
    return inside


class MARSReasoningTrainer(MARSTrainer):
    """MARS trainer that confines the noisy/multi-token objective to the reasoning
    region; AR (clean) loss covers all supervised tokens."""

    def compute_loss(
        self,
        model: transformers.PreTrainedModel,
        inputs,
        return_outputs: bool = False,
        **kwargs,
    ):
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        min_len = min(input_ids.shape[1], labels.shape[1])
        input_ids = input_ids[:, :min_len]
        labels = labels[:, :min_len]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :min_len]
        b, l = input_ids.shape
        token_cnt_per_seq = torch.sum(labels != -100, dim=1, keepdim=True)

        masked_indices = (labels != -100)                      # all supervised tokens
        reasoning_mask = reasoning_token_mask(input_ids)       # reasoning content tokens
        reasoning_sup = masked_indices & reasoning_mask        # supervised reasoning tokens

        # === Noise ONLY supervised reasoning tokens (MARS objective is reasoning-only) ===
        noised_input_ids = torch.where(
            reasoning_sup, self.processing_class.mask_token_id, input_ids
        )

        concat_input_ids = torch.cat([noised_input_ids, input_ids], dim=1)

        if self.accelerator.unwrap_model(model).config._attn_implementation == "sdpa":
            am = mars_attention_mask(
                b=None, h=None,
                q_idx=torch.arange(l * 2)[:, None],
                kv_idx=torch.arange(l * 2)[None, :],
                block_size=self.block_size, n=l,
            )
            attention_mask = am.unsqueeze(0).unsqueeze(0).expand(1, 1, 2 * l, 2 * l).to(input_ids.device)
        elif self.accelerator.unwrap_model(model).config._attn_implementation == "flex_attention":
            attention_mask = create_block_mask(
                partial(mars_attention_mask, block_size=self.block_size, n=l),
                B=None, H=None, Q_LEN=l * 2, KV_LEN=l * 2,
            )
        else:
            raise NotImplementedError

        base_pos = torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, l)
        concat_position_ids = torch.cat([base_pos, base_pos], dim=1)

        outputs = model(
            input_ids=concat_input_ids,
            attention_mask=attention_mask,
            position_ids=concat_position_ids,
        )
        outputs = self._postprocess_outputs(outputs)
        full_logits = outputs.logits
        noisy_logits = full_logits[:, :l]
        clean_logits = full_logits[:, l:]

        # Degenerate: nothing supervised.
        if not masked_indices.any():
            self.epoch_meter.update(
                split="train" if model.training else "eval",
                nll_sum=noisy_logits.sum() * 0.0,
                token_cnt=token_cnt_per_seq.sum(),
            )
            zero = noisy_logits.sum() * 0.0
            return (zero, outputs) if return_outputs else zero

        # Fix any input/label mismatch on supervised tokens.
        mismatch = (input_ids[masked_indices] != labels[masked_indices])
        if mismatch.any():
            input_ids[masked_indices] = labels[masked_indices]

        # === Clean (AR) CE over ALL supervised tokens ===
        clean_ce = F.cross_entropy(
            clean_logits[masked_indices], labels[masked_indices], reduction="none"
        )

        # === Noisy (MARS multi-token) CE over supervised REASONING tokens only ===
        if reasoning_sup.any():
            noisy_ce = F.cross_entropy(
                noisy_logits[reasoning_sup], labels[reasoning_sup], reduction="none"
            )
            noisy_sum = noisy_ce.sum()
        else:
            noisy_ce = noisy_logits.sum() * 0.0
            noisy_sum = noisy_ce

        # === Combine + normalize (token-normalized over all supervised tokens) ===
        denom = token_cnt_per_seq.sum().clamp(min=1)
        if self.loss_normalization_type == "batch":
            loss = (self.noisy_weight * noisy_sum + self.ar_weight * clean_ce.sum()) / b
        else:  # 'token' (default for this path); 'sequence' falls back to token here
            loss = (self.noisy_weight * noisy_sum + self.ar_weight * clean_ce.sum()) / denom

        self.epoch_meter.update(
            split="train" if model.training else "eval",
            nll_sum=noisy_sum,
            token_cnt=reasoning_sup.sum().clamp(min=1),
        )

        outputs.logits = noisy_logits
        return (loss, outputs) if return_outputs else loss
