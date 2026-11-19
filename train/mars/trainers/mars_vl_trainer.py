"""
MARS trainer for Qwen2.5-VL (multimodal agentic trajectories, e.g. DeepEyesV2).

Same objective as ``MARSTrainer``:
    Loss = CE(noisy_logits, labels) + ar_weight * CE(clean_logits, labels)
over a ``[noisy | clean]`` dual-stream concatenation with the MARS structured
attention mask. The multimodal differences are confined to this class:

  * **Vision runs once.** Images live in the (never-masked) prompt/observation
    region, so both streams share identical vision features. We embed each
    stream's token ids, scatter the vision features into the image-pad
    positions of each stream, and concatenate *embeddings* — the vision tower
    is not run twice and never sees mask tokens.
  * **3D mRoPE.** Position ids come from ``get_rope_index`` on the single
    clean stream and are duplicated for the noisy stream (logical positions
    are stream-local, exactly like the 1D case in ``MARSTrainer``).
  * **Padding-aware mask.** Unlike the text trainer (uniform lengths after
    ``group_by_length``), VLM batches mix very different sequence lengths, so
    the per-batch boolean mask folds in key-side padding.

The forward goes straight through ``model.model.language_model`` (the A2D text
decoder that accepts a prebuilt 4D mask) + ``model.lm_head``; the surrounding
``Qwen2_5_VLModel`` plumbing is bypassed because vision scatter is done here.
"""

from functools import partial
from types import SimpleNamespace

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
import transformers

from .attention_mask import mars_attention_mask
from .mars_trainer import MARSTrainer


class MARSVLTrainer(MARSTrainer):

    def __init__(self, *args, mars_noisy_chunks: int = 1, **kwargs):
        # mars_noisy_chunks (k): the dataset is expanded x k; each row trains
        # ONE block-aligned noisy chunk as [noisy_chunk | full clean] (peak
        # forward ~= (1 + 1/k)*L instead of 2L), a normal single forward/backward
        # so DeepSpeed treats a trajectory's k chunks as ordinary micro-steps.
        # k=1 keeps the original 2L forward. Mathematically identical (MARS mask
        # rule 2/3: a noisy token depends only on same-block noisy + earlier
        # clean), verified to fp32 machine precision.
        super().__init__(*args, **kwargs)
        self.mars_noisy_chunks = int(mars_noisy_chunks)

    def _vision_embeds(self, model, stream_ids, pixel_values, image_grid_thw):
        """Embed one stream's token ids and scatter vision features into the
        image-pad positions. ``stream_ids``: [b, l]."""
        base = model.model.language_model.embed_tokens(stream_ids)
        if pixel_values is None:
            return base
        image_embeds = torch.cat(
            model.model.get_image_features(pixel_values, image_grid_thw), dim=0
        )
        image_mask = (stream_ids == model.config.image_token_id).unsqueeze(-1)
        return base.masked_scatter(image_mask, image_embeds.to(base.dtype))

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
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)

        assert input_ids.shape == labels.shape, (
            "input_ids/labels length mismatch; the MM collator must keep them aligned "
            "(truncation here could cut through image-pad spans)"
        )
        b, l = input_ids.shape
        token_cnt_per_seq = torch.sum(labels != -100, dim=1, keepdim=True)
        unwrapped = self.accelerator.unwrap_model(model)

        # === 1. Full masking of supervised (assistant) tokens ===
        masked_indices = (labels != -100)

        # Chunked-noisy path (memory): this row is ONE block-aligned noisy chunk
        # of a trajectory (the dataset is expanded x k with a per-row chunk id).
        # We forward only [noisy_chunk | full clean] -> peak ~ (1 + 1/k)*L, as a
        # normal single forward/backward, so DeepSpeed treats the k chunks of a
        # trajectory as ordinary gradient-accumulation micro-steps (no checkpoint,
        # no double grad reduction). Mathematically identical to the 2L forward.
        if "mars_chunk_id" in inputs and not return_outputs:
            return self._single_chunk_loss(
                unwrapped, model, input_ids, labels, attention_mask, pixel_values,
                image_grid_thw, masked_indices, token_cnt_per_seq, b, l,
                int(inputs["mars_chunk_id"].reshape(-1)[0]),
                int(inputs["mars_chunk_k"].reshape(-1)[0]),
            )

        noised_input_ids = torch.where(
            masked_indices, self.processing_class.mask_token_id, input_ids
        )

        # === 2. Dual-stream embeddings: [noisy | clean], vision scattered into both ===
        noisy_embeds = self._vision_embeds(unwrapped, noised_input_ids, pixel_values, image_grid_thw)
        clean_embeds = self._vision_embeds(unwrapped, input_ids, pixel_values, image_grid_thw)
        concat_embeds = torch.cat([noisy_embeds, clean_embeds], dim=1)

        # === 3. MARS attention mask (+ key-side padding), 3D mRoPE positions ===
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if unwrapped.config._attn_implementation == "sdpa":
            mars_mask = mars_attention_mask(
                b=None, h=None,
                q_idx=torch.arange(l * 2)[:, None],
                kv_idx=torch.arange(l * 2)[None, :],
                block_size=self.block_size, n=l,
            ).to(input_ids.device)
            pad_kv = torch.cat([attention_mask, attention_mask], dim=1).bool()  # [b, 2l]
            full_mask = mars_mask.unsqueeze(0) & pad_kv[:, None, None, :]  # [b, 1, 2l, 2l]
        else:
            raise NotImplementedError(
                "MARSVLTrainer currently supports attn_implementation='sdpa' only"
            )

        position_ids, _ = unwrapped.model.get_rope_index(
            input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask
        )  # [3, b, l]
        concat_position_ids = torch.cat([position_ids, position_ids], dim=-1)  # [3, b, 2l]

        # === 4. Forward through the A2D text decoder + lm_head ===
        hidden = model.model.language_model(
            inputs_embeds=concat_embeds,
            attention_mask=full_mask,
            position_ids=concat_position_ids,
            use_cache=False,
        ).last_hidden_state
        logits = unwrapped.lm_head(hidden)

        outputs = self._postprocess_outputs(SimpleNamespace(logits=logits))
        full_logits = outputs.logits
        noisy_logits = full_logits[:, :l]
        clean_logits = full_logits[:, l:]

        # === 5. Degenerate case ===
        if not masked_indices.any():
            self.epoch_meter.update(
                split="train" if model.training else "eval",
                nll_sum=noisy_logits.sum() * 0.0,
                token_cnt=token_cnt_per_seq.sum(),
            )
            zero = noisy_logits.sum() * 0.0
            return (zero, outputs) if return_outputs else zero

        # === 6. Fix input/label mismatch (e.g. EOS vs IM_END), as in MARSTrainer ===
        mismatch = (input_ids[masked_indices] != labels[masked_indices])
        if mismatch.any():
            input_ids[masked_indices] = labels[masked_indices]

        # === 7. Noisy-stream CE + auxiliary AR (clean-stream) CE ===
        noisy_ce = F.cross_entropy(
            noisy_logits[masked_indices], labels[masked_indices], reduction="none"
        )
        clean_ce = F.cross_entropy(
            clean_logits[masked_indices], labels[masked_indices], reduction="none"
        )
        token_loss = noisy_ce + self.ar_weight * clean_ce

        if self.loss_normalization_type == "batch":
            token_loss_normalized = token_loss / b
        elif self.loss_normalization_type == "sequence":
            token_loss_normalized = (
                token_loss / token_cnt_per_seq.expand(-1, l)[masked_indices] / b
            )
        elif self.loss_normalization_type == "token":
            token_loss_normalized = token_loss / token_cnt_per_seq.sum()
        else:
            raise ValueError("Invalid loss_normalization_type.")
        loss = token_loss_normalized.sum()

        self.epoch_meter.update(
            split="train" if model.training else "eval",
            nll_sum=noisy_ce.sum(),
            token_cnt=token_cnt_per_seq.sum(),
        )

        outputs.logits = noisy_logits
        return (loss, outputs) if return_outputs else loss


    def _single_chunk_loss(
        self, unwrapped, model, input_ids, labels, attention_mask, pixel_values,
        image_grid_thw, masked_indices, token_cnt_per_seq, b, l, chunk_id, k,
    ):
        """Loss for ONE block-aligned noisy chunk vs the full clean stream.

        Forward = [noisy_chunk (blocks of this chunk, +1 prepended block for the
        right-shift boundary) | full clean]. A single normal forward/backward;
        the k chunks of a trajectory are separate dataset rows, so DeepSpeed sees
        them as ordinary micro-steps. Loss/normalization identical to the full 2L
        forward summed over chunks (verified to fp32 machine precision).
        """
        from .attention_mask import mars_chunk_mask

        shift = bool(self.right_shift_logits)
        block_size = self.block_size
        mask_id = self.processing_class.mask_token_id
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        n_blocks = (l + block_size - 1) // block_size
        blocks_per_chunk = (n_blocks + k - 1) // k
        b0 = chunk_id * blocks_per_chunk
        b1 = min((chunk_id + 1) * blocks_per_chunk, n_blocks)
        # Clamp to a valid non-empty 1-block range. NOTE: we deliberately do NOT
        # short-circuit empty/degenerate chunks -- every rank must run the same
        # full forward so DeepSpeed's per-param gradient reductions stay in lockstep
        # across ranks (a shortcut path desyncs NCCL -> collective timeout). Empty
        # chunks just yield zero loss (connected via +0*logits below). With k=2 on
        # real trajectories (hundreds of blocks) chunks are never empty anyway.
        if b1 <= b0:
            b0 = max(0, min(b0, n_blocks - 1))
            b1 = b0 + 1
        p0 = b0 * block_size
        p1 = min(b1 * block_size, l)

        pe = max(0, p0 - block_size) if shift else p0
        noised_input_ids = torch.where(masked_indices, mask_id, input_ids)
        position_ids, _ = unwrapped.model.get_rope_index(
            input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask
        )  # [3, b, l]

        noisy_e = self._vision_embeds(unwrapped, noised_input_ids, pixel_values, image_grid_thw)[:, pe:p1]
        clean_e = self._vision_embeds(unwrapped, input_ids, pixel_values, image_grid_thw)
        concat = torch.cat([noisy_e, clean_e], dim=1)
        m_ = p1 - pe
        npos = torch.arange(pe, p1, device=input_ids.device)
        mask2d = mars_chunk_mask(npos, l, block_size).to(input_ids.device)
        pad_kv = torch.cat([attention_mask[:, pe:p1], attention_mask], dim=1).bool()
        full_mask = mask2d.unsqueeze(0) & pad_kv[:, None, None, :]
        cpos = torch.cat([position_ids[..., pe:p1], position_ids], dim=-1).contiguous()

        hidden = model.model.language_model(
            inputs_embeds=concat, attention_mask=full_mask,
            position_ids=cpos, use_cache=False,
        ).last_hidden_state
        # Run lm_head ONLY on the hidden states the loss actually needs (the
        # chunk's noisy + clean loss positions), not the whole [noisy|clean]
        # stream. The full-vocab logits over the entire clean stream were a fixed
        # ~10GB (vocab 152k x fp32) regardless of chunk size -- the real OOM, not
        # the noisy activations. Gathering first keeps it ~2x(p1-p0) positions.
        noisy_hidden = hidden[:, :m_]   # [b, m_, H]
        clean_hidden = hidden[:, m_:]   # [b, l, H]
        loss_pos = torch.arange(p0, p1, device=input_ids.device)
        if shift:
            nsrc = (loss_pos - 1).clamp(min=0) - pe   # into noisy_hidden
            csrc = (loss_pos - 1).clamp(min=0)        # into clean_hidden
        else:
            nsrc = loss_pos - pe
            csrc = loss_pos
        nl_loss = unwrapped.lm_head(noisy_hidden[:, nsrc]).float()   # [b, p1-p0, V]
        cl_loss = unwrapped.lm_head(clean_hidden[:, csrc]).float()   # [b, p1-p0, V]

        chunk_labels = labels[:, p0:p1]
        cmask = masked_indices[:, p0:p1]
        nce = F.cross_entropy(nl_loss[cmask], chunk_labels[cmask], reduction="none")
        cce = F.cross_entropy(cl_loss[cmask], chunk_labels[cmask], reduction="none")
        tl = nce + self.ar_weight * cce
        if self.loss_normalization_type == "batch":
            tln = tl / b
        elif self.loss_normalization_type == "sequence":
            tln = tl / token_cnt_per_seq.expand(-1, p1 - p0)[cmask] / b
        elif self.loss_normalization_type == "token":
            tln = tl / token_cnt_per_seq.sum()
        else:
            raise ValueError("Invalid loss_normalization_type.")
        # +0*logits keeps the loss connected to the full forward (all params) even
        # when this chunk has no supervised tokens, so every rank produces grads
        # for the same params -> DeepSpeed reductions stay synchronized.
        loss = tln.sum() + (nl_loss.sum() + cl_loss.sum()) * 0.0
        self.epoch_meter.update(
            split="train", nll_sum=nce.sum().detach(), token_cnt=token_cnt_per_seq.sum(),
        )
        return loss
