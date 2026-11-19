"""MARS CHUNK-EXTENSION trainer (adaptive, on-policy block boundaries).

Same objective as ``MARSTrainer`` -- ``loss = CE(noisy) + ar_weight * CE(clean)``
over the supervised (response) tokens -- but the MARS blocks are NOT a fixed
``block_size``. Instead, each training step:

  1. FORWARD #1 (no grad): run the standard fixed-block MARS forward and read,
     per response position, the model's top-1 confidence of producing that token
     from masked context (right-shift convention).
  2. SEGMENT: walk each response left-to-right exactly like MARS generation at
     threshold ``tau`` -- offset-0 always accepted, extend the chunk while
     ``conf >= tau`` (capped at ``block_size``). These tau-acceptance spans are
     the blocks.
  3. BUILD the variable-block noisy attention mask from those spans.
  4. FORWARD #2 (with grad): the dual-stream MARS forward with the variable mask;
     compute the usual noisy + clean loss.

Why this "pushes the decoding frontier": with right-shift, chunk k's LAST masked
position predicts the FIRST token of chunk k+1 -- i.e. the noisy loss supervises
"the +1 token just beyond what the model is currently confident about", from the
chunk's own masked context. That is exactly the position whose confidence must
rise for the chunk to grow. Training lowers its loss -> its masked-confidence
rises -> next step's segmentation extends the chunk -> mean tokens/forward (the
frontier) climbs. The "+1 overlap" with the next chunk's offset-0 is handled by
the right-shift wiring (no token is duplicated); the next chunk's offset-0 is in
turn supervised by the clean (AR) stream.

The ONLY difference vs ``MARSTrainer`` is where the block boundaries come from,
so a run is directly comparable to a fixed-``block_size`` MARS run with identical
data/hparams.
"""

import torch
import torch.nn.functional as F
import transformers

from .attention_mask import mars_attention_mask, mars_variable_block_mask
from .mars_trainer import MARSTrainer

_PAD_BLOCK = 10 ** 6  # chunks never attend padding (block_q < pad_block always)


class MARSChunkExtTrainer(MARSTrainer):
    """MARS trainer whose blocks are the model's own tau-acceptance spans.

    Args (beyond MARSTrainer):
        tau: confidence threshold for chunk extension during segmentation.
        block_size: cap on a single chunk's length AND the provisional block
            size used by the segmentation forward (matches the base model, 16).
    """

    def __init__(self, *args, tau: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau
        self._fr_sum = 0.0   # running frontier (tokens/forward) for progress logging
        self._fr_cnt = 0

    # ------------------------------------------------------------------ #
    # FORWARD #1 + segmentation -> per-position block ids.                #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _segment(self, model, noised_input_ids, input_ids, labels, l):
        """Return block_ids [b, l]: 0=prompt, 1..=tau-acceptance chunks,
        _PAD_BLOCK=padding. One fixed-block MARS forward gives the per-position
        masked-context confidence used to grow chunks."""
        b = input_ids.shape[0]
        dev = input_ids.device
        unwrapped = self.accelerator.unwrap_model(model)
        assert unwrapped.config._attn_implementation == "sdpa", \
            "MARSChunkExtTrainer requires attn_implementation='sdpa'"

        # provisional fixed-block MARS forward (block_size = self.block_size)
        attn = mars_attention_mask(
            b=None, h=None,
            q_idx=torch.arange(l * 2)[:, None],
            kv_idx=torch.arange(l * 2)[None, :],
            block_size=self.block_size, n=l,
        ).unsqueeze(0).unsqueeze(0).expand(1, 1, 2 * l, 2 * l).to(dev)
        base_pos = torch.arange(l, device=dev).unsqueeze(0).expand(b, l)
        concat_pos = torch.cat([base_pos, base_pos], dim=1)
        concat_ids = torch.cat([noised_input_ids, input_ids], dim=1)

        # use the unwrapped module for the no-grad segmentation forward: under
        # ZeRO-2 params are replicated, so this is exact and avoids invoking the
        # DeepSpeed engine's forward bookkeeping twice per step.
        out = unwrapped(input_ids=concat_ids, attention_mask=attn, position_ids=concat_pos)
        out = self._postprocess_outputs(out)            # right-shift: logits[i] predicts token i
        noisy_logits = out.logits[:, :l].float()
        conf = F.softmax(noisy_logits, dim=-1).max(dim=-1).values  # [b, l]

        resp = labels != -100                            # [b, l]
        block_ids = torch.full((b, l), _PAD_BLOCK, dtype=torch.long, device=dev)
        bs = self.block_size
        for s in range(b):
            idx = torch.nonzero(resp[s], as_tuple=False).flatten()
            if idx.numel() == 0:
                block_ids[s, :] = 0
                continue
            P = int(idx[0].item())
            end = int(idx[-1].item()) + 1
            block_ids[s, :P] = 0                         # prompt -> block 0
            cs = conf[s]
            p = P
            blk = 1
            while p < end:
                c = 1                                    # offset-0 always accepted
                while c < bs and p + c < end and cs[p + c].item() >= self.tau:
                    c += 1
                block_ids[s, p:p + c] = blk
                blk += 1
                p += c
        return block_ids

    # ------------------------------------------------------------------ #
    # FORWARD #2 + loss (identical objective to MARSTrainer).             #
    # ------------------------------------------------------------------ #
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels = inputs["input_ids"], inputs["labels"]
        min_len = min(input_ids.shape[1], labels.shape[1])
        input_ids = input_ids[:, :min_len]
        labels = labels[:, :min_len]
        b, l = input_ids.shape
        token_cnt_per_seq = torch.sum(labels != -100, dim=1, keepdim=True)

        # full masking of the supervised region (MARS convention)
        masked_indices = (labels != -100)
        noised_input_ids = torch.where(
            masked_indices, self.processing_class.mask_token_id, input_ids
        )

        # ---- FORWARD #1: on-policy tau-segmentation -> variable blocks ----
        block_ids = self._segment(model, noised_input_ids, input_ids, labels, l)

        # ---- FORWARD #2: dual-stream MARS forward with the variable mask ----
        attention_mask = mars_variable_block_mask(block_ids).to(input_ids.device)
        base_pos = torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, l)
        concat_position_ids = torch.cat([base_pos, base_pos], dim=1)
        concat_input_ids = torch.cat([noised_input_ids, input_ids], dim=1)

        outputs = model(
            input_ids=concat_input_ids,
            attention_mask=attention_mask,
            position_ids=concat_position_ids,
        )
        outputs = self._postprocess_outputs(outputs)
        full_logits = outputs.logits
        noisy_logits = full_logits[:, :l]
        clean_logits = full_logits[:, l:]

        if not masked_indices.any():
            self.epoch_meter.update(
                split="train" if model.training else "eval",
                nll_sum=noisy_logits.sum() * 0.0,
                token_cnt=token_cnt_per_seq.sum(),
            )
            zero = noisy_logits.sum() * 0.0
            return (zero, outputs) if return_outputs else zero

        mismatch = (input_ids[masked_indices] != labels[masked_indices])
        if mismatch.any():
            input_ids[masked_indices] = labels[masked_indices]

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
            token_loss_normalized = token_loss / token_cnt_per_seq.expand(-1, l)[masked_indices] / b
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

        # progress-only: running mean of the decoding frontier (tokens/forward =
        # response_tokens / #chunks). Throttled rank-0 print so it never collides
        # with the Trainer's own loss logging.
        if model.training:
            with torch.no_grad():
                resp = labels != -100
                max_blk = torch.where(resp, block_ids, torch.zeros_like(block_ids)).amax(dim=1)
                fr = (resp.sum().float() / max_blk.clamp(min=1).sum().float()).item()
            self._fr_sum += fr
            self._fr_cnt += 1
            log_every = max(self.args.logging_steps, 1) * max(self.args.gradient_accumulation_steps, 1)
            if self.is_world_process_zero() and self._fr_cnt % log_every == 0:
                print(f"[chunkext] step~{self.state.global_step} "
                      f"frontier(tok/fwd)={self._fr_sum / self._fr_cnt:.3f} "
                      f"noisy_ce={noisy_ce.mean().item():.3f} clean_ce={clean_ce.mean().item():.3f}",
                      flush=True)

        outputs.logits = noisy_logits
        return (loss, outputs) if return_outputs else loss
