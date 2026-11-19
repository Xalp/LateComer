"""
MARS Trainers.

MARSTrainer: Full-mask trainer with auxiliary AR loss on the clean stream.
    Loss = CE(noisy_logits, labels) + ar_weight * CE(clean_logits, labels)
    The clean stream does standard causal AR prediction (already computed).
    Adding CE on it ensures AR signal stays ~50% regardless of block size.

MARSTrainerNoSFT: Full-mask trainer without auxiliary AR loss.
    Uses only the noisy stream for loss computation, with diffusion-timestep
    loss weighting.
"""

from functools import partial

import torch
import torch.nn.functional as F
import transformers
from torch.nn.attention.flex_attention import create_block_mask

from dllm.core.trainers.csbd3lm_v2 import CSBD3LMv2Trainer
from .attention_mask import mars_attention_mask


def _arrow_lengths(ds, column):
    """lengths of a list column straight from the Arrow offsets (no row materialisation); honours an indices mapping; None on any doubt"""
    try:
        import datasets as hfds, pyarrow.compute as pc
        if not isinstance(ds, hfds.Dataset) or column not in ds.column_names: return None
        tbl = ds.data.table if hasattr(ds.data, "table") else ds.data
        lens = pc.list_value_length(tbl.column(column)).to_numpy(zero_copy_only=False)
        if ds._indices is not None:
            it = ds._indices.table if hasattr(ds._indices, "table") else ds._indices
            lens = lens[it.column(0).to_numpy()]
        if len(lens) != len(ds): return None
        return lens.tolist()
    except Exception as e:
        print(f"[mars] arrow length shortcut failed ({e!r}); falling back to the slow sampler path"); return None


class MARSTrainer(CSBD3LMv2Trainer):
    """
    MARS trainer with auxiliary AR loss on the clean stream.
    Keeps AR signal ratio stable regardless of block size.
    """

    def __init__(self, *args, ar_weight: float = 1.0, noisy_weight: float = 1.0, consist_weight: float = 0.0, consist_tau: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ar_weight = ar_weight
        # noisy_weight=0 turns the dual-stream objective into a PURE-AR SFT (standard
        # training): the mask/noisy stream still runs but contributes no gradient, so
        # the mask embedding stays untrained ("self-born MARS" stage-A regime).
        self.noisy_weight = noisy_weight
        # consist_weight>0 adds a self-consistency term (MARS-Ouro v3, 10-15): KL(clean || noisy) per supervised position, the clean
        # stream's own next-token distribution (detached) teaching the noisy stream (mask rows and mid-loop-injected rows) -- i.e. the
        # proposer learns to predict what THIS model's AR would say, which is what a verifier accepts. Zero extra forwards: both
        # streams are already in the batch. consist_tau tempers the teacher.
        self.consist_weight = consist_weight
        self.consist_tau = consist_tau

    def _get_train_sampler(self, train_dataset=None):
        # 10-18: with group_by_length HF's LengthGroupedSampler measures every row in Python when the dataset has no `length` column
        # (and Trainer strips such a column before building the sampler): 1.89M Dolci rows = ~25 min at the start of EVERY attempt,
        # i.e. half the dead time of each T=16 restart. Same lengths (len(input_ids)) from the Arrow offsets in well under a second,
        # same sampler + same seed -> identical order, so resumed runs keep their data order.
        ds = train_dataset if train_dataset is not None else self.train_dataset
        lengths = _arrow_lengths(ds, "input_ids") if (self.args.group_by_length and ds is not None) else None
        if lengths is None:
            return super()._get_train_sampler(train_dataset)
        from transformers.trainer_pt_utils import LengthGroupedSampler
        model_input_name = self.processing_class.model_input_names[0] if self.processing_class is not None else None
        print(f"[mars] group_by_length: {len(lengths)} lengths from Arrow offsets (fast path)")
        return LengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps, dataset=ds, lengths=lengths, model_input_name=model_input_name)

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
        # Ensure input_ids and labels have the same length
        min_len = min(input_ids.shape[1], labels.shape[1])
        input_ids = input_ids[:, :min_len]
        labels = labels[:, :min_len]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :min_len]
        b, l = input_ids.shape
        token_cnt_per_seq = torch.sum(labels != -100, dim=1, keepdim=True)

        # === 1. Diffusion timesteps ===
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )

        # === 2. Full masking ===
        masked_indices = (labels != -100)
        noised_input_ids = torch.where(
            masked_indices, self.processing_class.mask_token_id, input_ids
        )

        # === 3. Forward pass ===
        concat_input_ids = torch.cat([noised_input_ids, input_ids], dim=1)

        if self.accelerator.unwrap_model(model).config._attn_implementation == "sdpa":
            attention_mask = mars_attention_mask(
                b=None, h=None,
                q_idx=torch.arange(l * 2)[:, None],
                kv_idx=torch.arange(l * 2)[None, :],
                block_size=self.block_size, n=l,
            )
            attention_mask = (
                attention_mask.unsqueeze(0).unsqueeze(0).expand(1, 1, 2 * l, 2 * l)
            )
            attention_mask = attention_mask.to(input_ids.device)
        elif (
            self.accelerator.unwrap_model(model).config._attn_implementation
            == "flex_attention"
        ):
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

        # === 4. Degenerate cases ===
        if not masked_indices.any():
            self.epoch_meter.update(
                split="train" if model.training else "eval",
                nll_sum=noisy_logits.sum() * 0.0,
                token_cnt=token_cnt_per_seq.sum(),
            )
            zero = noisy_logits.sum() * 0.0
            return (zero, outputs) if return_outputs else zero

        # === 5. Fix mismatch ===
        mismatch = (input_ids[masked_indices] != labels[masked_indices])
        if mismatch.any():
            input_ids[masked_indices] = labels[masked_indices]

        # === 6a. Noisy stream CE (full mask loss) ===
        noisy_ce = F.cross_entropy(
            noisy_logits[masked_indices], labels[masked_indices], reduction="none"
        )

        # === 6b. Clean stream CE (AR loss) ===
        clean_ce = F.cross_entropy(
            clean_logits[masked_indices], labels[masked_indices], reduction="none"
        )

        # === 7. Combine and normalize ===
        token_loss = self.noisy_weight * noisy_ce + self.ar_weight * clean_ce
        if self.consist_weight > 0:                                   # === 6c. self-consistency: noisy stream -> clean stream (detached) ===
            teacher = F.softmax(clean_logits[masked_indices].detach().float() / self.consist_tau, dim=-1)
            student = F.log_softmax(noisy_logits[masked_indices].float(), dim=-1)
            consist_kl = F.kl_div(student, teacher, reduction="none").sum(-1)
            token_loss = token_loss + self.consist_weight * consist_kl

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

        outputs.logits = noisy_logits
        return (loss, outputs) if return_outputs else loss


class MARSTrainerNoSFT(CSBD3LMv2Trainer):
    """
    MARS trainer without auxiliary AR loss.
    The noisy input is ALWAYS fully masked (p_mask = 1.0).
    This forces the model to predict the current block (causally) based ONLY
    on previous clean blocks. Uses diffusion-timestep loss weighting.
    """

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
        # Ensure input_ids and labels have the same length (truncate to shorter)
        min_len = min(input_ids.shape[1], labels.shape[1])
        input_ids = input_ids[:, :min_len]
        labels = labels[:, :min_len]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :min_len]
        b, l = input_ids.shape
        token_cnt_per_seq = torch.sum(labels != -100, dim=1, keepdim=True)

        # === 1. Sample diffusion timesteps ===
        # We keep t random for the loss weighting function (if it uses t),
        # but force full masking (p_mask = 1.0).
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )

        # FORCE FULL MASKING
        p_mask = torch.ones((b, l), device=input_ids.device)

        # === 2. Apply stochastic masking ===
        # Everything valid (not -100) is masked.
        masked_indices = (labels != -100)

        noised_input_ids = torch.where(
            masked_indices, self.processing_class.mask_token_id, input_ids
        )

        # === 3. Forward pass (MARS Mask Logic) ===
        concat_input_ids = torch.cat([noised_input_ids, input_ids], dim=1)

        if self.accelerator.unwrap_model(model).config._attn_implementation == "sdpa":
            attention_mask = mars_attention_mask(
                b=None,
                h=None,
                q_idx=torch.arange(l * 2)[:, None],
                kv_idx=torch.arange(l * 2)[None, :],
                block_size=self.block_size,
                n=l,
            )
            attention_mask = (
                attention_mask.unsqueeze(0).unsqueeze(0).expand(1, 1, 2 * l, 2 * l)
            )
            attention_mask = attention_mask.to(input_ids.device)
        elif (
            self.accelerator.unwrap_model(model).config._attn_implementation
            == "flex_attention"
        ):
            attention_mask = create_block_mask(
                partial(mars_attention_mask, block_size=self.block_size, n=l),
                B=None,
                H=None,
                Q_LEN=l * 2,
                KV_LEN=l * 2,
            )
        else:
            raise NotImplementedError

        base_pos = (
            torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, l)
        )
        concat_position_ids = torch.cat([base_pos, base_pos], dim=1)

        outputs = model(
            input_ids=concat_input_ids,
            attention_mask=attention_mask,
            position_ids=concat_position_ids,
        )
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits
        logits = logits[:, :l]

        # === 4. Degenerate cases ===
        if not masked_indices.any():
            self.epoch_meter.update(
                split="train" if model.training else "eval",
                nll_sum=logits.sum() * 0.0,
                token_cnt=token_cnt_per_seq.sum(),
            )
            return (
                (logits.sum() * 0.0, outputs) if return_outputs else logits.sum() * 0.0
            )

        # === 5. Loss weights ===
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_indices=masked_indices
        )

        # === 6. Cross-entropy ===
        mismatch = (input_ids[masked_indices] != labels[masked_indices])
        if mismatch.any():
            input_ids[masked_indices] = labels[masked_indices]

        token_loss = torch.nn.functional.cross_entropy(
            logits[masked_indices], labels[masked_indices], reduction="none"
        )
        token_loss = token_loss * loss_weights[masked_indices]

        # === 7. Normalize ===
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
            nll_sum=token_loss.sum(),
            token_cnt=token_cnt_per_seq.sum(),
        )
        return (loss, outputs) if return_outputs else loss
