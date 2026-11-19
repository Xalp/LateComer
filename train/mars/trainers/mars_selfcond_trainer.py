"""
MARSSelfCondTrainer: MARS block training with belief carry-over ("self-conditioning") for ProVe proposers.

Motivation (2026-09-29). In ProVe the proposer writes a block of B tokens from B placeholders; after the verify pass commits
a tokens, the next cycle's window overlaps the previous one on its first B-a positions, and the previous forward's
distributions for those positions are thrown away. Soft-Masked DLM (ICLR 2026) and SCMDM (2026) show that a masked
diffusion model can be taught to read those beliefs if they are blended INTO the mask embedding (the mask keeps the
"I don't know" channel) and the model is trained with two passes: pass 1 (no grad) produces the beliefs, pass 2 sees the
blended placeholders, the loss is computed on pass 2 only. Untrained, any blend halves the accept length (probe 09-29).

Training simulation of the deployment geometry:
  * pass 1: the standard MARS [noisy | clean] forward, but with the block grid shifted LEFT by a random a in [1, B]
    per sequence (per-position block ids), so that position p sees the same context it had one cycle earlier, when it
    sat a positions deeper in the previous window. Its right-shifted noisy logits at index p are the belief about token p.
  * pass 2: the standard grid. Positions whose depth (p mod B) is < B - a existed in the previous window and get
        e_p = (1 - lam_p) * E[mask] + lam_p * sum_{top-k} pi_i E[v_i],     lam_p = sigmoid(w_s) * sigmoid(w_a * (-H_p) + w_b)
    (H_p = entropy of the pass-1 belief; w_s, w_a, w_b are three trainable scalars stored as model.selfcond_omega);
    the last a positions of every block are new in deployment and stay pure mask. With probability sc_drop a sequence
    gets no injection at all, so the pure-mask behaviour of the first cycle is kept.
  * loss: exactly MARS's (noisy CE + ar_weight * clean CE) on pass 2. Pass 1 is a no-grad forward (~1/3 of a step).

Inference (probe / server): keep the previous cycle's distributions for the overlapping positions and build the same
blend for the next proposal pass; verification stays hard and exact.
"""
import math

import torch
import torch.nn.functional as F

from .attention_mask import mars_attention_mask, mars_variable_block_mask
from .mars_trainer import MARSTrainer


def selfcond_lambda(omega: torch.Tensor, entropy: torch.Tensor) -> torch.Tensor:
    """lam = sigmoid(w_s) * sigmoid(w_a * (-H) + w_b); omega = [w_s, w_a, w_b]."""
    om = omega.float()
    return torch.sigmoid(om[0]) * torch.sigmoid(om[1] * (-entropy.float()) + om[2])


class MARSSelfCondTrainer(MARSTrainer):
    def __init__(self, *args, sc_topk: int = 10, sc_drop: float = 0.1, sc_shift_max: int | None = None,
                 sc_log_every: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.sc_topk = sc_topk
        self.sc_drop = sc_drop
        self.sc_shift_max = sc_shift_max or self.block_size
        self.sc_log_every = sc_log_every
        self._sc_step = 0

    # ---- masks -------------------------------------------------------------------------------------------
    def _standard_mask(self, l: int, device):
        m = mars_attention_mask(b=None, h=None, q_idx=torch.arange(l * 2)[:, None], kv_idx=torch.arange(l * 2)[None, :],
                                block_size=self.block_size, n=l)
        return m.unsqueeze(0).unsqueeze(0).expand(1, 1, 2 * l, 2 * l).to(device)

    def _shifted_mask(self, shifts: torch.Tensor, l: int, device):
        """Block grid moved left by `shifts[b]`: block id of logical position p = (p + a) // B."""
        pos = torch.arange(l, device=device).unsqueeze(0)                       # [1, l]
        block_ids = (pos + shifts.to(device).unsqueeze(1)) // self.block_size      # [b, l], non-decreasing
        return mars_variable_block_mask(block_ids)                                 # [b, 1, 2l, 2l]

    # ---- loss --------------------------------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = inputs["input_ids"], inputs["labels"], inputs.get("attention_mask", None)
        min_len = min(input_ids.shape[1], labels.shape[1])
        input_ids, labels = input_ids[:, :min_len], labels[:, :min_len]
        b, l = input_ids.shape
        dev = input_ids.device
        B = self.block_size
        token_cnt_per_seq = torch.sum(labels != -100, dim=1, keepdim=True)

        unwrapped = self.accelerator.unwrap_model(model)
        if unwrapped.config._attn_implementation != "sdpa":
            raise NotImplementedError("MARSSelfCondTrainer needs sdpa (4-D boolean masks)")
        embed = unwrapped.get_input_embeddings()
        omega = unwrapped.selfcond_omega

        masked_indices = labels != -100
        noised_input_ids = torch.where(masked_indices, self.processing_class.mask_token_id, input_ids)
        concat_ids = torch.cat([noised_input_ids, input_ids], dim=1)
        base_pos = torch.arange(l, device=dev).unsqueeze(0).expand(b, l)
        concat_pos = torch.cat([base_pos, base_pos], dim=1)

        if not masked_indices.any():
            zero = embed.weight.sum() * 0.0
            self.epoch_meter.update(split="train" if model.training else "eval", nll_sum=zero.detach(),
                                    token_cnt=token_cnt_per_seq.sum())
            return (zero, None) if return_outputs else zero

        # ---- pass 1: beliefs from the previous cycle (shifted grid, no grad) ----
        shifts = torch.randint(1, self.sc_shift_max + 1, (b,), device=dev)          # a = accepted tokens last cycle
        keep = (torch.rand(b, device=dev) >= self.sc_drop)                            # sequences that get injection
        with torch.no_grad():
            out1 = model(input_ids=concat_ids, attention_mask=self._shifted_mask(shifts, l, dev), position_ids=concat_pos)
            out1 = self._postprocess_outputs(out1)
            logits1 = out1.logits[:, :l].float()                                       # belief about token p at index p
            logp = torch.log_softmax(logits1, dim=-1)
            probs = logp.exp()
            entropy = -(probs * logp).sum(-1)                                          # [b, l]
            topv, topi = probs.topk(self.sc_topk, dim=-1)                              # [b, l, k]
            topv = topv / topv.sum(-1, keepdim=True)
            del out1, logits1, logp, probs

        depth = base_pos % B                                                           # standard-grid depth of p
        inject = masked_indices & (depth < (B - shifts).unsqueeze(1)) & keep.unsqueeze(1)   # [b, l]
        lam = selfcond_lambda(omega, entropy) * inject.float()                         # [b, l]

        # ---- pass 2: standard grid, blended placeholders, loss ----
        E = embed.weight
        emb_noisy = embed(noised_input_ids)                                            # [b, l, d]
        belief_emb = (topv.unsqueeze(-1).to(E.dtype) * E[topi]).sum(-2)                # [b, l, d]
        lam_ = lam.unsqueeze(-1).to(E.dtype)
        emb_noisy = (1 - lam_) * emb_noisy + lam_ * belief_emb
        concat_embeds = torch.cat([emb_noisy, embed(input_ids)], dim=1)

        outputs = model(inputs_embeds=concat_embeds, attention_mask=self._standard_mask(l, dev), position_ids=concat_pos)
        outputs = self._postprocess_outputs(outputs)
        full_logits = outputs.logits
        noisy_logits, clean_logits = full_logits[:, :l], full_logits[:, l:]

        mismatch = input_ids[masked_indices] != labels[masked_indices]
        if mismatch.any():
            input_ids[masked_indices] = labels[masked_indices]
        noisy_ce = F.cross_entropy(noisy_logits[masked_indices], labels[masked_indices], reduction="none")
        clean_ce = F.cross_entropy(clean_logits[masked_indices], labels[masked_indices], reduction="none")
        token_loss = self.noisy_weight * noisy_ce + self.ar_weight * clean_ce
        if self.loss_normalization_type == "batch":
            token_loss_normalized = token_loss / b
        elif self.loss_normalization_type == "sequence":
            token_loss_normalized = token_loss / token_cnt_per_seq.expand(-1, l)[masked_indices] / b
        elif self.loss_normalization_type == "token":
            token_loss_normalized = token_loss / token_cnt_per_seq.sum()
        else:
            raise ValueError("Invalid loss_normalization_type.")
        loss = token_loss_normalized.sum()

        self.epoch_meter.update(split="train" if model.training else "eval", nll_sum=noisy_ce.sum(),
                                token_cnt=token_cnt_per_seq.sum())
        self._sc_step += 1
        if self.sc_log_every and self._sc_step % self.sc_log_every == 0 and self.accelerator.is_main_process:
            with torch.no_grad():
                inj = inject.float().sum().clamp_min(1)
                om = omega.float().tolist()
                print(f"[selfcond] step {self._sc_step} inject_frac={inject.float().sum().item()/masked_indices.float().sum().clamp_min(1).item():.3f} "
                      f"lam_mean={(lam.sum()/inj).item():.3f} H_mean={(entropy*inject).sum().item()/inj.item():.2f} "
                      f"omega={[round(x,3) for x in om]} noisy_ce={noisy_ce.mean().item():.3f} clean_ce={clean_ce.mean().item():.3f}", flush=True)
        outputs.logits = noisy_logits
        return (loss, outputs) if return_outputs else loss
