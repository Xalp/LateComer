"""
MARSFlowSCTrainer: ELF-ified MARS trained on its OWN beliefs ("flow + self-conditioning", 2026-10-01).

Why. MARSFlowTrainer builds the carried state from the TRUE token, z_t = t * E[x] + (1 - t) * eps, and the model learns to
read it almost perfectly (gtmix alpha=.2 -> 95 % match). At inference the state carries the model's own previous-cycle
belief x_hat, which at the carried positions is right only ~40-60 % of the time, and the model trusts it like partial
truth: carried positions match at .36-.52 vs .68-.74 for fresh ones (flow7b probes 10-01). No inference-side rule
(argmax states, small dt, confidence-gated t, top-k) recovers this. ELF fixes the same exposure bias by self-conditioning:
train on the model's own prediction. Here the state itself is the self-condition, so the training input at a carried
position becomes

    z_p = t_p * x_hat_p + (1 - t_p) * eps_p (+ time embedding of t_p),   x_hat_p = E^T softmax(pass-1 logits at p)

for a fraction sc_p of the sequences (default 1.0 = deployment-matched; the true-token version is what the plain flow
trainer does and is never seen at inference), where pass 1 is a no-grad forward at t = 0 (every supervised position = eps)
on the block grid shifted LEFT by a ~ U[0, B] (a = tokens committed last cycle; a = 0 is the same-context refinement that
--flow-steps 2 performs at inference). Pass 2 uses the standard grid; positions with depth < B - a carry the pass-1 state,
the last a positions of each block are fresh (t = 0); loss = MARS loss on pass 2 (noisy CE + ar_weight * clean CE). The
model therefore learns P(x | ctx, its own belief at strength t) = calibrated trust; the probe's --placeholder flow --flow-carry
replays exactly this geometry.

Logged every flow_log_every steps: pass-1 belief top-1 accuracy at carried positions, pass-2 accuracy at carried vs fresh
positions (the decisive number: carried must beat fresh and beat the belief), CE by t bucket.
"""
import torch
import torch.nn.functional as F

from .attention_mask import mars_variable_block_mask
from .mars_flow_trainer import MARSFlowTrainer, flow_time_features


class MARSFlowSCTrainer(MARSFlowTrainer):
    def __init__(self, *args, flow_sc_p: float = 1.0, flow_sc_shift_min: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow_sc_p = flow_sc_p
        self.flow_sc_shift_min = flow_sc_shift_min

    def _shifted_mask(self, shifts: torch.Tensor, l: int, device):
        """Block grid moved left by shifts[b]: block id of position p = (p + a) // B (a = 0 -> standard grid)."""
        pos = torch.arange(l, device=device).unsqueeze(0)
        block_ids = (pos + shifts.to(device).unsqueeze(1)) // self.block_size
        return mars_variable_block_mask(block_ids)

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels = inputs["input_ids"], inputs["labels"]
        min_len = min(input_ids.shape[1], labels.shape[1])
        input_ids, labels = input_ids[:, :min_len], labels[:, :min_len]
        b, l = input_ids.shape
        dev = input_ids.device
        B = self.block_size
        token_cnt_per_seq = torch.sum(labels != -100, dim=1, keepdim=True)

        unwrapped = self.accelerator.unwrap_model(model)
        if unwrapped.config._attn_implementation != "sdpa":
            raise NotImplementedError("MARSFlowSCTrainer needs sdpa (4-D boolean masks)")
        embed = unwrapped.get_input_embeddings()
        E = embed.weight
        V = len(self.processing_class)
        if self._noise_std is None:
            self._noise_std = (E[:V].detach().float().std(0) * self.flow_noise_scale).to(E.dtype)
        proj = unwrapped.flow_time_proj

        masked = labels != -100
        if not masked.any():
            zero = E.sum() * 0.0
            self.epoch_meter.update(split="train" if model.training else "eval", nll_sum=zero.detach(), token_cnt=token_cnt_per_seq.sum())
            return (zero, None) if return_outputs else zero

        # ---- deployment-shaped corruption schedule ----
        base_pos = torch.arange(l, device=dev).unsqueeze(0).expand(b, l)
        depth = base_pos % B
        nblk = (l + B - 1) // B
        t_blk = torch.rand(b, nblk, device=dev) * self.flow_tmax
        t_pos = t_blk.gather(1, base_pos // B)                                           # [b, l]
        a = torch.randint(self.flow_sc_shift_min, B + 1, (b,), device=dev)              # committed last cycle (0 = refinement)
        new_pos = depth >= (B - a).unsqueeze(1)                                          # last a of every block: fresh
        cold = (torch.rand(b, device=dev) < self.flow_p0).unsqueeze(1)                   # whole sequence fresh
        t_pos = torch.where(new_pos | cold, torch.zeros_like(t_pos), t_pos)
        t_pos = torch.where(masked, t_pos, torch.zeros_like(t_pos))
        carried = masked & (t_pos > 0)                                                   # [b, l]
        use_self = (torch.rand(b, device=dev) < self.flow_sc_p).unsqueeze(1) & carried   # state from own belief
        concat_pos = torch.cat([base_pos, base_pos], dim=1)

        clean_emb = embed(input_ids)                                                     # [b, l, d]
        eps = torch.randn(b, l, E.shape[1], device=dev, dtype=torch.float32) * self._noise_std.float()
        temb0 = proj(flow_time_features(torch.zeros(1, device=dev)).to(proj.weight.dtype)).float()   # [1, d]

        # ---- pass 1 (no grad): previous cycle's proposal = t = 0 everywhere, grid shifted left by a ----
        belief_acc = None
        if use_self.any():
            with torch.no_grad():
                z0 = eps + temb0.unsqueeze(0)
                noisy0 = torch.where(masked.unsqueeze(-1), z0.to(E.dtype), clean_emb)
                out1 = model(inputs_embeds=torch.cat([noisy0, clean_emb], dim=1), attention_mask=self._shifted_mask(a, l, dev),
                             position_ids=concat_pos)
                out1 = self._postprocess_outputs(out1)
                logits1 = out1.logits[:, :l]                                             # belief about token p at index p
                xhat = torch.zeros(b, l, E.shape[1], device=dev, dtype=torch.float32)
                top1 = torch.zeros(b, l, device=dev, dtype=torch.long)
                for i in range(b):                                                       # row by row: [l, V] probs
                    pr = logits1[i, :, :V].float().softmax(-1)
                    xhat[i] = pr @ E[:V].float()
                    top1[i] = pr.argmax(-1)
                belief_acc = (top1 == labels)[use_self].float().mean().item()
                del out1, logits1

        # ---- pass 2 (grad): standard grid; carried positions = own belief (or true token), fresh = eps ----
        tt = t_pos.unsqueeze(-1)
        x_src = torch.where(use_self.unsqueeze(-1), xhat, clean_emb.float()) if use_self.any() else clean_emb.float()
        z = tt * x_src + (1 - tt) * eps
        z = z + proj(flow_time_features(t_pos).to(proj.weight.dtype)).float()
        noisy_emb = torch.where(masked.unsqueeze(-1), z.to(E.dtype), clean_emb)
        outputs = model(inputs_embeds=torch.cat([noisy_emb, clean_emb], dim=1), attention_mask=self._standard_mask(l, dev),
                        position_ids=concat_pos)
        outputs = self._postprocess_outputs(outputs)
        full_logits = outputs.logits
        noisy_logits, clean_logits = full_logits[:, :l], full_logits[:, l:]

        mismatch = input_ids[masked] != labels[masked]
        if mismatch.any():
            input_ids[masked] = labels[masked]
        noisy_ce = F.cross_entropy(noisy_logits[masked], labels[masked], reduction="none")
        clean_ce = F.cross_entropy(clean_logits[masked], labels[masked], reduction="none")
        token_loss = self.noisy_weight * noisy_ce + self.ar_weight * clean_ce
        if self.loss_normalization_type == "batch":
            token_loss_normalized = token_loss / b
        elif self.loss_normalization_type == "sequence":
            token_loss_normalized = token_loss / token_cnt_per_seq.expand(-1, l)[masked] / b
        elif self.loss_normalization_type == "token":
            token_loss_normalized = token_loss / token_cnt_per_seq.sum()
        else:
            raise ValueError("Invalid loss_normalization_type.")
        loss = token_loss_normalized.sum()

        self.epoch_meter.update(split="train" if model.training else "eval", nll_sum=noisy_ce.sum(), token_cnt=token_cnt_per_seq.sum())
        self._flow_step += 1
        if self.flow_log_every and self._flow_step % self.flow_log_every == 0 and self.accelerator.is_main_process:
            with torch.no_grad():
                tm = t_pos[masked]
                pred = noisy_logits[masked].argmax(-1) == labels[masked]
                sel_self = use_self[masked]
                sel_fresh = ~carried[masked]
                def bucket(lo, hi):
                    sel = (tm >= lo) & (tm < hi)
                    return f"{noisy_ce[sel].mean().item():.3f}(n={int(sel.sum())})" if sel.any() else "-"
                def acc(sel):
                    return f"{pred[sel].float().mean().item():.3f}(n={int(sel.sum())})" if sel.any() else "-"
                print(f"[flowsc] step {self._flow_step} belief_acc={belief_acc if belief_acc is None else round(belief_acc, 3)} "
                      f"acc2 self={acc(sel_self)} fresh={acc(sel_fresh)} | ce t=0:{bucket(0, 1e-6)} t<.3:{bucket(1e-6, .3)} "
                      f"t<.6:{bucket(.3, .6)} t>=.6:{bucket(.6, 1.01)} clean_ce={clean_ce.mean().item():.3f} "
                      f"|time_proj|={proj.weight.float().norm().item():.3f}", flush=True)
        outputs.logits = noisy_logits
        return (loss, outputs) if return_outputs else loss
