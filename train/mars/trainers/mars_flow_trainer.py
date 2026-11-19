"""
MARSFlowTrainer: MARS block training with a CONTINUOUS corruption ("ELF-ified MARS", 2026-10-01).

MARS's noisy stream is x-prediction at a single corruption level: the input at every block position is the mask (in
practice a near-zero vector) and the loss is the cross-entropy of the clean token. ELF (Embedded Language Flows,
arXiv 2605.10938) does x-prediction along a rectified-flow path z_t = t * x + (1 - t) * eps and therefore knows what to do
with a partially denoised state. Here the two are merged with the smallest possible change:

  * noisy-stream input at a supervised position p:   z_p = t_p * E[x_p] + (1 - t_p) * eps_p (+ time embedding of t_p)
        E = the model's own input-embedding table, eps_p ~ N(0, diag(std(E)^2)) * noise_scale, fixed per position,
        t_p in [0, t_max]. The "unknown" end is Gaussian noise, not the zero vector, so the interpolation changes
        DIRECTION with t and survives the first RMSNorm (a zero end would be a pure scale, invisible to the model).
  * t schedule simulates deployment: per sequence a ~ U[1, B] (tokens the previous cycle committed); positions with
        block depth < B - a existed in the previous window and get the block's t_blk ~ U(0, t_max), the last a positions
        of every block are new and get t = 0; with probability p0 the whole sequence is at t = 0 (cold cycle).
  * output and loss are UNCHANGED from MARS: CE of the clean token at every supervised noisy position (discrete
        x-prediction = ELF's decode head; no MSE branch is needed because the output already is a distribution) plus
        ar_weight * CE on the clean stream. Block-causal dual-stream attention unchanged. One forward per step.

Inference (ProVe-flow, see placeholder_probe.py --placeholder flow): new window positions start at eps (t = 0); after a
proposal pass the expected embedding x_hat = E^T softmax(logits) gives the Euler update z_{t+dt} = z_t + dt (x_hat - eps);
positions that are not committed carry their z_t into the next cycle; the verify pass stays hard and exact.
"""
import math

import torch
import torch.nn.functional as F

from .attention_mask import mars_attention_mask
from .mars_trainer import MARSTrainer

TIME_FEATS = 64


def flow_time_features(t: torch.Tensor) -> torch.Tensor:
    """Sinusoidal features of t in [0, 1]: [..., 64] = [sin(pi k t), cos(pi k t)] for k = 1..32."""
    k = torch.arange(1, TIME_FEATS // 2 + 1, device=t.device, dtype=torch.float32)
    ang = math.pi * t.float().unsqueeze(-1) * k
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class MARSFlowTrainer(MARSTrainer):
    def __init__(self, *args, flow_p0: float = 0.3, flow_tmax: float = 0.9, flow_noise_scale: float = 1.0,
                 flow_log_every: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow_p0 = flow_p0
        self.flow_tmax = flow_tmax
        self.flow_noise_scale = flow_noise_scale
        self.flow_log_every = flow_log_every
        self._flow_step = 0
        self._noise_std = None

    def _standard_mask(self, l: int, device):
        m = mars_attention_mask(b=None, h=None, q_idx=torch.arange(l * 2)[:, None], kv_idx=torch.arange(l * 2)[None, :],
                                block_size=self.block_size, n=l)
        return m.unsqueeze(0).unsqueeze(0).expand(1, 1, 2 * l, 2 * l).to(device)

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
            raise NotImplementedError("MARSFlowTrainer needs sdpa (4-D boolean masks)")
        embed = unwrapped.get_input_embeddings()
        E = embed.weight
        if self._noise_std is None:
            V = len(self.processing_class)
            self._noise_std = (E[:V].detach().float().std(0) * self.flow_noise_scale).to(E.dtype)

        masked = labels != -100
        if not masked.any():
            zero = E.sum() * 0.0
            self.epoch_meter.update(split="train" if model.training else "eval", nll_sum=zero.detach(), token_cnt=token_cnt_per_seq.sum())
            return (zero, None) if return_outputs else zero

        # ---- corruption level per position (deployment-shaped) ----
        base_pos = torch.arange(l, device=dev).unsqueeze(0).expand(b, l)
        depth = base_pos % B
        nblk = (l + B - 1) // B
        t_blk = torch.rand(b, nblk, device=dev) * self.flow_tmax                       # per block
        t_pos = t_blk.gather(1, base_pos // B)                                          # [b, l]
        a = torch.randint(1, B + 1, (b,), device=dev)                                  # committed last cycle
        new_pos = depth >= (B - a).unsqueeze(1)                                         # last a of every block: fresh
        cold = (torch.rand(b, device=dev) < self.flow_p0).unsqueeze(1)                  # whole sequence at t = 0
        t_pos = torch.where(new_pos | cold, torch.zeros_like(t_pos), t_pos)
        t_pos = torch.where(masked, t_pos, torch.zeros_like(t_pos))

        # ---- noisy stream: z_t at supervised positions, real tokens elsewhere (prompt) ----
        clean_emb = embed(input_ids)                                                    # [b, l, d]
        eps = torch.randn(b, l, E.shape[1], device=dev, dtype=torch.float32) * self._noise_std.float()
        z = t_pos.unsqueeze(-1) * clean_emb.float() + (1 - t_pos.unsqueeze(-1)) * eps
        proj = unwrapped.flow_time_proj
        z = z + proj(flow_time_features(t_pos).to(proj.weight.dtype)).float()   # proj is bf16 under the DeepSpeed engine
        noisy_emb = torch.where(masked.unsqueeze(-1), z.to(E.dtype), clean_emb)
        concat_embeds = torch.cat([noisy_emb, clean_emb], dim=1)
        concat_pos = torch.cat([base_pos, base_pos], dim=1)

        outputs = model(inputs_embeds=concat_embeds, attention_mask=self._standard_mask(l, dev), position_ids=concat_pos)
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
                def bucket(lo, hi):
                    sel = (tm >= lo) & (tm < hi)
                    return f"{noisy_ce[sel].mean().item():.3f}(n={int(sel.sum())})" if sel.any() else "-"
                print(f"[flow] step {self._flow_step} ce t=0:{bucket(0, 1e-6)} t<.3:{bucket(1e-6, .3)} t<.6:{bucket(.3, .6)} "
                      f"t>=.6:{bucket(.6, 1.01)} clean_ce={clean_ce.mean().item():.3f} |time_proj|={unwrapped.flow_time_proj.weight.float().norm().item():.3f}",
                      flush=True)
        outputs.logits = noisy_logits
        return (loss, outputs) if return_outputs else loss
