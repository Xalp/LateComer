"""train_mars_firstt.py — W5 Arm-F (11-04): stage-2 block training with a TRUNCATED-READOUT loss on half the blocks,
the mirror of the paper arm's late-entry (mid-loop injection) training. NEW FILE on purpose: the shared train_mars.py and
the a2d-ouro wrapper are imported from HDFS by RUNNING jobs at every retry, so Arm-F patches the model class at runtime in
THIS process only and leaves every shared file untouched.

Mechanics (mirrors injection_plan's 'step' mode block selection):
  env MARS_FIRSTT = probability per 16-block of the NOISY half ([noisy | clean] MARS layout) that ALL its positions read out
  the gate mixture renormalised over the first t passes (last kept pass takes the stick-breaking remainder — exactly the
  t-pass model ar_t), with t ~ U{1..MARS_FIRSTT_TMAX (default 3)} per block. Every other position (unselected noisy blocks,
  the whole clean half) reads out the standard full-T stick-breaking mixture, so MARS_FIRSTT=0 degenerates to the nolate
  recipe. The trainer computes the loss from the returned logits (the _split_readout precedent), so losses/masks/weights
  are untouched.

Pre-registered loss fingerprint (verify from trainer_state after launch): at the shared seed/data, step-10 logged loss must
sit ABOVE nolate's 156.94 (half the noisy blocks read out truncated, where the ar_sft init is far weaker at t = 1, 2);
exactly 156.94 means the knob did not fire; ~133.9 means injection fired (wrong knob).

Usage: identical to train_mars.py (same CLI); plus env MARS_FIRSTT / MARS_FIRSTT_TMAX.
"""
import os

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

import train_mars as tm
from dllm.pipelines.a2d import A2DOuroLMHeadModel

FIRSTT_P = float(os.environ.get("MARS_FIRSTT", "0") or 0)
FIRSTT_TMAX = int(os.environ.get("MARS_FIRSTT_TMAX", "3") or 3)

_orig_forward = A2DOuroLMHeadModel.forward


def _firstt_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None,
                    labels=None, use_cache=None, cache_position=None, logits_to_keep=0, use_weighted_exit=None,
                    exit_at_step=None, exit_threshold=None, **kwargs):
    if not (self.training and FIRSTT_P > 0 and input_ids is not None):
        return _orig_forward(self, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                             past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                             cache_position=cache_position, logits_to_keep=logits_to_keep, use_weighted_exit=use_weighted_exit,
                             exit_at_step=exit_at_step, exit_threshold=exit_threshold, **kwargs)
    B, N = input_ids.shape; half = N // 2; bs = int(getattr(self, "block_size", 16)); T = self.model.total_ut_steps
    tsel = torch.full((B, N), T, dtype=torch.long, device=input_ids.device)       # per-position readout budget (T = full)
    for b in range(B):
        for s0 in range(0, half, bs):
            if torch.rand(1).item() < FIRSTT_P:
                tsel[b, s0: min(s0 + bs, half)] = int(torch.randint(1, FIRSTT_TMAX + 1, (1,)).item())
    _, hs, gl = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                           inputs_embeds=inputs_embeds, use_cache=False, **kwargs)
    lam = torch.stack([g.squeeze(-1).float().sigmoid() for g in gl])              # [T, B, N]
    rem_before = torch.cat([torch.ones_like(lam[:1]), torch.cumprod(1 - lam, 0)[:-1]], 0)   # prod_{j<i}(1 - lam_j)
    wfull = lam * rem_before
    ar = torch.arange(T, device=lam.device)[:, None, None]
    tl = tsel[None]
    W = torch.where(ar < tl - 1, wfull, torch.where(ar == tl - 1, rem_before, torch.zeros_like(wfull)))
    mix = None
    for i in range(T):                                                            # per-pass accumulation, _split_readout convention
        step = self.lm_head(hs[i]) * W[i].unsqueeze(-1).to(hs[i].dtype)
        mix = step if mix is None else mix + step
    return CausalLMOutputWithPast(logits=mix)


A2DOuroLMHeadModel.forward = _firstt_forward
print(f"[firstt] A2DOuroLMHeadModel.forward patched (this process only): first-t truncated readout "
      f"p={FIRSTT_P} tmax={FIRSTT_TMAX} (per noisy-half 16-block; unselected positions = full-T mixture)", flush=True)

if __name__ == "__main__":
    tm.train()
