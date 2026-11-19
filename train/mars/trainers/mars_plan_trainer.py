"""
MARSPlanTrainer: plan-aware next-token generation ("generate bidirectional, commit causal", user's design 2026-10-01).

Structure. The committed stream (clean) stays a strictly causal AR model: KV cache, prefix cache and exact verification are
untouched. Each step appends a BLOCK of B rows after the committed prefix: the HEAD (the last committed token, re-fed as
the first row of the block) followed by B-1 plan slots (noise / flow states). The block is bidirectional inside and sees
the clean rows strictly before the head. The head row predicts the next token WHILE SEEING THE PLAN (the slots' hidden
states, which are future-token sketches computed in the same forward); slot j predicts the token j positions further
(MARS plan supervision, which also makes the slots usable as ProVe proposals). Once the token is committed it only exists
in the causal clean stream: the representation that produced it saw the plan, the representation later tokens attend to
did not. That gap is intended.

Training (one forward per pass, MARS cost): block grid shifted by a random offset per sequence (so every position is a head
somewhere in the data); the head position of every block holds the REAL token in the noisy stream (no state, no noise),
all other supervised positions hold flow states z = t*x + (1-t)*eps with the deployment-shaped t schedule of
MARSFlowSCTrainer (carried / fresh / cold) and self-conditioning (x = the model's own pass-1 belief, pass 1 = no-grad
plan forward on the grid shifted left by a, a in {0 (same-context refinement), 1 (one token committed per step), 2..B}).
Loss = MARS loss on pass 2 (CE at every supervised position of the noisy stream + ar_weight * clean CE); the head rows'
CE is the plan-aware AR objective, the slot rows' CE is the plan objective.

Logged: head-row accuracy vs clean-row accuracy at the SAME positions (does seeing the plan beat causal AR?), belief /
self / fresh accuracies, CE by t bucket.
"""
import torch
import torch.nn.functional as F

from .attention_mask import mars_plan_block_mask
from .mars_flow_sc_trainer import MARSFlowSCTrainer
from .mars_flow_trainer import flow_time_features


class MARSPlanTrainer(MARSFlowSCTrainer):
    def __init__(self, *args, plan_p_a0: float = 0.3, plan_p_a1: float = 0.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.plan_p_a0 = plan_p_a0
        self.plan_p_a1 = plan_p_a1

    def _plan_mask(self, offs: torch.Tensor, l: int, device):
        """Block id of position p = (p + offs[b]) // B; noisy bidirectional inside the block."""
        pos = torch.arange(l, device=device).unsqueeze(0)
        return mars_plan_block_mask((pos + offs.to(device).unsqueeze(1)) // self.block_size)

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
            raise NotImplementedError("MARSPlanTrainer needs sdpa (4-D boolean masks)")
        embed = unwrapped.get_input_embeddings()
        E = embed.weight
        V = len(self.processing_class)
        if self._noise_std is None:
            self._noise_std = (E[:V].detach().float().std(0) * self.flow_noise_scale).to(E.dtype)
        proj = unwrapped.flow_time_proj

        masked = labels != -100                                                          # supervised positions
        if not masked.any():
            zero = E.sum() * 0.0
            self.epoch_meter.update(split="train" if model.training else "eval", nll_sum=zero.detach(), token_cnt=token_cnt_per_seq.sum())
            return (zero, None) if return_outputs else zero

        # ---- grids: pass 2 at offset `shift`, pass 1 (previous cycle) at offset shift + a ----
        base_pos = torch.arange(l, device=dev).unsqueeze(0).expand(b, l)
        shift = torch.randint(0, B, (b,), device=dev)
        u = torch.rand(b, device=dev)
        a = torch.where(u < self.plan_p_a0, torch.zeros_like(shift),
                        torch.where(u < self.plan_p_a0 + self.plan_p_a1, torch.ones_like(shift), torch.randint(2, B + 1, (b,), device=dev)))
        depth2 = (base_pos + shift.unsqueeze(1)) % B
        head2 = depth2 == 0                                                              # real token in the noisy stream
        depth1 = (base_pos + shift.unsqueeze(1) + a.unsqueeze(1)) % B
        head1 = depth1 == 0
        noised2 = masked & ~head2
        noised1 = masked & ~head1

        # ---- corruption level per position (deployment-shaped), heads never noised ----
        blk2 = (base_pos + shift.unsqueeze(1)) // B
        t_blk = torch.rand(b, (l + 2 * B) // B + 1, device=dev) * self.flow_tmax
        t_pos = t_blk.gather(1, blk2)
        new_pos = depth2 >= (B - a).unsqueeze(1)                                         # last a of each block: fresh
        cold = (torch.rand(b, device=dev) < self.flow_p0).unsqueeze(1)
        t_pos = torch.where(new_pos | cold | ~noised2, torch.zeros_like(t_pos), t_pos)
        carried = noised2 & (t_pos > 0)
        use_self = (torch.rand(b, device=dev) < self.flow_sc_p).unsqueeze(1) & carried
        concat_pos = torch.cat([base_pos, base_pos], dim=1)

        clean_emb = embed(input_ids)
        eps = torch.randn(b, l, E.shape[1], device=dev, dtype=torch.float32) * self._noise_std.float()
        temb0 = proj(flow_time_features(torch.zeros(1, device=dev)).to(proj.weight.dtype)).float()

        # ---- pass 1 (no grad): previous cycle's plan forward at t = 0 on the a-shifted grid ----
        belief_acc = None
        xhat = None
        if use_self.any():
            with torch.no_grad():
                z0 = eps + temb0.unsqueeze(0)
                noisy0 = torch.where(noised1.unsqueeze(-1), z0.to(E.dtype), clean_emb)
                out1 = model(inputs_embeds=torch.cat([noisy0, clean_emb], dim=1), attention_mask=self._plan_mask(shift + a, l, dev),
                             position_ids=concat_pos)
                out1 = self._postprocess_outputs(out1)
                logits1 = out1.logits[:, :l]                                             # belief about token p at index p
                xhat = torch.zeros(b, l, E.shape[1], device=dev, dtype=torch.float32)
                top1 = torch.zeros(b, l, device=dev, dtype=torch.long)
                for i in range(b):
                    pr = logits1[i, :, :V].float().softmax(-1)
                    xhat[i] = pr @ E[:V].float()
                    top1[i] = pr.argmax(-1)
                belief_acc = (top1 == labels)[use_self].float().mean().item()
                del out1, logits1

        # ---- pass 2 (grad): heads = real tokens, carried slots = own belief at strength t, fresh slots = eps ----
        tt = t_pos.unsqueeze(-1)
        x_src = torch.where(use_self.unsqueeze(-1), xhat, clean_emb.float()) if xhat is not None else clean_emb.float()
        z = tt * x_src + (1 - tt) * eps
        z = z + proj(flow_time_features(t_pos).to(proj.weight.dtype)).float()
        noisy_emb = torch.where(noised2.unsqueeze(-1), z.to(E.dtype), clean_emb)
        outputs = model(inputs_embeds=torch.cat([noisy_emb, clean_emb], dim=1), attention_mask=self._plan_mask(shift, l, dev),
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
                lab = labels[masked]
                pred_n = noisy_logits[masked].argmax(-1) == lab
                pred_c = clean_logits[masked].argmax(-1) == lab
                headpred = (depth2 == 1)[masked]                 # token p predicted by the head row p-1 (plan-aware AR)
                sel_self = use_self[masked]
                sel_fresh = (~carried & noised2)[masked]
                def bucket(lo, hi):
                    sel = (tm >= lo) & (tm < hi) & ~headpred
                    return f"{noisy_ce[sel].mean().item():.3f}(n={int(sel.sum())})" if sel.any() else "-"
                def acc(v, sel):
                    return f"{v[sel].float().mean().item():.3f}(n={int(sel.sum())})" if sel.any() else "-"
                print(f"[plan] step {self._flow_step} HEAD noisy={acc(pred_n, headpred)} clean={acc(pred_c, headpred)} "
                      f"ce_head={noisy_ce[headpred].mean().item() if headpred.any() else float('nan'):.3f}/{clean_ce[headpred].mean().item() if headpred.any() else float('nan'):.3f} | "
                      f"belief_acc={belief_acc if belief_acc is None else round(belief_acc, 3)} acc2 self={acc(pred_n, sel_self)} fresh={acc(pred_n, sel_fresh)} | "
                      f"ce t=0:{bucket(0, 1e-6)} t<.3:{bucket(1e-6, .3)} t<.6:{bucket(.3, .6)} t>=.6:{bucket(.6, 1.01)} "
                      f"clean_ce={clean_ce.mean().item():.3f} |time_proj|={proj.weight.float().norm().item():.3f}", flush=True)
        outputs.logits = noisy_logits
        return (loss, outputs) if return_outputs else loss
