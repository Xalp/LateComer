"""A2D (MARS) wrapper for ByteDance's looped Ouro model (2026-10-09).

Same contract as A2DQwen2: the trainer passes a [1,1,2l,2l] MARS attention mask (bool, sdpa) and concatenated
[noisy | clean] input ids; the model must use that mask as-is (no causal mask creation). Ouro's own forward already accepts
a dict mapping attention types to masks, so the wrapper converts a 2D/4D tensor into {"full_attention": mask}.
Logits: at training time Ouro trains the gate-weighted mixture of every recurrent step's logits (its own loss path), so the
wrapper returns that mixture (use_weighted_exit) whenever no explicit exit rule is requested — every loop's mask-row
predictions get trained, which is what the per-loop previews need.
"""
from typing import Optional

import torch
import transformers
from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

from .configuration_ouro import OuroConfig
from .modeling_ouro import OuroForCausalLM, OuroModel, OuroPreTrainedModel
from torch import nn


class A2DOuroConfig(OuroConfig):
    model_type = "a2d-ouro"


class A2DOuroModel(OuroModel):
    config_class = A2DOuroConfig

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values: Optional[Cache] = None,
                inputs_embeds=None, use_cache=None, cache_position=None, inject_loop=None, inject_rows=None, inject_embeds=None, inject_k=None, **kwargs):
        """inject_loop / inject_rows [B,T] bool / inject_embeds [B,T,H]: before loop `inject_loop` (1-based, >= 2) the hidden state of the
        selected rows is replaced by the given embeddings -- training-time simulation of the streaming pipeline, where a mask row's
        left neighbours turn into committed tokens in the middle of its loops (MARS-Ouro v2 'mid-loop injection', 2026-10-14).
        inject_k [B,T] long (0 = never): PER-ROW injection loop -- row i is replaced before loop inject_k[i] (the 'staircase' plan, 10-16:
        the prefix grows loop by loop as in decoding)."""
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4:               # MARS / probe masks are passed through as-is
            attention_mask = {"full_attention": attention_mask, "sliding_attention": attention_mask}
        if use_cache is None: use_cache = False
        K = int(getattr(self, "loop_detach", 0) or 0) if self.training else 0    # 10-20: truncated backprop across loops (windows of K loops)
        if inject_loop is None and inject_k is None and not K:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values,
                                   inputs_embeds=inputs_embeds, use_cache=use_cache, cache_position=cache_position, **kwargs)
        # ---- re-implementation of OuroModel.forward's loop with the injection hook (no cache: training only) ----
        from transformers.modeling_outputs import BaseModelOutputWithPast
        if inputs_embeds is None: inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None: cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None: position_ids = cache_position.unsqueeze(0)
        if not isinstance(attention_mask, dict):
            from transformers.masking_utils import create_causal_mask
            mask_kwargs = {"config": self.config, "input_embeds": inputs_embeds, "attention_mask": attention_mask, "cache_position": cache_position, "past_key_values": None, "position_ids": position_ids}
            attention_mask = {"full_attention": create_causal_mask(**mask_kwargs)}
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        hidden_states_list = []; gate_list = []
        for current_ut in range(self.total_ut_steps):
            if K and current_ut > 0 and current_ut % K == 0:
                hidden_states = hidden_states.detach()                   # gradient path <= K loops: the 16-loop weight-shared recurrence exploded grad norms (1e3-1e5 vs 15 at T=4)
            if inject_k is not None:
                sel = inject_k == (current_ut + 1)
                if bool(sel.any()): hidden_states = torch.where(sel.unsqueeze(-1), inject_embeds.to(hidden_states.dtype), hidden_states)
            elif current_ut + 1 == inject_loop and inject_rows is not None:
                hidden_states = torch.where(inject_rows.unsqueeze(-1), inject_embeds.to(hidden_states.dtype), hidden_states)
            for decoder_layer in self.layers[: self.config.num_hidden_layers]:
                hidden_states = decoder_layer(hidden_states, attention_mask=attention_mask[decoder_layer.attention_type], position_ids=position_ids,
                                              past_key_value=None, use_cache=False, cache_position=cache_position, position_embeddings=position_embeddings,
                                              current_ut=current_ut, **kwargs)
            hidden_states = self.norm(hidden_states)
            hidden_states_list.append(hidden_states); gate_list.append(self.early_exit_gate(hidden_states))
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=None), hidden_states_list, gate_list


class A2DOuroLMHeadModel(OuroForCausalLM):
    config_class = A2DOuroConfig

    def __init__(self, config):
        OuroPreTrainedModel.__init__(self, config)
        self.model = A2DOuroModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.chunk_size = getattr(config, "chunk_size", 2)
        self.early_exit_step = getattr(config, "early_exit_step", None)
        self.early_exit_threshold = None          # training: never pick a single step by threshold; the mixture below trains all loops
        self.post_init()

    # explicit signature: HF Trainer keeps only dataset columns named in forward's signature (a *args/**kwargs forward made it drop
    # input_ids/labels -> "No columns in the dataset match the model's forward method signature", 10-11)
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, cache_position=None, logits_to_keep=0, use_weighted_exit=None, exit_at_step=None, exit_threshold=None, **kwargs):
        if use_weighted_exit is None and exit_at_step is None and exit_threshold is None: use_weighted_exit = True
        ttt = getattr(self, "ttt_prob", 0.0)
        if self.training and ttt > 0 and input_ids is not None and "inject_loop" not in kwargs and "inject_k" not in kwargs:
            kwargs.update(self.injection_plan(input_ids, ttt))
        if self.training and getattr(self, "readout_mode", "mix") == "split":
            return self._split_readout(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, inputs_embeds=inputs_embeds, **kwargs)
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values,
                               inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, cache_position=cache_position, logits_to_keep=logits_to_keep,
                               use_weighted_exit=use_weighted_exit, exit_at_step=exit_at_step, exit_threshold=exit_threshold, **kwargs)


    def _split_readout(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, **kwargs):
        """MARS_READOUT=split (10-19): train every row on the readout its decoder will actually use, instead of the gate mixture over
        all T loops. With T_max=16 that 16-loop mixture is 60-80 % untrained deep loops: optimising it diluted the deep rows' gradient
        12 ways and, by step 1200, eroded the 4-loop model itself (AR-4 .87 -> .77) because the 4-loop readout was never trained as
        such any more. Here: rows injected after loop D (readout_depth, 4) -> logits of the LAST loop (what DeepInject reads);
        every other row (clean, never-injected masks, rows injected at k <= D) -> the D-loop gate mixture with loop D taking the
        remainder (exactly the stream's readout at STREAM_T=D and the v2 objective). Gates D+1..T are not used."""
        from transformers.modeling_outputs import CausalLMOutputWithPast
        _, hs, gl = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, inputs_embeds=inputs_embeds, use_cache=False, **kwargs)
        D = min(int(getattr(self, "readout_depth", 4)), len(hs)); rem = None; mix = None
        for i in range(D):
            lam = torch.sigmoid(gl[i].squeeze(-1).float()); rem1 = torch.ones_like(lam) if rem is None else rem
            p = (lam * rem1) if i < D - 1 else rem1; rem = rem1 * (1.0 - lam)
            step = self.lm_head(hs[i]) * p.unsqueeze(-1).to(hs[i].dtype); mix = step if mix is None else mix + step
        inject_k = kwargs.get("inject_k"); inject_rows = kwargs.get("inject_rows"); inject_loop = kwargs.get("inject_loop")
        deep = (inject_k > D) if inject_k is not None else (inject_rows if (inject_rows is not None and inject_loop is not None and inject_loop > D) else None)
        if deep is not None and len(hs) > D and bool(deep.any()):
            TL = int(getattr(self, "readout_tl", 0) or 0); T = len(hs)
            if TL <= 0 or inject_k is None:
                deep_h = hs[-1]                                                              # last loop (recipe t16sL)
            else:                                                                            # loop k+TL-1: the TL-th token loop after injection = what the tl=TL deep decoder reads (10-20)
                li = (inject_k + (TL - 1)).clamp(max=T) - 1                                  # 0-based loop index per row
                stack = torch.stack(hs[D:], 0)                                               # [T-D, B, L, H] (only loops > D can be read here)
                idx = (li - D).clamp(min=0).unsqueeze(0).unsqueeze(-1).expand(1, *li.shape, stack.shape[-1])
                deep_h = torch.gather(stack, 0, idx)[0]
            mix = torch.where(deep.unsqueeze(-1), self.lm_head(deep_h), mix)
        return CausalLMOutputWithPast(logits=mix)

def _injection_plan(self, input_ids, prob):
    """MARS layout [noisy | clean] of the same sequence: for each block of block_size in the noisy half, with probability `prob`, rows get
    their TRUE tokens (clean half) injected mid-loop.
    mode 'step' (default, v2): pick loop k in {2..T} (shared by the batch) and prefix length j in {1..block_size}; the first j rows are
    injected before loop k.
    mode 'stair' (user 10-16, env MARS_TTT_MODE=stair, MARS_TTT_PADD=0.5): the prefix GROWS loop by loop like the decoding stream -- before
    each loop t >= 2, keep flipping a coin with P=padd and inject the next not-yet-injected row of the block on heads, stop on tails
    (P(+1)=.5, P(+2)=.25, ...); injected rows stay. Every row gets its own injection loop (inject_k), the batch shares nothing."""
    B, T2 = input_ids.shape; l = T2 // 2; bs = getattr(self, "block_size", 16); mask_id = getattr(self, "mask_token_id", None)
    noisy, clean = input_ids[:, :l], input_ids[:, l:]; T = self.model.total_ut_steps
    mode = getattr(self, "ttt_mode", "step"); padd = float(getattr(self, "ttt_padd", 0.5))
    if mode == "stair":
        kk = torch.zeros(B, T2, dtype=torch.long, device=input_ids.device)
        for b in range(B):
            for s0 in range(0, l, bs):
                if torch.rand(1).item() >= prob: continue
                c = 0; e = min(s0 + bs, l)
                for t in range(2, T + 1):
                    while s0 + c < e and torch.rand(1).item() < padd:
                        if mask_id is None or noisy[b, s0 + c] == mask_id: kk[b, s0 + c] = t
                        c += 1
        if not bool((kk > 0).any()): return {}
        src = input_ids.clone(); src[:, :l] = clean
        return {"inject_k": kk, "inject_embeds": self.model.embed_tokens(src)}
    rows = torch.zeros(B, T2, dtype=torch.bool, device=input_ids.device)
    k = int(torch.randint(2, T + 1, (1,)).item())
    for b in range(B):
        for s0 in range(0, l, bs):
            if torch.rand(1).item() >= prob: continue
            j = int(torch.randint(1, bs + 1, (1,)).item()); e = min(s0 + j, l)
            seg = rows[b, s0:e]; seg[:] = (noisy[b, s0:e] == mask_id) if mask_id is not None else True
    if not rows.any(): return {}
    src = input_ids.clone(); src[:, :l] = clean                                         # the true token of every noisy position
    emb = self.model.embed_tokens(src)
    return {"inject_loop": k, "inject_rows": rows, "inject_embeds": emb}


A2DOuroLMHeadModel.injection_plan = _injection_plan

transformers.AutoConfig.register("a2d-ouro", A2DOuroConfig)
transformers.AutoModel.register(A2DOuroConfig, A2DOuroLMHeadModel)
transformers.AutoModelForCausalLM.register(A2DOuroConfig, A2DOuroLMHeadModel)
transformers.AutoModelForMaskedLM.register(A2DOuroConfig, A2DOuroLMHeadModel)
