"""A2D variant of Qwen2.5-VL for MARS training.

Same recipe as ``a2d/models/qwen2``: the *text* decoder accepts a caller-built
attention mask (4D tensor or flex-attention BlockMask) verbatim instead of
constructing a causal mask, so the MARS ``[noisy | clean]`` dual-stream mask can
be injected. Everything else — vision tower, patch merger, mRoPE — is inherited
unchanged from Qwen2.5-VL.

The vision tower never sees masked tokens: images live in the (unmasked) prompt
region of both streams, and the MARS trainer scatters the (frozen) vision
features into the embeddings of both streams itself.
"""

from typing import Optional, Union

import torch
from torch import nn

import transformers
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLTextModel,
)

if transformers.utils.is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
else:
    BlockMask = torch.Tensor


class A2DQwen25VLConfig(Qwen2_5_VLConfig):
    model_type = "a2d-qwen2_5_vl"  # <- NEW model_type


class A2DQwen25VLTextModel(Qwen2_5_VLTextModel):

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        # -------------------------------------------------------------
        # A2D: caller-provided mask, no causal mask construction.
        # 4D tensor / BlockMask -> used verbatim; 2D padding mask / None ->
        # bidirectional padding-only mask (matches a2d-qwen2 behavior).
        # -------------------------------------------------------------
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long
                )
            if not (
                isinstance(attention_mask, BlockMask)
                or (isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4)
            ):
                attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)
            causal_mask_mapping = {"full_attention": attention_mask}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = attention_mask
        # -------------------------------------------------------------

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class A2DQwen25VLModel(Qwen2_5_VLModel):
    config: A2DQwen25VLConfig

    def __init__(self, config):
        # Mirror Qwen2_5_VLModel.__init__ but swap in the A2D text decoder.
        transformers.modeling_utils.PreTrainedModel.__init__(self, config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.language_model = A2DQwen25VLTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()


class A2DQwen25VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    config: A2DQwen25VLConfig

    def __init__(self, config):
        transformers.modeling_utils.PreTrainedModel.__init__(self, config)
        self.model = A2DQwen25VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()


transformers.AutoConfig.register("a2d-qwen2_5_vl", A2DQwen25VLConfig)
transformers.AutoModel.register(A2DQwen25VLConfig, A2DQwen25VLForConditionalGeneration)
transformers.AutoModelForVision2Seq.register(A2DQwen25VLConfig, A2DQwen25VLForConditionalGeneration)
