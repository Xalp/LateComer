import os as _os, sys as _sys
_REPO_TRAIN = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', 'train'))
if _REPO_TRAIN not in _sys.path:
    _sys.path.insert(0, _REPO_TRAIN)
"""Stage 2: MARS + Late-Entry Block Training for Huginn-0125 (A2DHuginn).

Follows the exact paper Stage-2 recipe on allenai/Dolci-Instruct-SFT:
  - Initialized from Stage-1 Huginn AR SFT checkpoint
  - Adds <|mask|> special token and pads embeddings to multiple of 64 (65600) with
    synchronized vocab_size & padded_vocab_size and tied lm_head
  - 1 epoch, block_size=16, max_length=1024, lr=1e-5, warmup_ratio=0.03,
    right_shift_logits=True, ar_weight=1.0, bf16, global batch 256
  - MARS_TTT=0.5: on half of supervised 16-token noisy blocks, samples K < 32 and
    verified candidate prefix length j, feeds [clean prefix | <|mask|> suffix] through
    Prelude, resets Core state to s_0 at step 32-K, and executes only the final K Core
    steps over full 32-step clean context KV before Coda readout.
"""

import os
from dataclasses import dataclass, field
from functools import partial

import torch
if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)

import accelerate
import transformers

import dllm
from mars.trainers import MARSTrainer
from huginn.model.modeling_a2d_huginn import A2DHuginnConfig, A2DHuginnForCausalLM

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "models/hg_ar_sft/checkpoint-final"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "allenai/Dolci-Instruct-SFT"
    max_length: int = 1024
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(default=True)


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/hg_mars_blk16"
    group_by_length: bool = True
    num_train_epochs: float = 1.0
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    block_size: int = 16
    right_shift_logits: bool = True
    ar_weight: float = field(default=1.0)
    noisy_weight: float = field(default=1.0)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    config = A2DHuginnConfig.from_pretrained(model_args.model_name_or_path)
    config.model_type = A2DHuginnConfig.model_type
    config._attn_implementation = "sdpa"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "<|mask|>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
    else:
        tokenizer.mask_token = "<|mask|>"
    tokenizer.eot_token = "<|end_turn|>"
    tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids("<|end_turn|>")

    device_map = (
        {"": accelerate.PartialState().local_process_index}
        if not transformers.modeling_utils.is_deepspeed_zero3_enabled() and torch.cuda.is_available()
        else None
    )
    model = A2DHuginnForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    if len(tokenizer) > model.config.vocab_size:
        pad = int(os.environ.get("MARS_VOCAB_PAD", "64") or 64) or None
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=pad)
        logger.info(
            f"[hg-s2] resized embeddings to vocab_size={model.config.vocab_size}, "
            f"padded_vocab_size={model.config.padded_vocab_size} (len(tokenizer)={len(tokenizer)}, pad={pad})"
        )

    model.total_steps = int(os.environ.get("HG_TOTAL_STEPS", str(config.mean_recurrence)))
    model.grad_steps = int(os.environ.get("HG_GRAD_STEPS", str(config.mean_backprop_depth)))
    model.grad_ckpt_core = bool(int(os.environ.get("HG_GRAD_CKPT_CORE", "0") or 0))
    model.ttt_prob = float(os.environ.get("MARS_TTT", "0.5") or 0.0)
    model.firstt_prob = float(os.environ.get("MARS_FIRSTT", "0.0") or 0.0)
    model.block_size = int(training_args.block_size)
    model.mask_token_id = int(tokenizer.mask_token_id)

    grid_env = os.environ.get("HG_LATE_K_GRID", "").strip()
    if grid_env:
        model.late_k_grid = tuple(int(x) for x in grid_env.split(",") if x.strip())

    logger.info(
        f"[hg-s2] configured Late-Entry MARS: total_steps={model.total_steps}, grad_steps={model.grad_steps}, "
        f"ttt_prob={model.ttt_prob}, firstt_prob={model.firstt_prob}, block_size={model.block_size}, "
        f"mask_token_id={model.mask_token_id}, late_k_grid={model.late_k_grid}"
    )

    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        if not data_args.load_preprocessed_data:
            map_fn = partial(
                dllm.utils.default_mdlm_sft_map_fn,
                tokenizer=tokenizer,
                mask_prompt_loss=data_args.mask_prompt_loss,
            )
            dataset = dataset.map(
                map_fn,
                num_proc=data_args.num_proc,
                desc="Mapping Dolci-Instruct-SFT for Huginn Stage 2",
            )
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    accelerate.PartialState().wait_for_everyone()
    logger.info("Starting Huginn Stage 2 MARS + Late-Entry training...")

    trainer = MARSTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        block_size=training_args.block_size,
        right_shift_logits=training_args.right_shift_logits,
        ar_weight=training_args.ar_weight,
        noisy_weight=training_args.noisy_weight,
        consist_weight=float(os.environ.get("MARS_CONSIST", "0") or 0),
        consist_tau=float(os.environ.get("MARS_CONSIST_TAU", "1") or 1),
        data_collator=dllm.core.trainers.bd3lm.AppendEOSBlockWrapper(
            transformers.DataCollatorForSeq2Seq(
                tokenizer,
                return_tensors="pt",
                padding=True,
            ),
            block_size=training_args.block_size,
        ),
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    final = os.path.join(training_args.output_dir, "checkpoint-final")
    trainer.save_model(final)
    if trainer.accelerator.is_main_process:
        trainer.processing_class.save_pretrained(final)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
