"""
Stage 2: MARS Training (with SFT loss).
Trains the model to predict masked blocks with auxiliary AR loss on clean stream.
Loss = CE(noisy_logits, labels) + ar_weight * CE(clean_logits, labels)

Usage:
    accelerate launch --config_file configs/zero2.yaml --num_processes 8 \
        train/train_mars.py \
        --model_name_or_path models/ar_sft/checkpoint-final \
        --output_dir models/mars_blk4 \
        --dataset_args allenai/Dolci-Instruct-SFT \
        --num_train_epochs 5 \
        --learning_rate 5e-6 \
        --per_device_train_batch_size 48 \
        --block_size 4 \
        --right_shift_logits \
        --ar_weight 1.0 \
        --max_length 512 \
        --bf16
"""

import os
from dataclasses import dataclass, field
from functools import partial

import torch
if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):  # 10-11: exclude the cuDNN SDPA backend while chasing sticky 'unspecified launch failure's on H100
    torch.backends.cuda.enable_cudnn_sdp(False)
import transformers
import accelerate

import dllm
from dllm.pipelines.a2d import A2DQwen2Config, A2DQwen3Config, A2DQwen3InstructConfig, A2DOuroConfig
from mars.trainers import MARSTrainer

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "models/ar_sft/checkpoint-final"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "allenai/Dolci-Instruct-SFT"
    max_length: int = 512
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )
    agentic_sft: bool = field(
        default=False,
        metadata={
            "help": "Use the agentic multi-turn map fn (passes `tools` to the chat "
            "template and supervises assistant tokens across all turns). Use this "
            "for tool-call trajectory data like Gen-Verse/Open-AgentRL-SFT-3K."
        },
    )
    reasoning_sft: bool = field(
        default=False,
        metadata={
            "help": "Use the reasoning map fn (conversations with separate "
            "`reasoning_content` rendered into <think>...</think>) + MARSReasoningTrainer "
            "(MARS noisy loss only on reasoning tokens, AR loss on all supervised). "
            "For data like stepfun-ai/Step-3.5-Flash-SFT."
        },
    )


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/mars_blk4"
    group_by_length: bool = True
    num_train_epochs: int = 5
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 48
    per_device_eval_batch_size: int = 16
    block_size: int = 4
    right_shift_logits: bool = False
    ar_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for AR loss on clean stream. 0 = pure mask, 1 = equal weight."},
    )
    noisy_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the noisy/mask stream. 0 = pure-AR SFT (standard training; mask embedding untouched)."},
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    base_config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True,
    )
    a2d_config_map = {
        "qwen2": A2DQwen2Config,
        "qwen3": A2DQwen3InstructConfig,
        # already-A2D checkpoints (e.g. released MARS models): continue MARS training directly
        "a2d-qwen2": A2DQwen2Config,
        "a2d-qwen3": A2DQwen3Config,
        "a2d-qwen3-it": A2DQwen3InstructConfig,
        "ouro": A2DOuroConfig,
        "a2d-ouro": A2DOuroConfig,
    }
    A2DConfigClass = a2d_config_map.get(base_config.model_type)
    if A2DConfigClass is None:
        raise ValueError(f"Unsupported model type: {base_config.model_type}. Supported: {list(a2d_config_map.keys())}")
    config = A2DConfigClass.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True,
    )
    config.model_type = A2DConfigClass.model_type

    model = dllm.utils.get_model(model_args=model_args, config=config)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args, config=config)
    if len(tokenizer) > model.config.vocab_size:                 # e.g. Ouro: the added <|mask|> needs a new embedding / lm_head row
        # MARS_VOCAB_PAD=64 pads the new vocab to a multiple of 64 (10-12: 49153 rows = unaligned bf16 GEMM/embedding dims, a
        # suspect for the sticky CUDA faults seen only in stage 2); the padded rows are never targets, decoders slice logits to len(tokenizer)
        pad = int(os.environ.get("MARS_VOCAB_PAD", "0") or 0) or None
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=pad); model.config.vocab_size = model.get_input_embeddings().weight.shape[0]
        logger.info(f"resized token embeddings to {model.config.vocab_size} (len(tokenizer)={len(tokenizer)}, pad_to_multiple_of={pad})")
    # MARS-Ouro v2: mid-loop injection (MARS_TTT = probability per block); the wrapper builds the plan from the [noisy | clean] layout
    if hasattr(model, "injection_plan"):
        model.ttt_prob = float(os.environ.get("MARS_TTT", "0") or 0); model.block_size = training_args.block_size
        model.ttt_mode = os.environ.get("MARS_TTT_MODE", "step"); model.ttt_padd = float(os.environ.get("MARS_TTT_PADD", "0.5") or 0.5)   # 'stair' = per-row geometric staircase (10-16)
        model.mask_token_id = tokenizer.convert_tokens_to_ids("<|mask|>") if "<|mask|>" in tokenizer.get_vocab() else None
        model.readout_mode = os.environ.get("MARS_READOUT", "mix"); model.readout_depth = int(os.environ.get("MARS_READOUT_DEPTH", "4") or 4)   # 10-19: 'split' = per-row decoder-aligned readout (see _split_readout)
        model.readout_tl = int(os.environ.get("MARS_READOUT_TL", "0") or 0)                 # 10-20: deep rows read at loop k+TL-1 (0 = last loop)
        model.model.loop_detach = int(os.environ.get("MARS_LOOP_DETACH", "0") or 0)        # 10-20: truncated backprop across loops
        logger.info(f"readout mode: {model.readout_mode} (depth {model.readout_depth}, tl {model.readout_tl}); loop_detach {model.model.loop_detach}")
        logger.info(f"mid-loop injection: ttt_prob={model.ttt_prob} block_size={model.block_size} mask_id={model.mask_token_id}")

    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        if not data_args.load_preprocessed_data:
            if data_args.reasoning_sft:
                from mars.data import reasoning_sft_map_fn
                base_map_fn = reasoning_sft_map_fn
            elif data_args.agentic_sft:
                from mars.data import agentic_sft_map_fn
                base_map_fn = agentic_sft_map_fn
            else:
                base_map_fn = dllm.utils.default_mdlm_sft_map_fn
            map_fn = partial(
                base_map_fn,
                tokenizer=tokenizer,
                mask_prompt_loss=data_args.mask_prompt_loss,
            )
            dataset = dataset.map(
                map_fn,
                num_proc=data_args.num_proc,
                desc="Mapping dataset to SFT format",
            )
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    accelerate.PartialState().wait_for_everyone()
    logger.info("Start MARS training (with SFT loss)...")
    if data_args.reasoning_sft:
        from mars.trainers import MARSReasoningTrainer
        TrainerClass = MARSReasoningTrainer
        logger.info("Using MARSReasoningTrainer (MARS noisy loss on reasoning tokens only).")
    else:
        TrainerClass = MARSTrainer
    trainer = TrainerClass(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        block_size=training_args.block_size,
        right_shift_logits=training_args.right_shift_logits,
        ar_weight=training_args.ar_weight,
        noisy_weight=training_args.noisy_weight,
        consist_weight=float(os.environ.get("MARS_CONSIST", "0") or 0),      # MARS-Ouro v3: self-consistency KL (clean stream teaches noisy/injected rows)
        consist_tau=float(os.environ.get("MARS_CONSIST_TAU", "1") or 1),
        data_collator=(
            dllm.core.trainers.bd3lm.AppendEOSBlockWrapper(
                transformers.DataCollatorForSeq2Seq(
                    tokenizer,
                    return_tensors="pt",
                    padding=True,
                ),
                block_size=training_args.block_size,
            )
        ),
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    final = os.path.join(training_args.output_dir, "checkpoint-final")
    trainer.save_model(final)
    # tokenizer files are written by ONE rank: with 64 ranks racing on chat_template.jinja the losers raised
    # FileExistsError, the process group died mid-save and checkpoint-final had no weights (72B, 2026-10-03)
    if trainer.accelerator.is_main_process:
        trainer.processing_class.save_pretrained(final)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
