"""
Stage 1: Standard Autoregressive SFT.
Pure next-token prediction with causal attention.

Usage:
    accelerate launch --config_file configs/zero2.yaml --num_processes 8 \
        train/train_ar_sft.py \
        --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
        --output_dir models/ar_sft \
        --dataset_args allenai/Dolci-Instruct-SFT \
        --num_train_epochs 5 \
        --learning_rate 5e-6 \
        --per_device_train_batch_size 48 \
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

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "allenai/Dolci-Instruct-SFT"
    max_length: int = 512
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/ar_sft"
    num_train_epochs: int = 5
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 48
    per_device_eval_batch_size: int = 16


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:  # Ouro ships without a pad token; the Seq2Seq collator pads with it (labels stay -100)
        tokenizer.pad_token = tokenizer.eos_token

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
                desc="Mapping dataset to SFT format",
            )
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    accelerate.PartialState().wait_for_everyone()
    logger.info("Start standard AR SFT training...")

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        processing_class=tokenizer,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, return_tensors="pt", padding=True
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
