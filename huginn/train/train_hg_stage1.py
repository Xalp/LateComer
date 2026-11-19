import os as _os, sys as _sys
_REPO_TRAIN = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', 'train'))
if _REPO_TRAIN not in _sys.path:
    _sys.path.insert(0, _REPO_TRAIN)
"""Stage 1: Standard Autoregressive SFT for Huginn-0125 (Raven / A2DHuginn).

Follows the exact paper Stage-1 recipe on allenai/Dolci-Instruct-SFT:
  - 1 epoch, max_length=1024, lr=1e-5, warmup_ratio=0.03, bf16, global batch 256
  - Fixed T=32 recurrence steps (24 no-grad + 8 grad steps matching Huginn's native
    mean_recurrence=32, mean_backprop_depth=8)
  - Proper shifted CausalLM next-token cross-entropy loss
  - Fast Arrow-offset LengthGroupedSampler (~0.5s instead of ~25min on 1.89M rows)
"""

import os
from dataclasses import dataclass, field
from functools import partial

import torch
if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)

import accelerate
import transformers
from transformers.trainer_pt_utils import LengthGroupedSampler

import dllm
from huginn.model.modeling_a2d_huginn import A2DHuginnConfig, A2DHuginnForCausalLM

logger = dllm.utils.get_default_logger(__name__)


def _arrow_lengths(ds, column: str = "input_ids"):
    try:
        import datasets as hfds
        import pyarrow.compute as pc
        if not isinstance(ds, hfds.Dataset) or column not in ds.column_names:
            return None
        tbl = ds.data.table if hasattr(ds.data, "table") else ds.data
        lens = pc.list_value_length(tbl.column(column)).to_numpy(zero_copy_only=False)
        if ds._indices is not None:
            it = ds._indices.table if hasattr(ds._indices, "table") else ds._indices
            lens = lens[it.column(0).to_numpy()]
        if len(lens) != len(ds):
            return None
        return lens.tolist()
    except Exception as exc:
        print(f"[hg-s1] arrow length shortcut failed ({exc!r}); falling back to default sampler", flush=True)
        return None


class FastGroupedTrainer(transformers.Trainer):
    def _get_train_sampler(self, train_dataset=None):
        ds = train_dataset if train_dataset is not None else self.train_dataset
        lengths = _arrow_lengths(ds, "input_ids") if (self.args.group_by_length and ds is not None) else None
        if lengths is None:
            return super()._get_train_sampler(train_dataset)
        model_input_name = self.processing_class.model_input_names[0] if self.processing_class is not None else None
        print(f"[hg-s1] group_by_length: {len(lengths)} lengths read from Arrow offsets", flush=True)
        return LengthGroupedSampler(
            self.args.train_batch_size * self.args.gradient_accumulation_steps,
            dataset=ds,
            lengths=lengths,
            model_input_name=model_input_name,
        )


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "tomg-group-umd/huginn-0125"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "allenai/Dolci-Instruct-SFT"
    max_length: int = 1024
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(default=True)


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/hg_ar_sft"
    num_train_epochs: float = 1.0
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    config = A2DHuginnConfig.from_pretrained(model_args.model_name_or_path)
    config.model_type = A2DHuginnConfig.model_type
    config._attn_implementation = "sdpa"

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
    model.total_steps = int(os.environ.get("HG_TOTAL_STEPS", str(config.mean_recurrence)))
    model.grad_steps = int(os.environ.get("HG_GRAD_STEPS", str(config.mean_backprop_depth)))
    model.grad_ckpt_core = bool(int(os.environ.get("HG_GRAD_CKPT_CORE", "0") or 0))
    model.ttt_prob = 0.0
    model.firstt_prob = 0.0
    logger.info(
        f"[hg-s1] loaded A2DHuginnForCausalLM: total_steps={model.total_steps}, "
        f"grad_steps={model.grad_steps}, grad_ckpt_core={model.grad_ckpt_core}, vocab={model.config.vocab_size}"
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
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
                desc="Mapping Dolci-Instruct-SFT for Huginn Stage 1",
            )
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    accelerate.PartialState().wait_for_everyone()
    logger.info("Starting Huginn Stage 1 AR SFT...")

    trainer = FastGroupedTrainer(
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
    if trainer.accelerator.is_main_process:
        trainer.processing_class.save_pretrained(final)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
