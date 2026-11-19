"""train_latelat.py -- K-pass late-lattice fine-tune of Ouro (DESIGN.md); launcher: launch_latelat.sh.

Data = the stage-1 / v2 / train pipeline unchanged (dllm.data.load_sft_dataset + default_mdlm_sft_map_fn with mask_prompt_loss +
post_process_dataset: drop prompt_len > max_length, right-truncate to max_length). Collator = DataCollatorForSeq2Seq (pad = EOS, labels
-100), padded to a multiple of 16. Model = LatticeOuroLM (lattice_model.py) from the staged init (v2 = ouro26_1p1_ttt/mars_blk16), sdpa,
bf16. Objective / K,B schedule / logging: lattice_trainer.py. Placeholder id = the tokenizer's <|mask|> (v2: 49152, the MARS [P]).
"""
import hashlib, json, os, sys, time
from dataclasses import asdict, dataclass, field
from functools import partial

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))          # train/late_lattice/, MARS/

import torch
if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):  # as train_late.py: no cuDNN SDPA backend on H100
    torch.backends.cuda.enable_cudnn_sdp(False)
import accelerate
import transformers

import dllm
from lattice_model import LatticeOuroLM
from lattice_trainer import LatticeTrainer, StopAtStep, parse_spec

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "/tmp/ouro_latelat_init"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "/tmp/ouro_latelat_data"
    max_length: int = 1024
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = True
    subset_n: int = field(default=0, metadata={"help": "smoke only: keep the first N processed rows"})
    pad_to_multiple_of: int = 16


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "lat_run"
    group_by_length: bool = True
    num_train_epochs: float = 1
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 1
    eval_strategy: str = "no"
    save_only_model: bool = False
    lat_k_choices: str = field(default="1,2,3", metadata={"help": "block passes K, drawn per micro-batch"})
    lat_k_weights: str = field(default="1,2,1", metadata={"help": "weights for lat_k_choices"})
    lat_b_choices: str = field(default="4,8", metadata={"help": "proposals B (dense latt:B), drawn per micro-batch"})
    lat_b_weights: str = field(default="", metadata={"help": "weights for lat_b_choices (default uniform)"})
    lat_alpha: float = field(default=0.8, metadata={"help": "per-offset acceptance prob. of the step-length sampler"})
    lat_extra_max: int = field(default=1024, metadata={"help": "max extra rows (non-joined slots + placeholders) per sequence"})
    lat_extra_frac: float = field(default=1.0, metadata={"help": "extra rows <= frac x valid chain rows"})
    lat_tok_loss: str = field(default="kl", metadata={"help": "token-slot loss: kl (to detached stream 1) | ce"})
    lat_w1: float = 1.0
    lat_w_tok: float = 1.0
    lat_w_ph: float = 1.0
    stop_at_step: int = field(default=0, metadata={"help": "stop (and save) at this step while the LR schedule spans max_steps"})


def md5(p):
    try: return hashlib.md5(open(p, "rb").read()).hexdigest()
    except Exception: return None


def load_data(data_args, tokenizer):
    dataset = dllm.data.load_sft_dataset(data_args.dataset_args, load_preprocessed_data=data_args.load_preprocessed_data)
    if not data_args.load_preprocessed_data:
        dataset = dataset.map(partial(dllm.utils.default_mdlm_sft_map_fn, tokenizer=tokenizer, mask_prompt_loss=data_args.mask_prompt_loss),
                              num_proc=data_args.num_proc, desc="Mapping dataset to SFT format")
    return dllm.utils.post_process_dataset(dataset, data_args)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)
    ks, kw = parse_spec(training_args.lat_k_choices, training_args.lat_k_weights)
    bs, bw = parse_spec(training_args.lat_b_choices, training_args.lat_b_weights)
    assert training_args.lat_tok_loss in ("kl", "ce"), training_args.lat_tok_loss

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.padding_side == "right", tokenizer.padding_side
    dev = {"": accelerate.PartialState().local_process_index} if torch.cuda.is_available() else None
    model = LatticeOuroLM.from_pretrained(model_args.model_name_or_path, dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
                                          attn_implementation="sdpa", device_map=dev)
    assert model.config._attn_implementation == "sdpa"
    T = int(model.model.total_ut_steps); assert all(1 <= k <= T for k in ks), (ks, T)
    assert len(tokenizer) <= model.config.vocab_size, (len(tokenizer), model.config.vocab_size)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>"); stop_ids = tuple(sorted({int(tokenizer.eos_token_id), int(im_end)}))
    mask_id = tokenizer.convert_tokens_to_ids("<|mask|>")
    assert isinstance(mask_id, int) and mask_id != tokenizer.unk_token_id and 0 <= mask_id < model.config.vocab_size, mask_id
    lat_cfg = dict(T=T, stop_ids=stop_ids, mask_id=int(mask_id), alpha=training_args.lat_alpha, extra_max=training_args.lat_extra_max,
                   extra_frac=training_args.lat_extra_frac, tok_loss=training_args.lat_tok_loss, w1=training_args.lat_w1,
                   w_tok=training_args.lat_w_tok, w_ph=training_args.lat_w_ph)
    logger.info(f"late lattice: T={T} K~{dict(zip(ks, kw))} B~{dict(zip(bs, bw))} cfg={lat_cfg} vocab {model.config.vocab_size} len(tok) {len(tokenizer)}")

    with accelerate.PartialState().local_main_process_first():
        dataset = load_data(data_args, tokenizer)
        if data_args.subset_n:
            dataset["train"] = dataset["train"].select(range(min(data_args.subset_n, len(dataset["train"]))))
    logger.info(f"train rows: {len(dataset['train'])}")
    accelerate.PartialState().wait_for_everyone()

    callbacks = []
    if training_args.stop_at_step and training_args.stop_at_step < (training_args.max_steps or 10 ** 12):
        callbacks.append(StopAtStep(training_args.stop_at_step))
    trainer = LatticeTrainer(
        model=model, args=training_args, train_dataset=dataset["train"], processing_class=tokenizer,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True, pad_to_multiple_of=data_args.pad_to_multiple_of),
        callbacks=callbacks, k_choices=ks, k_weights=kw, b_choices=bs, b_weights=bw, lat_cfg=lat_cfg, lat_seed=training_args.seed)
    if trainer.accelerator.is_main_process:                                                # provenance (recipes come from run artifacts, not comments)
        os.makedirs(training_args.output_dir, exist_ok=True)
        man = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), argv=sys.argv, world=trainer.accelerator.num_processes,
                   model_args=asdict(model_args), data_args=asdict(data_args), lat=dict(K=ks, Kw=kw, B=bs, Bw=bw, cfg=lat_cfg, T=T,
                   stop_at_step=training_args.stop_at_step), train_rows=len(dataset["train"]),
                   lr=training_args.learning_rate, sched=str(training_args.lr_scheduler_type), warmup=training_args.warmup_ratio,
                   max_steps=training_args.max_steps, ga=training_args.gradient_accumulation_steps, pdb=training_args.per_device_train_batch_size,
                   code_md5={f: md5(os.path.join(HERE, f)) for f in ("lattice_model.py", "lattice_trainer.py", "train_latelat.py", "launch_latelat.sh")})
        p = os.path.join(training_args.output_dir, f"lat_manifest_{time.strftime('%m%d_%H%M%S')}.json")
        json.dump(man, open(p + ".tmp", "w"), indent=1, default=str); os.replace(p + ".tmp", p)
    final = os.path.join(training_args.output_dir, "checkpoint-final")
    extend = os.path.exists(final)
    accelerate.PartialState().wait_for_everyone()
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    if extend: final = f"{final}-s{trainer.state.global_step}"                             # never overwrite an earlier final
    trainer.save_model(final)
    if trainer.accelerator.is_main_process:
        trainer.processing_class.save_pretrained(final)
        print(f"[lat] final weights -> {final} (global_step {trainer.state.global_step})", flush=True)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
