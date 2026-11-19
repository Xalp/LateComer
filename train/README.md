# Two-Stage Fine-Tuning Recipe (`Section 2.1, Appendices A-B`)

Complete training scripts and vendored loop-block packages (`dllm/`, `mars/`) for adapting looped language models to **late-entry** and **matched early-exit** block decoding without adding inference parameters.

## Files

| Path | Description |
| :--- | :--- |
| `train_ar_sft.py` | **Stage 1:** Full-depth ($T = 4$) autoregressive SFT on `allenai/Dolci-Instruct-SFT` |
| `train_mars.py` | **Stage 2 (Late Entry):** Dual-stream `[noisy | clean]` mask-placeholder block fine-tuning where 50% of 16-token blocks enter at a uniformly sampled pass $E \in \{2, \dots, T\}$ over clean $T$-pass context |
| `train_mars_firstt.py` | **Stage 2 (Matched Early Exit):** Identical data, hyperparameters, FLOPs, and mask-placeholder block layout, reversing only the kept-pass pattern to the first $K \in \{1, \dots, T-1\}$ passes |
| `late_lattice/` | Optional Stage-3 teacher-forced $K$-pass late-lattice KL distillation (`train_latelat.py`, `DESIGN.md`) |
| `dllm/` | Vendored `A2DOuro` looped Transformer model wrapper, data loader, and SFT collators |
| `mars/` | Vendored dual-stream block attention mask builder (`attention_mask.py`) and `MARSTrainer` |

## Reproducing Ouro-2.6B / Ouro-1.4B Checkpoints

Both stages train for 1 epoch on `allenai/Dolci-Instruct-SFT` (`max_length = 1024`, `bf16`, global batch size `256`, cosine LR peaking at `1e-5` with `3%` warmup):

```bash
# Stage 1: Full-depth SFT
accelerate launch --num_processes 8 train/train_ar_sft.py \
  --model_name_or_path ByteDance/Ouro-2.6B \
  --dataset_args allenai/Dolci-Instruct-SFT \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --warmup_ratio 0.03 \
  --max_length 1024 \
  --bf16 \
  --output_dir checkpoints/ouro-2.6b-stage1

# Stage 2: Late-Entry Block Fine-Tuning (-> Xalphinions/ouro-2.6b-late-entry)
accelerate launch --num_processes 8 train/train_mars.py \
  --model_name_or_path checkpoints/ouro-2.6b-stage1 \
  --dataset_args allenai/Dolci-Instruct-SFT \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --warmup_ratio 0.03 \
  --max_length 1024 \
  --block_size 16 \
  --bf16 \
  --output_dir checkpoints/ouro-2.6b-late-entry

# Stage 2 (Matched Control): Early-Exit Block Fine-Tuning (-> Xalphinions/ouro-2.6b-early-exit)
accelerate launch --num_processes 8 train/train_mars_firstt.py \
  --model_name_or_path checkpoints/ouro-2.6b-stage1 \
  --dataset_args allenai/Dolci-Instruct-SFT \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --warmup_ratio 0.03 \
  --max_length 1024 \
  --block_size 16 \
  --bf16 \
  --output_dir checkpoints/ouro-2.6b-early-exit
```

For **Huginn-3.5B ($T = 32$ Prelude-Core-Coda)** Stage-1/Stage-2 training, see [`../huginn/README.md`](../huginn/README.md).
