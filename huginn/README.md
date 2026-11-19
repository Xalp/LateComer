# Huginn-3.5B (`T = 32` Prelude-Core-Coda) Late-Entry Adaptation & Decoding

This directory implements two-stage late-entry (and matched early-exit) adaptation, single-token probes, and late-entry block decoding with $\lceil T / K \rceil$-stage pipelined cache completion for **Huginn-3.5B** (`tomg-group-umd/huginn-0125`; Geiping et al., 2025), corresponding to **Section 4.4** and **Appendix H** of the paper. For **Wake Verification ($\tau \in [0, \infty]$)** and **Intra-Model Recurrence Depth Scaling ($T \in \{8, 16, 32, 64\}$)** on Huginn-3.5B (`Sections 4.5–4.6`), see [`../wake_verify/`](../wake_verify/).

## Architecture Overview

Unlike purely looped models (Ouro), Huginn-3.5B wraps a $4$-layer recurrent **Core** block iterated $T = 32$ times via a learned linear input-injection adapter $W_{\mathrm{adap}}$ between a $2$-layer non-recurrent **Prelude** and a $2$-layer non-recurrent **Coda** ($132$ effective Transformer layers per token across $8$ unique parameter layers).

Under late-entry block decoding at budget $K < 32$ and block size $B$:
1. The active $(1 + B)^2$ mask-placeholder block runs the $2$-layer Prelude once (storing exact Prelude KV states and anchors $u_i$ so Prelude is **never** recomputed during catch-up).
2. Candidate rows skip the initial $32 - K$ Core steps, enter directly at step $32 - K + 1$ from Huginn's fixed-seed initialized latent state $s_i^{(32-K)} = s_i^{(0)}$ over a $32$-pass context KV cache, and read out through the $2$-layer Coda ($4 + 4K$ serial layers vs. $132$ at full depth).
3. Over the next $\lceil 32 / K \rceil$ steps, committed token groups advance through true Core steps $1 \to 32$ at $K$ steps per call alongside the active block, fusing Coda cache refresh into the final catch-up call at **zero extra serial forward calls**.

## Layout

| Path | Description |
| :--- | :--- |
| `model/raven_{config,modeling}_minimal.py` | Standalone Huginn-0125 (`RavenForCausalLM`) implementation (no `trust_remote_code` required) |
| `model/modeling_a2d_huginn.py` | `A2DHuginnForCausalLM`: Stage-1 full-depth SFT + Stage-2 MARS late-entry block exposure (`8`-step truncated BPTT) |
| `model/modeling_a2d_huginn_early.py` | `A2DHuginnEarlyExitForCausalLM`: matched early-exit Stage-2 control |
| `train/train_hg_stage1.py` | Stage-1 full-depth ($T = 32$) instruction SFT on `allenai/Dolci-Instruct-SFT` |
| `train/train_hg_stage2.py` | Stage-2 late-entry MARS block fine-tuning ($50\%$ full $T=32$, $50\%$ last $K \in \{2,4,6,8,12,16,24\}$) |
| `train/train_hg_stage2_early.py` | Stage-2 matched early-exit MARS block fine-tuning (first $K$ Core steps) |
| `decode/hg_fast_dec.py` | `HuginnFastEngine`: unified CUDA-graphed single-call paged FlashAttention-3 decoder (`blk:K:B`, `eblk:K:B`, and `pipe` / `p24` / `p16` / `sync` / `none` cache completion) |
| `decode/hg_block_dec.py` | `HuginnBlockDecoder`: eager PyTorch SDPA reference block decoder (`pipe` / `sync` / `none`) |
| `decode/eval_hg_single.py` | Teacher-forced trajectory probe and single-token sequential decoder (`r:K`, `early:32:K`, `late:32:K`) |
| `decode/hg_model_io.py` | Checkpoint loader, deterministic $s_0$ helper, and dynamic RoPE extension beyond `4096` tokens |
| `eval/score_gsm8k.py` | Exact `lm-evaluation-harness` `gsm8k_cot` strict/flexible scorer and paired McNemar sign test |
| `eval/score_hg_formal.py` | Multi-benchmark shard aggregator and gold-verified scorer (`GSM8K`, `IFEval`, `MATH500`, `MBPP`, `BBH`, `MMLU-Pro`) |
| `tests/test_hg_fast_cpu.py` | `128 / 128` CPU exactness gates verifying paged KV tables, RoPE, and `pipe` / `p24` / `p16` / `sync` / `none` / `eblk` equivalence |
| `tests/test_hg_*_cpu.py` | Standalone CPU mathematical exactness and gradient-flow gates (no GPU required) |
| `results/` | Scored full-dataset markdown tables from all formal Huginn-3.5B evaluations |

## Checkpoints ([Hugging Face Collection](https://huggingface.co/collections/Xalphinions/late-entry))

- [`Xalphinions/huginn-3.5b-late-entry`](https://huggingface.co/Xalphinions/huginn-3.5b-late-entry): Two-stage adapted Huginn-3.5B ($T = 32$) for late-entry decoding (`7,406` Stage-2 steps).
- [`Xalphinions/huginn-3.5b-early-exit`](https://huggingface.co/Xalphinions/huginn-3.5b-early-exit): Matched early-exit Stage-2 control (`7,406` Stage-2 steps).

## Quickstart

### 1. Block Decoding (`B = 8`, Unified CUDA-Graphed Paged FA3 Engine)

```bash
python -m huginn.decode.hg_fast_dec \
  --model Xalphinions/huginn-3.5b-late-entry \
  --tasks data/tasks/gsm8k.jsonl \
  --cfgs "blk:32:0,blk:32:8,blk:16:8,blk:12:8,blk:8:8,blk:8:8:p16" \
  --max-new 512 \
  --out-dir outputs/huginn_gsm8k
```

Configuration syntax:
- `blk:32:0` : Full-depth sequential decoding ($T = 32$, batch size `1`, paired speed reference)
- `blk:32:8` : Full-depth block decoding ($K = 32, B = 8$)
- `blk:16:8`, `blk:12:8`, `blk:8:8` : Late-entry block decoding with pipelined $\lceil 32 / K \rceil$-stage cache completion
- `blk:8:8:p16`, `blk:8:8:p24` : Partial-cache approximations completing only the final $16$ (`2` stages, `1`-block lag) or $24$ (`3` stages, `2`-block lag) Core passes (leaving earlier Core passes uncomputed)
- `blk:8:8:sync` / `blk:8:8:none` : CUDA-graphed synchronous repair (`0` lag) and no-catch-up (`never`) ablations
- `eblk:8:8` : Truncated early-exit block decoding (initial $K$ Core passes, depth-$K$ cache)

### 2. Teacher-Forced Probe & Single-Token Curves

```bash
# Teacher-forced agreement & KL probe across K in {1,2,4,6,8,12,16,24,32}
python -m huginn.decode.eval_hg_single \
  --mode probe \
  --model /path/to/huginn-3.5b-late-entry \
  --tasks data/tasks/gsm8k.jsonl \
  --probe-n 100 \
  --out outputs/probe_late.json

# Single-token sequential decoding on full GSM8K (n = 1,319)
python -m huginn.decode.eval_hg_single \
  --mode single \
  --model /path/to/huginn-3.5b-late-entry \
  --tasks data/tasks/gsm8k.jsonl \
  --cfgs "r:32,late:32:16,late:32:12,late:32:8,late:32:4,r:8,r:4" \
  --out-dir outputs/single_gsm8k
```

### 3. Two-Stage Fine-Tuning (`allenai/Dolci-Instruct-SFT`)

Both stages run $1$ epoch at sequence length `1,024`, learning rate `1e-5`, warmup ratio `0.03`, `bf16`, global batch size `256` (`8`-step truncated backpropagation matching Geiping et al., 2025):

```bash
# Stage 1: Full-depth (T = 32) instruction SFT
accelerate launch -m huginn.train.train_hg_stage1 \
  --model_name_or_path tomg-group-umd/huginn-0125 \
  --dataset_args allenai/Dolci-Instruct-SFT \
  --num_train_epochs 1 --learning_rate 1e-5 --warmup_ratio 0.03 \
  --max_length 1024 --bf16 --output_dir checkpoints/hg_stage1

# Stage 2: Late-entry MARS block fine-tuning (use train_hg_stage2_early for matched early exit)
accelerate launch -m huginn.train.train_hg_stage2 \
  --model_name_or_path checkpoints/hg_stage1 \
  --dataset_args allenai/Dolci-Instruct-SFT \
  --num_train_epochs 1 --learning_rate 1e-5 --warmup_ratio 0.03 \
  --block_size 16 --max_length 1024 --bf16 --output_dir checkpoints/hg_stage2_late
```

### 4. CPU Verification Tests

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. python huginn/tests/test_hg_fast_cpu.py
OMP_NUM_THREADS=1 PYTHONPATH=. python huginn/tests/test_hg_eval_cpu.py
OMP_NUM_THREADS=1 PYTHONPATH=. python huginn/tests/test_hg_early_cpu.py
OMP_NUM_THREADS=1 PYTHONPATH=. python huginn/tests/test_hg_sft_cpu.py
```

## Summary of Formal Results (`B = 8`, Single H100 GPU, `8,871` Examples Across Four Capability-Active Benchmarks)

| Benchmark | $n$ | Full Depth ($T=32$) | Lossless $K=32$ Block | Lossy $K=16$ Late Entry | Lossy $K=12$ Late Entry | Lossy $K=8$ Late Entry |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: |
| **GSM8K** | `1,319` | `.393` (`21.7 tok/s`) | `.397` (`3.37×`) | **`.405` (`5.84×`, `+74%`)** | `.395` (`7.27×`, `+116%`) | `.354` (**`9.13×`**, `+171%`; `.369` at **`9.93×`** `p16`) |
| **IFEval** | `541` | `.551` (`21.6 tok/s`) | `.551` (`2.93×`) | **`.566` (`5.24×`, `+79%`)** | `.545` (`6.44×`, `+120%`) | `.542` (**`8.96×`**, `+206%`) |
| **BBH** | `6,511` | `.283` (`22.6 tok/s`) | `.286` (`3.79×`) | **`.287` (`6.65×`, `+75%`)** | `.281` (`8.20×`, `+116%`) | `.274` (**`10.86×`**, `+186%`) |
| **MBPP** | `500` | `.234` (`22.4 tok/s`) | `.236` (`4.46×`) | **`.232` (`7.76×`, `+74%`)** | `.234` (`9.75×`, `+119%`) | `.228` (**`12.41×`**, `+178%`) |

Full-depth $K = 32$ block decoding (`2.93` to `4.46×`) serves as the lossless $32$-pass reference boundary; lossy late entry at $K = 16$ reaches `5.24` to `7.76×` (`+74%` to `+79%` over $K = 32$) with no statistically significant accuracy drop on any benchmark, and $K = 12$ reaches `6.44` to `9.75×` within `0.6` points across all four tasks. At $K = 8$, whereas four-stage full completion (`blk:8:8`, `.354` at `9.13×`) trails CUDA-graphed synchronous repair (`.387` at `2.71×`) by `3.3` points on GSM8K, the two-stage partial-cache approximation (`blk:8:8:p16`, passes `17..32`) narrows this gap to `1.9` points (`.369` at `215.5 tok/s`, `9.93×`, `3.66×` faster than synchronous repair) and beats trained early exit by `+9.7 pts` at matched `~220 tok/s` on GSM8K (`+4.8 pts` at matched `~225 tok/s` and `+12.4 pts` at $K = 4$ on IFEval). See `results/` for full per-config breakdowns, trained early-exit comparison tables, and the 5-policy cache-completion lag and depth ablation.
