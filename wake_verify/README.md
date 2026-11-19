# Wake Verification, Recurrence Depth Scaling & Physical Roofline Law

This module implements **Wake Verification** (`Section 3.4` and `Section 4.5`), **Intra-Model Recurrence Depth Scaling ($T \in \{8, 16, 32, 64\}$)** (`Section 4.6`), **Drafter-Verifier Factorization**, and the **Zero-Shot Physical $L_p$ Roofline Throughput Law** (`Section 4.7`) for *Late Entry in Looped Language Models*.

## How Wake Verification Works

In $K$-pass late-entry block decoding ($1 \le K < T$), candidate blocks execute only the final $K$ passes on the critical path to emit a **tentatively accepted frontier**, while at most $L = \lceil T / K \rceil$ earlier groups complete exact causal passes $1 \to T$ as piggybacked batch rows inside ongoing forward calls at **zero extra serial calls**.

Whenever the oldest tentative group $[u, v)$ finishes pass $T$ at the back of the wake, its hidden states $\tilde{h}_i^{(1 \dots T)}$ (or Coda states on Huginn) are mathematically identical to sequential $T$-pass forward states. Projecting those $\le 1 + B$ finished wake rows through `lm_head` **inside the exact same single output GEMM** as the $(1 + B)^2$ active block rows triggers a rollback at the earliest index $i^* \in [u, v)$ where:

$$y^*_{i^*} \neq \hat{y}_{i^*} \quad \text{and} \quad z_{1:T}^{(i^*)}(y^*_{i^*}) - z_{1:T}^{(i^*)}(\hat{y}_{i^*}) > \tau$$

- **`tau = 0` (Exact Wake, Proposition 1):** Guarantees bit-for-bit $T$-pass greedy equivalence in exact arithmetic (matching $K = T$ block decoding within `bf16` reduction order at `0.007%` to `0.09%` token discrepancy, below the `0.25%` to `0.33%` sequential `bf16` floor) and strict finite-step termination ($i^* + 1 > u$).
- **`0 < tau < inf` (Margin-Gated Wake):** Filters benign near-tie `bf16` or phrasing flips ($\Delta z \le \tau$) while rolling back high-margin reasoning deviations ($\Delta z > \tau$).
- **`tau = inf` (Non-Rollback Lossy Late Entry):** Zero rollbacks for peak throughput.

## Module Layout

| File | Description |
| :--- | :--- |
| `ouro_wake.py` | `WakeKLattice`: Ouro-2.6B ($T = 4$) single-graph Wake decoder (`wk:K:latt:B:tau`, `wks:K:latt:B:tau`) |
| `hg_wake.py` | `HuginnWakeDecoder`: CUDA-graphed paged FA3 Huginn-3.5B Wake decoder supporting arbitrary test-time $T \in \{8, 16, 32, 64\}$ (`wblk:K:B:tau`, `wsblk:K:B:tau`) |
| `suffix_draft.py` | Parameter-free $4/3/2$-gram context-suffix drafter for testing drafter-verifier factorization |
| `run_ouro_wake.py` | Full-dataset Ouro-2.6B Wake & Roofline evaluation runner with CUDA-graph gates and teacher-forced audits |
| `score_wake.py` | Unified full-dataset scorer with gold-verified `GSM8K`, `MATH500`, `IFEval`, and `MBPP` evaluators and paired sign tests |
| `analyze_roofline_v8.py` | Calibrates $L_p$ Roofline constants strictly on $K = 4$ modes and evaluates zero-shot $K < 4$ concurrency predictions across $C \in [1, 32]$ (`8.3% MAPE`, `11/12` $B^*(C)$ match) |
| `test_wake_cpu.py` | Deterministic `fp64` CPU test suite (`258 / 258` adversarial checks across divisible/non-divisible $T/K$ and premature-`<eos>` sequences for both Ouro and Huginn) |
| `results/` | Full-dataset scored summaries (`WAKE_SUMMARY.json` and `ROOFLINE_LAW.json`) |

## Quickstart

### 1. Run CPU `fp64` Exactness Gates (Proposition 1, Zero GPU Needed)

```bash
OMP_NUM_THREADS=2 PYTHONPATH=. python wake_verify/test_wake_cpu.py
```

### 2. Ouro-2.6B ($T = 4$) Wake Verification (`B = 8`)

```bash
python wake_verify/run_ouro_wake.py \
  --model Xalphinions/ouro-2.6b-late-entry \
  --prompts prompts/gsm8k_prompts.jsonl \
  --modes "ar,lk:4:latt:8,wk:3:latt:8:0,wk:3:latt:8:0.25,lk:3:latt:8,wk:2:latt:8:0,wk:2:latt:8:0.15,wk:2:latt:8:0.3,lk:2:latt:8,wks:2:latt:8:0" \
  --max-new 1024 \
  --out outputs/ouro_wake_gsm8k.json
```

### 3. Huginn-3.5B Wake Verification & $T \in \{8, 16, 32, 64\}$ Recurrence Scaling

```bash
# T = 32 Wake spectrum (Exact tau=0, Margin tau=0.25, Lossy tau=inf, and Suffix-Ngram wsblk)
PYTHONPATH=. python -m wake_verify.hg_wake \
  --model Xalphinions/huginn-3.5b-late-entry \
  --T 32 \
  --tasks data/tasks/gsm8k.jsonl \
  --cfgs "blk:32:0,blk:32:8,wblk:16:8:0,wblk:16:8:0.25,blk:16:8,wblk:8:8:0,wblk:8:8:0.25,blk:8:8" \
  --max-new 512 \
  --out-dir outputs/hg_wake_t32

# Intra-model T = 64 recurrence scaling (260 effective layers/token, identical checkpoint)
PYTHONPATH=. python -m wake_verify.hg_wake \
  --model Xalphinions/huginn-3.5b-late-entry \
  --T 64 \
  --tasks data/tasks/gsm8k.jsonl \
  --cfgs "blk:64:0,blk:64:8,wblk:32:8:0,blk:32:8,wblk:16:8:0,wblk:16:8:0.25,blk:16:8" \
  --max-new 512 \
  --out-dir outputs/hg_wake_t64
```

### 4. Reproduce Zero-Shot Physical Roofline Law (`8.3% MAPE`)

```bash
python wake_verify/analyze_roofline_v8.py
```

## Summary of Full-Dataset Results (Single H100 80GB GPU, `B = 8`)

### Wake Verification Spectrum (`Paper Table 7`)

| Model | Benchmark | Budget | Seq. $T$ (`tok/s`) | $K = T$ Block | Exact Wake ($\tau = 0$) | `rb/100t` | Margin Wake ($\tau > 0$) | `rb/100t` | Lossy ($\tau = \infty$) |
| :--- | :--- | :---: | :---: | :---: | :---: | ---: | :---: | ---: | :---: |
| **Ouro-2.6B** ($T=4$) | GSM8K | $K=3$ | `.868` (`62.3`) | `.867` / `4.13×` | **`.867` / `4.19× (+2%)`** | `1.14` | **`.870` (`.25`) / `4.55× (+10%)`** | `0.29` | `.853` / `5.18×` |
| **Ouro-2.6B** ($T=4$) | GSM8K | $K=2$ | `.868` (`62.3`) | `.867` / `4.13×` | **`.867` / `5.20× (+26%)`** | `2.02` | **`.877` (`.15`) / `5.52× (+34%)`** | `1.26` | `.842` / `6.82×` |
| **Ouro-2.6B** ($T=4$) | MATH500 | $K=2$ | `.532` (`62.7`) | `.526` / `4.08×` | **`.526` / `5.23× (+28%)`** | `2.11` | **`.548` (`.30`) / `5.83× (+43%)`** | `0.96` | `.492` / `6.61×` |
| **Ouro-2.6B** ($T=4$) | IFEval | $K=2$ | `.732` (`63.8`) | `.739` / `2.15×` | **`.739` / `2.66× (+24%)`** | `5.98` | `.719` (`.15`) / `2.94× (+36%)` | `4.08` | `.699` / `3.76×` |
| **Ouro-2.6B** ($T=4$) | MBPP | $K=2$ | `.616` (`63.4`) | `.626` / `3.79×` | **`.626` / `4.61× (+22%)`** | `2.00` | **`.618` (`.30`) / `5.13× (+35%)`** | `0.70` | `.602` / `6.20×` |
| **Huginn-3.5B** ($T=32$) | GSM8K | $K=16$ | `.391` (`21.9`) | `.394` / `3.35×` | **`.396` / `5.21× (+56%)`** | `0.48` | **`.404` (`.25`) / `5.42× (+62%)`** | `0.05` | `.399` / `5.80×` |
| **Huginn-3.5B** ($T=32$) | GSM8K | $K=8$ | `.391` (`21.9`) | `.394` / `3.35×` | **`.392` / `6.09× (+82%)`** | `2.22` | **`.397` (`.25`) / `7.15× (+114%)`** | `0.88` | `.353` / `9.13×` |
| **Huginn-3.5B** ($T=32$) | IFEval | $K=16$ | `.560` (`21.5`) | `.547` / `2.91×` | **`.547` / `4.85× (+66%)`** | `0.37` | **`.560` (`.25`) / `5.10× (+75%)`** | `0.02` | `.562` / `5.21×` |
| **Huginn-3.5B** ($T=32$) | IFEval | $K=8$ | `.560` (`21.5`) | `.547` / `2.91×` | **`.556` / `6.46× (+122%)`** | `2.03` | `.543` (`.25`) / `7.59× (+161%)` | `0.59` | `.542` / `8.85×` |
| **Huginn-3.5B** ($T=32$) | MBPP | $K=16$ | `.234` (`21.9`) | `.236` / `4.47×` | **`.236` / `6.64× (+49%)`** | `0.25` | **`.234` (`.25`) / `6.79× (+52%)`** | `0.03` | `.232` / `7.80×` |
| **Huginn-3.5B** ($T=32$) | MBPP | $K=8$ | `.234` (`21.9`) | `.236` / `4.47×` | **`.236` / `7.69× (+72%)`** | `1.41` | **`.236` (`.25`) / `8.93× (+100%)`** | `0.56` | `.228` / `12.50×` |

### Intra-Model Recurrence Depth Scaling on Huginn-3.5B (`GSM8K n = 1,319`, `Paper Table 8`)

| Depth $T$ (Eff. Layers) | Seq. $T$ (`tok/s`) | $K = T$ Block | $K = T/2$ Exact ($\tau=0$) | $K = T/2$ Lossy ($\tau=\infty$) | $K = T/4$ Exact ($\tau=0$) | $K = T/4$ `rb/100t` | $K = T/4$ Lossy ($\tau=\infty$) |
| :--- | :---: | :---: | :---: | :---: | :---: | ---: | :---: |
| **$T = 8$** (`36L`) | `.267` (`78.6`) | `2.86×` | `.265` / `2.63×` (`rb 5.56`) | `.167` / `3.78×` | **`.262`** / `1.96×` | `12.46` | `.030` / `4.82×` |
| **$T = 16$** (`68L`) | `.382` (`42.1`) | `3.32×` | `.379` / **`4.21×`** (`rb 1.85`) | `.361` / `5.15×` | **`.375` / `3.47×`** | `6.25` | `.146` / `6.56×` |
| **$T = 32$** (`132L`) | `.391` (`21.9`) | `3.35×` | `.396` / **`5.21×`** (`rb 0.48`) | `.399` / `5.80×` | **`.392` / `6.09×`** | `2.22` | `.353` / `9.13×` |
| **$T = 64$** (`260L`) | `.397` (`11.1`) | `3.34×` | `.397` / **`5.42×`** (`rb 0.26`) | `.394` / `5.97×` | **`.395` / `8.42× (+153%)`** | **`0.58`** | **`.392` / `10.57×`** |
