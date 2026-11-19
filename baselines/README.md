# Full-Depth Speculative Baselines: LoopSpec & LoopAhead (`Section 4.3, Appendix G`)

Benchmark clients, reimplementation code, and 21,403-example head-to-head aggregation for full-depth ($T = 4$) speculative decoding baselines on **Ouro-2.6B**.

## Files

| Path | Description |
| :--- | :--- |
| `loopspec/client_ids.py` | Token-ID HTTP client driving the official LoopSpec SGLang server on identical tokenized benchmark prompts |
| `loopahead/` | Self-contained PyTorch / CUDA-graphed reimplementation of **LoopAhead** (`drafter/lad.py`, `drafter/train_lad.py`, `run_la.py`, plus full design & protocol notes in `PLAN.md` and `REPORT.md`) |
| `make_headtohead_full.py` | Pools full-test-set runs across all six benchmarks (`21,403` documents) normalizing every speedup against the fastest sequential $T = 4$ baseline per task |
| `results/` | Full six-benchmark comparison tables (`HEADTOHEAD_FULL.json`, `HEADTOHEAD_FULL.md`) |
