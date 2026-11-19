# Ouro Paged-FA3 Engine & Late-Entry Block Decoder (`Sections 3.1-3.3, 4.1-4.2`)

Standalone single-stream looped Transformer inference engine and $K$-pass block decoder for **Ouro-2.6B / Ouro-1.4B** ($T = 4$ recurrent passes).

## Files

| File | Description |
| :--- | :--- |
| `engine.py` | Direct safetensors loader, fused RMSNorm / SiLU-and-Mul / RoPE ops, paged FlashAttention-3 (`sgl_kernel`) + PyTorch reference attention, static per-pass KV regions, CUDA-graphed prefill, and synchronous full-depth block lattice (`Lattice`) |
| `latek.py` | Generalized $K$-pass block decoder (`LateKLattice`) supporting pipelined $1 \to T$ cache completion (`lk:`), no-completion ablation (`lknb:`), full-rejection restart repair (`lkdb:`), and truncated early-exit block decoding (`eeblk:`) |
| `late.py` | Specialized two-pass ($K = 2$) late-entry lattice (`LateLattice`) |
| `sparse.py` | Candidate + mask-placeholder $(1 + B)^2$ geometry and bottom-right causal tail decomposition |
| `rolled.py` | Rolled multi-pass KV memory layout |
| `run_latek.py` | Unified CLI benchmark runner reporting decode `tok/s`, end-to-end `tok/s`, accepted run lengths, and per-document generations |
| `results/` | Aggregated single-H100 throughput and accuracy tables (`results.json`, `results_tables.md`) |

## Quickstart

```bash
# Decode GSM8K with full-depth sequential (ar), full-depth block (K=4), and late-entry block (K=3, K=2)
python decode/run_latek.py \
  --model Xalphinions/ouro-2.6b-late-entry \
  --prompts eval/data/gsm8k_prompts.jsonl \
  --modes "ar,lk:4:latt:8,lk:3:latt:8,lk:2:latt:8" \
  --max-new 1024 \
  --out results/ouro_latek_gsm8k.json

# Matched trained early-exit block control (Paper Table 4)
python decode/run_latek.py \
  --model Xalphinions/ouro-2.6b-early-exit \
  --prompts eval/data/gsm8k_prompts.jsonl \
  --modes "ar,eeblk:3:latt:8,eeblk:2:latt:8,eeblk:1:latt:8" \
  --max-new 1024 \
  --out results/ouro_early_exit_gsm8k.json
```

### Mode String Reference

* `ar`: Full-depth ($T = 4$) sequential greedy decoding in the same CUDA-graphed engine (paired baseline).
* `lk:<K>:latt:<B>`: Late-entry block decoding running the final $K \in \{1, 2, 3, 4\}$ passes with block size $B$ and pipelined $1 \to T$ KV completion.
* `lknb:<K>:latt:<B>`: No-completion ablation keeping provisional late-pass KV states without background catch-up (`Paper Section 4.2`).
* `lkdb:<K>:latt:<B>`: Restart-repair variant computing a full $T$-pass readout upon zero-proposal acceptance (`Paper Appendix F`).
* `eeblk:<K>:latt:<B>`: Truncated-model early-exit block decoding running only the first $K$ passes (`Paper Table 4`).
