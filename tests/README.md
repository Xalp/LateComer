# Deterministic CPU Exactness & Verification Suite

All mathematical exactness invariants (`fp64` and `fp32`) and physical Roofline law calibrations run deterministically on CPU in under a minute without requiring a GPU.

## Run Full Suite (`9` Verification Modules)

```bash
python tests/run_all_cpu_tests.py
```

## Individual Modules

| Script | What It Verifies |
| :--- | :--- |
| `../wake_verify/test_wake_cpu.py` | **Proposition 1 (`tau = 0` greedy-exactness)** & EOS wake-drain across Ouro ($T=4, K \in \{1..3\}$) and Huginn ($T=8, K \in \{2,4,6\}$) under adversarial wrong/random/suffix drafters (`258/258` `fp64` checks) |
| `../huginn/tests/test_hg_fast_cpu.py` | Huginn paged-KV fast engine bit-equivalence against explicit 4D-mask `RefBlockDecoder` across `pipe`, `p24`, `p16`, `sync`, `none`, and `eblk` (`128/128` checks) |
| `test_cpu_latek.py` | Ouro $K \in \{1..4\}$ block decoder equivalence against independent HF-module `RefLateK`, plus per-call read/write and KV depth instrumentation |
| `test_cpu_eeblk.py` | Truncated-model early-exit block decoder (`eeblk:`) equivalence and cache invariants |
| `test_cpu_lkdb.py` | Restart-repair (`lkdb:`) deep-readout invariants upon full proposal rejection |
| `../concurrency/test_cpu_batch_latek.py` | Multi-request ($R \in \{1,2,3,5\}$) `LateKBatchExecutor` with staggered online slot refills vs. independent $R=1$ runs (`546` checks) |
| `../concurrency/sglang_tests/test_cpu_core.py` | SGLang `LoopStridedMHAPool` physical KV layout and shuffled-slot equivalence |
| `../wake_verify/analyze_roofline_v8.py` | Zero-shot $L_p$ Roofline law calibration on $K=4$ (`8.27%` MAPE across $K \in \{3,2,1\}$) and concurrency $B^\star(C)$ prediction (`11/12` match) |
