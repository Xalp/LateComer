# Entry Norm Mismatch & Internal Mechanism Probes (`Sections 2.1-2.3, Appendix C`)

Diagnostic probes examining why naive off-the-shelf late entry collapses and how two-stage fine-tuning enables deep context to substitute for skipped token-local passes.

## Structure

| Subdirectory | Paper Section | Contents |
| :--- | :--- | :--- |
| `entry/` | Sec. 2.1, Table 1, App. C | Off-the-shelf hidden-state norm mismatch (`~0.85` embedding norm vs. `~43` pass-4 norm), scalar rescaling, tied-unembedding tests, and closed-form Ridge / MLP frozen entry adapters (`entry_core.py`, `entry_run.py`, `entry_analyze.py`, `test_entry_cpu.py`, `results/entry_summary.md`) |
| `mechanism/` | Sec. 2.2-2.3, Fig. 2 | Teacher-forced $4 \times 4$ context-depth ($D$) vs. token-depth ($K$) grid, cross-pass $(K, V)$ coherence swaps, long-range context ablations, and confidence-stratified agreement (`mech_core.py`, `mech_run.py`, `mech_analyze.py`, `test_mech_cpu.py`, `results/mech_summary.md`) |

## CPU Verification

```bash
OMP_NUM_THREADS=2 python probes/entry/test_entry_cpu.py
OMP_NUM_THREADS=2 python probes/mechanism/test_mech_cpu.py
```
