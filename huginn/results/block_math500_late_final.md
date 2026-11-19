# Huginn Formal Evaluation: `math500` (`mars_late_fast`, primary=`mv_lenient`)

| Config | Kind | B | K | BG | N | Primary Acc | Gap vs `r:32` (p) | vs `eblk` (p) | `1+E[c]` | Tok/s | Speedup |
|---|---|---:|---:|---|---:|---:|---|---|---:|---:|---:|
| `blk:32:0` | blk | 0 | 32 | none | 500 | 3.80% | - | - | 1.00 | 22.33 | 1.00x |
| `blk:32:8` | blk | 8 | 32 | none | 500 | 4.00% | +0.20% (1) | - | 4.96 | 92.43 | 4.14x |
| `blk:16:8` | blk | 8 | 16 | pipe | 500 | 3.60% | -0.20% (1) | - | 4.90 | 159.08 | 7.12x |
| `blk:12:8` | blk | 8 | 12 | pipe | 500 | 3.80% | +0.00% (1) | - | 4.82 | 197.85 | 8.86x |
| `blk:8:8` | blk | 8 | 8 | pipe | 500 | 5.00% | +1.20% (0.0312) | - | 4.78 | 262.65 | 11.76x |
