# Head-to-head, FULL test sets (21,403 documents)

| task | n | len | ours AR | LS AR | LA AR | denom(src) | LS 1:2 | LA best(arm) | K=4 | K=3 | K=2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 1319 | 108 | 63.6 | 67.2 | 63.2 | 67.2 (loopspec) | 190.9 (2.84x) | 197.8 (2.94x, la_fav) | 3.82x | 4.80x | 6.35x |
| ifeval | 541 | 347 | 63.5 | 64.7 | 62.9 | 64.7 (loopspec) | 151.7 (2.34x) | 173.0 (2.67x, nolad) | 2.12x | 2.75x | 3.68x |
| math500 | 500 | 311 | 62.7 | 61.4 | 62.1 | 62.7 (ours) | 178.4 (2.85x) | 162.0 (2.59x, nolad) | 4.11x | 5.07x | 6.63x |
| mbpp | 500 | 82 | 62.1 | 65.1 | 63.4 | 65.1 (loopspec) | 186.5 (2.86x) | 188.9 (2.90x, la_fav) | 3.64x | 4.51x | 5.95x |
| bbh | 6511 | 71 | 62.5 | 65.2 | 62.8 | 65.2 (loopspec) | 171.1 (2.62x) | 212.2 (3.25x, la_fav) | 3.29x | 4.35x | 6.33x |
| mmlupro | 12032 | 195 | 62.4 | 61.2 | 61.8 | 62.4 (ours) | 171.7 (2.75x) | 152.1 (2.44x, nolad) | 3.56x | 4.56x | 6.12x |

problems: loopahead lafull_mmlupro_s0_mmlupro.json: meta.done false (keeping completed modes: ['ar', 'la']); loopahead lafull_mmlupro_s3_mmlupro.json: meta.done false (keeping completed modes: ['ar']); loopahead lafull_mmlupro_s5_mmlupro.json: meta.done false (keeping completed modes: ['ar']); loopahead lavfull_mmlupro_s3_mmlupro.json: meta.done false (keeping completed modes: ['ar']); loopahead lavfull_mmlupro_s5_mmlupro.json: meta.done false (keeping completed modes: ['ar', 'la'])
