# LoopAhead reproduction — consolidated report (Stages 1–3 + favourable arm)

**Paper:** LoopAhead (arXiv 2610.21594, no released code). **This is OUR reimplementation**;
every number below is labelled as such, never as theirs. Assumptions where the paper is
silent: PLAN.md §4 (24 recorded items). Ledger: `jobs.tsv`. All artifacts under
`<data-root>

Status: COMPLETE — Stages 1–3 plus the favourable (anti-under-training) arm; see §5 for
the published max(faithful, favourable, no-drafter) table.

---

## 1. Stage 1 — feasibility (2026-10-30)

Full algorithm extraction from the arXiv HTML; verdict: reproducible. See PLAN.md. Key
facts used downstream: Ouro hyperparameters d1=1, d2=2, R=4, K=3, B_ret=32, B_exe=16,
d_guard=2, stride 1 (their Table 7); drafter ≈11.1M, 2 layers, hidden 512 (their §3.1/A.5);
training mix 9,025/782/782 (their Table 6); greedy LoopAhead is exact by construction.

## 2. Stage 2 — teacher data + drafter (2026-10-31 → 11-03)

**Teacher data** (engine exact full-depth AR, bf16 greedy, the audited decoder — batched HF
generate was rejected: released modeling ignores left padding):
train 9,025 / dev 782, per-source counts exactly their Table 6; 2.39M train gen tokens
(mean 264, p50 212, cap 512), 0 empty. Draft vocab 8,192 of 24,986 distinct, coverage 97.6%.
One defect found: a platform-kill mid-flush glued a truncated record to its regenerated copy
(dev00361); repaired, 782 unique sids verified.

**Drafter** (job `07f32e55bf3aab02`): faithful A.5 recipe, 11,051,008 params (asserted ==
their stated count), 13,536 updates. Loss: train CE 6.60→3.15, dev CE 6.59→3.519, smooth
monotone, best = final (no overfit; plateaued: Δ last 1k upd −0.021).

**Held-out acceptance** (teacher-forced cumulative q_j, 2,048 dev anchors):

| source | q1 | q2 | q3 | q1+q2+q3 ≈ E[accepted/call] |
|---|---|---|---|---|
| aggregate | .661 | .394 | .211 | 1.27 |
| gsm8k | .747 | .494 | .240 | 1.48 |
| bbh | .734 | .516 | .315 | 1.57 |
| codealpaca | .787 | .509 | .309 | 1.61 |
| math | .604 | .230 | .070 | 0.90 |
| ultrachat | .347 | .091 | .021 | 0.46 |

vs their B.2 GSM8K anchor (q1≈64.3%, q3≈36.1%): ours is front-loaded (q1 +10.4 pts, q3
−12.1 pts); E[L]=1.48 vs their 1.37–1.65 band (≈1.49 at geometric q2). **Drafter certified
not under-trained on GSM8K.** Weak tails: math, ultrachat → favourable arm (§5).
Vocab restriction to 8,192 costs ≤0.3 pt everywhere (their claim reproduces).

## 3. Stage 3 — measurement (job `453aba994fa205a3`, 2026-11-03)

Protocol: released Ouro-2.6B, 120 matched fair_* docs per task, batch 1, greedy, bf16, one
H100, isolated GPU, AR measured FIRST in the same process (shared denominator), decode-only
tok/s (prefill excluded, their convention), per-token full-depth audit.

| task | AR tok/s | la tok/s | **la ×** | la:nolad × | la tok/step | nolad tok/step | la step ms | nolad step ms |
|---|---|---|---|---|---|---|---|---|
| gsm8k | 63.0 | 174.5 | **2.77** | 2.84 | 1.663 | 0.923 | 9.61 | 5.13 |
| math500 | 61.3 | 135.1 | **2.20** | 2.61 | 1.344 | 0.887 | 9.96 | 5.53 |
| bbh | 62.5 | 176.1 | **2.82** | 2.80 | 1.672 | 0.911 | 9.57 | 5.14 |
| mbpp | 62.9 | 170.7 | **2.71** | 2.90 | 1.574 | 0.923 | 9.32 | 5.01 |
| ifeval | 63.0 | 119.2 | **1.89** | 2.74 | 1.120 | 0.890 | 9.36 | 5.08 |
| mmlupro | 61.5 | 112.5 | **1.83** | 2.47 | 1.159 | 0.853 | 10.28 | 5.57 |

Audit (full-depth re-check of every committed token): diff share 0.16–0.36% in ALL modes
*including plain AR* — i.e. bf16 near-tie noise from GEMM-shape changes, not an algorithmic
loss. Greedy losslessness holds as in the paper.

**Findings:**
1. **The no-drafter tree scheduler reproduces their ablation.** Their no-LAD arm: 2.88×
   (Ouro-1.4B, A40). Ours: 2.47–2.90× across six tasks on 2.6B/H100 over a CUDA-graph
   engine AR — same level against a much stronger denominator.
2. **The drafter mechanism works at the token level** (tok/step +80% on gsm8k/bbh; LAD
   supplies ~half of committed tokens) **but its serving cost eats the gain on our
   hardware**: a LAD call fires on ~96% of steps and raises step cost 5.1→9.6 ms
   (drafter forward + draft-vocab restriction + tree bookkeeping + mean rows 4→12).
   Full-LA beats no-LAD only on BBH (best acceptance). On their A40, a loop-step is ~3×
   slower, so the same per-step toll hides proportionally (risk recorded in PLAN §4
   BEFORE measuring).
3. **Gap to their headline (5.21× GSM8K on 2.6B) is dominated by the denominator.** Our AR
   runs 63 tok/s (CUDA-graph engine, step 15.8 ms = 4 loop passes × ~3.95 ms); their AR
   implementation is unspecified PyTorch on A40. Our repro over our engine-AR: 1.83–2.82×.
   Both facts must travel together when quoting.

## 4. Honest-labelling notes
- Greedy-only scope; batch 1; different GPU and AR denominator than theirs (absolute tok/s
  reported above so any reader can re-derive).
- Our six tasks ≠ their five (we add ifeval, mmlupro; no HumanEval).
- Audit shares put a ceiling of ~0.4% on token divergence; identical for AR itself.
- The implementation is ours; where it could flatter us it was checked (same-process
  denominator, prefill excluded, isolated GPU, single sync/step), and where it could hurt them
  (host-side LAD toll on a fast GPU) it is stated explicitly above.

## 5. Favourable arm (anti-under-training; user directive) — COMPLETE

Trigger: weak acceptance on math/ultrachat-like tasks in the faithful arm. Published numbers
must not be attributable to OUR under-training.

**Data** (jobs `32dc4d0fb5350b2a`, `09912122a0b22baa`): +2,000 MATH + +1,000 UltraChat teacher
prompts, provably disjoint and identical protocol — `build_mix_ext.py` execs `build_mix.py`'s
own source for the pools/dedup/protection, replays the identical seeded rng stream, ASSERTS the
shipped mix reproduces exactly per source, then takes the NEXT pool items (MANIFEST_ext.json).
data_fav = 12,025 train docs, 3.75M gen tokens; draft vocab rebuilt (8,192 of 29,333, cov 97.4%).

**Training** (job `42e988b561c27ca0`): 6 epochs x 64 anchors/req = 72,150 updates (4x the
faithful optimization). Dev CE 3.5192 -> **2.9601**; same smooth monotone shape, plateaued.

| acceptance (dev) | faithful | **favourable** | their B.2 anchor |
|---|---|---|---|
| q1 gsm8k | .747 | **.762** | .643 |
| q3 gsm8k | .240 | **.346** | .361 |
| E[L] gsm8k | 1.48 | **1.67** | 1.37-1.65 |
| E[L] bbh | 1.57 | **1.86** | - |
| E[L] math | 0.90 | **1.31** | - |
| E[L] ultrachat | 0.46 | **0.64** | - |

The faithful 3-epoch recipe DID leave acceptance on the table at depth (math q3 .070->.192,
~3x). With the extra budget, GSM8K sits at their anchor on both ends.

**Re-measurement** (job `3c05ff4a147018b1`, all six tasks, ar+la, same timing protocol, isolated GPU; AR re-measured in-process — its denominator reproduced the faithful job within 0.8%,
a useful cross-job stability check):

| task | AR tok/s | faithful la x | **favourable la x** | la:nolad x | **MAX (published)** |
|---|---|---|---|---|---|
| gsm8k | 62.5 | 2.77 | **3.09** | 2.84 | **3.09** (fav la) |
| math500 | 61.3 | 2.20 | **2.51** | 2.61 | **2.61** (nolad) |
| bbh | 62.4 | 2.82 | **3.33** | 2.80 | **3.33** (fav la) |
| mbpp | 62.8 | 2.71 | **3.02** | 2.90 | **3.02** (fav la) |
| ifeval | 62.8 | 1.89 | **2.10** | 2.74 | **2.74** (nolad) |
| mmlupro | 61.0 | 1.83 | **2.03** | 2.47 | **2.47** (nolad) |

Every cell improved 10-18%. The gain is PURELY acceptance: tok/step rose (gsm8k 1.663->1.883,
bbh 1.672->1.921) while step cost stayed flat (~9.3-10.1 ms), confirming the §3 diagnosis that
the drafter's per-step toll is fixed and only acceptance moves the product.

**The qualitative conclusion changes on three tasks.** With the better drafter, full LoopAhead
now BEATS its own no-drafter ablation on gsm8k (3.09 vs 2.84), bbh (3.33 vs 2.80) and mbpp
(3.02 vs 2.90) — i.e. the drafter earns its cost where acceptance is high. It still loses on
ifeval (2.10 vs 2.74), mmlupro (2.03 vs 2.47) and math500 (2.51 vs 2.61), where acceptance
stays low; on those tasks the paper's own no-LAD configuration is the better one on our
hardware. Audit diff shares unchanged (0.16-0.36%, same as plain AR) — losslessness intact.

**Directive satisfied:** the faithful arm is reported as the faithful result, and no published
cell is now limited by our training budget — each takes max over the faithful drafter, the
4x-budget drafter, and the no-drafter configuration.
