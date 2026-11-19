# K-pass late lattice: entry training (design, 10-28)

Goal: make the K-pass late lattice accurate. On v2, the untrained K=2 lattice (`late2/late.py`, `l2:lkv:am:latt:B`) commits tokens
that differ from the full-depth argmax 2.0% of the time on GSM8K and 5.8% on IFEval (`late2_full/results.md`). The loss comes from the
2-pass verifier, i.e. from the token slots' readout after entering at pass 3. This fine-tune trains exactly the rows that decoder runs.

Init: v2 = `Xalphinions/ouro-2.6b-late-entry`. It already knows MARS placeholders and mid-pass token injection.
TAG `ouro26_latelat`. Output `checkpoints/ouro26_latelat/` (a new directory; launcher refuses foreign TAGs).
Code: this directory only. `stage_late.sh` and `export_ckpt.py` are byte copies of `../train/`.

## 1. What the decoder computes (derived from late.py and late_k/latek.py)

Notation: T = 4 passes. K = block passes per lattice step, entry pass E = T-K+1. B = proposals, D = B+1 token slots.
Dense shape `latt:B`: one placeholder set of size B after every token slot.

**Step s, anchor c = Lc.**
- The block holds token slot j (j = 0..B) at position c+j, with input x_{c+j}. Slot 0 is the last committed token; slots 1..B hold the
  proposals.
- It also holds placeholder (a, i) (a = 0..B, i = 1..B) at position c+a+i, with input `<|mask|>`.
- All rows enter at pass E and run passes E..T in K calls; the hidden state carries across the calls.
- Readout = the row's gate mixture over its own passes, restarted at entry: lam_k·rem for k < T, rem at T, rem = 1 at E.

**Inside the block (lattice mask, same pass).** Token slot j sees slots 0..j. Placeholder (a, i) sees slots 0..a and (a, 1..i).

**Context positions q < c, at pass k (lateKV reading = the decoder's best reading).** If q is a prompt position, read the exact
full-depth K/V at pass k. If q was committed at step s0, the background has finished d = min(T, K(s-s0-1)+kk+1) passes, counting the
current call. If d >= k, the decoder reads the background's pass-k K/V, which is the exact full-depth K/V. If d < k, it reads the pass-k
K/V that q's own TOKEN SLOT wrote in step s0.

With k = E+kk, "d < k" becomes K(s-s0-1) < T-K, which is **r·K < T with r = s - s0**. That does not depend on the pass. So:

| K | context read from the committing step's token slot (late) | from full depth |
|---|---|---|
| 1 | positions committed 1, 2, 3 steps ago | older, prompt |
| 2 | 1 step ago | older, prompt |
| 3 | 1 step ago | older, prompt |
| 4 | none (= the exact lattice) | everything |

**Verification.** Proposal i (in token slot i) is accepted iff it equals the argmax of token slot i-1. With c accepted, the step commits
proposals 1..c plus slot c's argmax, and **token slots 0..c join the context**, carrying their late K/V. ℓ = c+1 tokens are committed per
step. The next proposals are placeholder set c's argmaxes; placeholder (c, i), at position Lc+c+i, predicts x_{Lc+c+i+1}.

**First step after prefill.** c = Lc = P0 and there are no proposals, so ℓ = 1. Every committed token is a token-slot readout; the
prefill's first token is the only exception.

## 2. Teacher forcing: what is exact and what is not

We drive the decoder on the true text: the proposals are the true tokens, and the step lengths ℓ_i are sampled. The readouts the decoder
USES are then reproduced exactly:
- the verifier/correction readouts of slots 0..c, which see only accepted (= true) tokens;
- the readouts of placeholder set c, which sees slots 0..c.

The rows the decoder computes but DISCARDS see true tokens here instead of the rejected proposal: slots > c, and sets a > c. We still
train them, with true inputs; that is harmless. In decoding, the first step's slots 1..B hold `<|mask|>` filler. We feed true tokens
there (an unused readout; ℓ_1 = 1 as in decoding).

## 3. Training layout per sequence (two streams)

Positions 0..n-1, chain start a = first supervised position (= the decoder's P0).

- **Stream 1** (rows 0..n-1) is the plain full-depth Ouro forward, causal, T passes, gate-mixture readout, standard next-token loss.
  Stream 1 never reads stream 2.
- **Stream 2** exists only at passes E..T (K passes), with input embeddings at pass E. Its rows are:
  - **Chain rows** (n rows, row n+p ↔ position p, input E[x_p]): one per committed position. These are the joined token slots. The row
    at p, in step i = st(p), is slot p - c_i of block i. The late K/V that later steps read are exactly these rows' per-pass K/V, so the
    recursion (step i's slots read step i-1's slots, which read step i-2's, ...) is modelled exactly in one forward. No approximation
    is needed.
  - **Extra rows** of selected blocks (E rows): the non-joined token slots j = ℓ_i..B (input x_{c+j}) and all B(B+1) placeholders
    (input `<|mask|>`, the v2 [P] embedding, id 49152). Nobody reads them, so including only a subset of blocks is exact for every row.

### Step boundaries

The anchors are c_1 = a (ℓ_1 = 1), then c_{i+1} = c_i + ℓ_i, with ℓ_i = 1 + min(B, G) and G ~ Geometric (P(accept) = α per offset,
α = 0.8 by default). This gives E[ℓ] = 3.36 (B=4) / 4.33 (B=8), against the measured 3.54 / 4.26 tokens per step (late2_full).

### Mask (one mask for all K stream-2 passes; True = attend; query → key)

For a query of step i with "same-step bound" b (the chain row at p: b = p; extra token slot j: b = c_i+j; placeholder (a', i'):
b = c_i+a'):
- **stream 1 → stream 1:** causal. **stream 1 → stream 2:** never.
- **stream 2 → stream-1 key j:** j < c_i and (j < a or (i - st(j))·K >= T). These are the full-depth reads.
- **stream 2 → chain key j:** a <= j <= b, st(j) <= i, and (i - st(j))·K < T. These are the late reads, plus the own block's joined
  slots.
- **extra → extra (same block only):** token slot j sees extra slots ℓ..j. Placeholder (a', i') sees extra slots ℓ..a' and (a', 1..i').
- **Unused rows** (chain rows p < a, padding) see self + stream 1 j < p. They carry no loss and are never read by a used row.

RoPE position of a row = the decoder's position. Each pass is one SDPA call per layer over [stream 1 | stream 2] with this bool mask:
a query at pass k reads only pass-k keys, i.e. per-pass K/V exactly as the decoder.

## 4. Losses (per micro-batch; each term = mean over sequences of the per-sequence mean, MARS 'sequence' normalisation)

- L1: stream-1 CE at every supervised next token (keeps full depth).
- L_tok: token-slot rows (chain rows + extra token slots) at position p with a valid target (label at p+1, no EOS/`<|im_end|>` at
  a..p). Default **KL(stream-1 full-depth distribution at p, detached ‖ slot readout)**, which distills the full-depth prediction, since
  that is exactly what the audit measures. Option `--lat_tok_loss ce` uses CE to x_{p+1}. Both are always logged.
- L_ph: placeholder CE to x_{p+1} (p = the placeholder's position).
- loss = w1·L1 + w_tok·L_tok + w_ph·L_ph, all weights 1 (equal, as asked; no evidence to deviate).

KL is the default only if it is not worse in the smoke. The criterion is held-out token-slot agreement with the full-depth argmax, plus
token-slot CE, at K = 1/2/3 (§7).

## 5. Per micro-batch randomness (identical on every rank, deterministic from (seed, step, micro-step))

- K ~ {1:1, 2:2, 3:1} (mostly 2). K=4 (the exact lattice) is supported by the same code and can be added with `LAT_K=1,2,3,4`.
- B ~ {4, 8} (the port defaults).
- The anchors and the block selection are per rank (seeded by (seed, step, micro, rank)).
- Extra-row budget per sequence: min(`--lat_extra_max`, `--lat_extra_frac` × #valid chain rows), padded to a multiple of 16. Blocks
  are chosen at random among anchors whose slot 0 has a target, until the budget is full. The bench fixes extra_max against memory.

## 6. Recipe (full run; NOT launched)

lr 1e-5 peak, warmup 3%, cosine to ZERO at step 600 (SCHED_STEPS = MAX_STEPS = 600). The integrated LR, 600 × ~5e-6 = 3e-3, equals the
late-entry run A's checkpoint-300, which reached the v2 level. That run's damage came from staying near 1e-5 for 2000+ steps.

Other settings:
- effective batch 256 = 1 × 32 × 8 GPUs, ZeRO-2, max len 1024, one 8-GPU node.
- Checkpoints every 150 steps (weights + optimizer, none deleted).
- Data = the Dolci pipeline of train / v2 (prompt filter <= 1024, right truncation, mask_prompt_loss).
- Stage-1 collator, padded to a multiple of 16.
- bf16, sdpa, `HAS_TT_DATA` in the yaml envsList, wandb key read in memory (never written).

## 7. Tests

**CPU, fp32 (and float64), tiny random A2D-Ouro** (`test_latelat_cpu.py`):
- 0. masks: the vectorised mask == a per-entry rule written independently from the decoder's depth formula d(q).
- a. stream 1 == the plain forward (logits and SFT gradients).
- b. stream-2 logits == the decoder's block readouts, teacher-forced on the same text and anchors:
  - `late2/late.py` LateLattice (K=2, the run decoder) for B = 2, 4;
  - `ref_latek.py`, an independent HF-module step-by-step reference with explicit context K/V gathering, for K = 1, 2, 3, 4 and B = 2, 4;
  - ref_latek(K=2) == late.py (validates the reference);
  - the snapshot of `late_k/latek.py`, when present, for K = 1..4.
  - The negative control (simpler approximation: every context position read at full depth) must differ for K = 1, 2, 3.
- c. gradients: d L1 / d stream-2 inputs == 0. For single stream-2 rows, the input rows with a nonzero gradient == the hop-limited
  reachability of the mask. No target leak: a token slot never reaches x_{p+1}, a placeholder never reaches a token past its set.
- d. loss == the formula recomputed from full logits (CE and KL).
- e. trainer end to end on CPU: L1, L_tok and L_ph fall; checkpoints keep weights; a checkpoint loads into late.py's engine
  (`E.load_weights`) and decodes.

**GPU, real v2, fp32** (`check_real.py`): the same stream-2 vs late.py (K=2) and ref_latek (K=1,2,3) on Dolci text, B = 4, 8. It also
reports the size of the simpler approximation's error (max |Δlogit|, argmax disagreement), i.e. "measure the mismatch".

**Smoke job** (`run_latelat_smoke.sh`, 8 GPUs, retries 0, ≤ 30 optimizer steps per arm):
- bench (memory, s/fwd-bwd);
- check_real;
- arm KL (TAG `ouro26_latelat_smokeKL`, real config, 30-step cosine, ckpt 15/30);
- arm CE (TAG `ouro26_latelat_smokeCE`, 30 steps);
- held-out eval of init / both arms (`eval_heldout.py`: token-slot agreement with full depth, CE, placeholder acc, stream-1 CE per K);
- export of checkpoint-30, then load + decode in late.py's LateLattice (`l2:lkv:am:latt:4`, torch attention) on 3 GSM8K prompts,
  init vs ckpt.
