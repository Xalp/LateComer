# Late-entry mechanism probes (teacher forced): summary, 2026-10-26 17:19 UTC

_Raw: probes/mechanism/results/<model>/raw_<task>.pkl (mech_run.py, 10-23 jobs). Aggregation: mech_probes/mech_analyze.py (unchanged, imported) + entry_probe/mech_finish.py (extras). Figures: figures/fig_mech_*.pdf/.png (entry_probe/mech_figs2.py). CPU only._

## Reading (what the probes say)

Setup in one line: L_t = the token x_p enters as its raw embedding at pass 4-t+1, runs t passes, and reads the true full-depth K/V of every earlier position (teacher forced on v2's own text). Agreement = argmax equals the model's own 4-pass argmax. "Untrained" = sft, abl, stock26 (no mid-pass training); v2 = trained for late entry.

**P1: late entry works only after training.** On GSM8K v2's L_1 agrees .962, against .898 for v2's own 1-pass model (+6.4 pts [6.0, 6.9]). L_2 is .984 and L_3 .992. On IFEval: .887 vs .757 (+13.0), L_2 .943, L_3 .969. Late entry removes 63 % (GSM8K) and 54 % (IFEval) of the 1-pass model's disagreements. In the untrained models L_1 is far BELOW their own 1-pass model: GSM8K .70-.76 vs .88-.94 (−18 pts), IFEval .52-.59 vs .74-.82 (−17 to −25). The D×t grid shows the same split. In v2, more context depth helps a 1-pass slot: (D,1) = .898 / .945 / .962 / .962 for D = 1..4 (GSM8K) and .757 / .809 / .857 / .887 (IFEval). In the untrained models every late start (t < D) is worse than the D-pass model itself. The 1-pass slot already drops to .67-.71 at D = 2 (GSM8K).

**Untrained failures are gross; v2's remaining misses are hard positions.** Take positions where the full-depth model is at least 95 % confident (66-68 % of GSM8K positions). There v2's L_1 agrees .998, sft's .769, abl's .798 and stock26's .841. v2's misses sit where the model itself is unsure: at full-depth confidence < .5 (6 % of GSM8K positions, 24 % of IFEval) its L_1 agrees .692 / .631.

**P3: the keys-vs-values split has no separable answer.** Take keys from one pass and values from another, for every context position. The slot then collapses in all four models, trained or not:
- deep K with pass-1 V gives .30 (v2) and .08-.17 (untrained) on GSM8K;
- pass-1 K with deep V gives .000 (v2) and .03-.05 (untrained).
These are far below both coherent endpoints (deep K+V .962, pass-1 K+V .898 for v2). The argmax degenerates to one token. In v2, pass-1 K + deep V predicts `<|endoftext|>` at 100 % of positions, and deep K + pass-1 V predicts `\n` at 34 % (GSM8K) / 67 % (IFEval). P4 mixes depths ACROSS positions without any collapse (most frequent argmax ≤ 20 % in every P4 config). So the constraint is within a position: a pass-k key only works with its pass-k value. The late slot's information comes from coherent (K, V) pairs, not from keys or values alone. Any decoder that shares or caches K/V must keep each position's pair from the same pass.

**P4: where the deep-context gain comes from (v2, recent vs older positions).** The slot reads pass-4 K/V for some positions and pass-1 K/V for the rest.
- GSM8K (gap .064):
  - deep K/V for p−1 alone recovers 25 % of the gain;
  - the last 2 recover 40 %, the last 8 58 %, the last 64 86 %;
  - deep for everything but p−1 recovers 80 %, everything but the last 8 54 %;
  - prompt only 41 %, generated part only 69 %, position 0 alone 4 %.
- IFEval (gap .130):
  - p−1 alone recovers −7 % (n.s.);
  - the last 8 recover 20 % and the last 64 47 %;
  - deep for everything but the last 8 recovers 67 %;
  - prompt only 42 %, generated part only 59 %.
- Per position the newest tokens matter most. On GSM8K one deep position is worth 1.6 pts at distance 1, 1.0 at 2, 0.3 at 3-4, 0.14 at 5-8, 0.08 at 9-16 and 0.02 at 17-64.
- But the long tail is large. Adding deep K/V for the ~93 (GSM8K) / ~299 (IFEval) positions more than 64 back, on top of a deep newest 64, adds 0.9 pts (14 % of the gain) / 6.9 pts (53 %). Deep K/V for those older positions alone (newest 64 at pass 1) recovers 22 % / 48 %.

So depth helps everywhere in the context. It is recency-weighted per position, but on long texts most of it comes from older positions. For the "4 4 4 3 2 1" decoder (variant B), keeping the newest 2 positions at pass 1 costs v2 only 1.9 (GSM8K) / 2.3 (IFEval) pts against all-deep. B gives them passes 2 and 3, so this is an upper bound on B's teacher-forced cost.

**P4 in the untrained models: the deep K/V are readable, but not all at once.**
- Deep K/V for only part of the context often helps them beyond their own 1-pass slot. Deep prompt only gives +1.5 / +0.7 / −0.1 pts on GSM8K and +4.3 / +4.1 / +2.2 on IFEval (sft / abl / stock26). Deep for only the newest 64 positions gives +3.6 / +4.1 / +0.4 on GSM8K.
- The collapse needs the newest positions AND the older ones deep together:
  - deep everywhere except the newest 2: −10.8 / −12.9 / −12.8 (GSM8K);
  - deep everywhere except the newest 8: +0.6 / −1.1 / −1.6;
  - all deep: −18.3 / −18.2 / −17.7.
- On IFEval the deep GENERATED part also hurts (gen-only −5.4 / +0.4 / −9.9).

**P5 and P2: the untrained slot cannot address an all-deep context; the trained slot behaves like a pass-4 query.**
- v2's L_1 query attends almost exactly like v2's normal pass-4 query at p (layer-mean, GSM8K / IFEval):
  - entropy +0.07 / +0.13 nats;
  - mass on position 0 −.022 / −.023;
  - mass on p−7..p +.007 / +.005.
- The untrained L_1 queries attend diffusely: entropy +0.83..+1.00 / +0.79..+0.90 nats, position-0 mass −.13..−.16 / −.10..−.12, local mass −.03.
- The final states follow:
  - v2's L_1 state is close to its pass-4 state: cos .927 / .930, linear CKA .961 / .961. Its cosine with the pass-k state rises with k (.74 → .86 → .90 → .93).
  - The untrained L_1 state is unlike any normal pass state: cos ≤ .68, CKA .25-.37 with every pass.
- The logit lens shows no change in WHERE the prediction forms. All queries settle in the last ~8 layers of the pass: median settle layer 40-43 (GSM8K), 43-46 (IFEval). The untrained L_1 lags by 1-2 layers and reaches only .47-.52 agreement at layer 44 (v2 .80).

**Relation to the entry probe (coordinate gap vs information gap).** The entry probe found that the untrained late slot's failure is mostly a coordinate gap. A frozen-model linear map of the token embedding (LE) closes 70-78 % of the E→oracle agreement gap on GSM8K and 46-51 % on IFEval. The mechanism probes show what that gap looks like inside the model and agree with it:
1. The information is present in the deep K/V. v2's trained embedding query reads them to .962 / .887, and even untrained queries gain from deep K/V of part of the context (prompt; GSM8K's newest 64).
2. The failure is in addressing. The untrained raw-embedding query spreads its attention over an all-deep context, misses the sink pattern of a deep query, and ends off-manifold (CKA .25-.37). That is what a query in the wrong coordinates does. Training (v2) fixes exactly this: its L_1 attention matches pass 4.
3. What remains after training is the information gap: v2 L_1 vs full depth −3.8 (GSM8K) / −11.3 (IFEval) pts; the entry probe measures v2 E vs oracle as 3.4 / 9.6. It has three properties:
   - It sits at positions the model itself finds hard.
   - No subset of deep context positions reduces it (all-deep is the best P4 configuration).
   - It shrinks with the slot's OWN passes (L_2, L_3).
   This is the part that only the position's own processing (the oracle's pass-1..3 input) supplies.
4. P3 adds a design constraint that the entry probe could not see: whatever the entry, the slot must read same-pass (K, V) pairs.

**Tying input and output embeddings (user suggestion, relayed).** Not testable here: the mechanism probes use frozen weights and tying is a training change. The relevant evidence comes from the entry probe on these models (tie_word_embeddings = False):
- Using output rows W_out[x] as the entry zero-shot makes the slot copy x_p (92-100 %, agreement .01-.04), because W_out rows mean "the next token is v".
- E and W_out are nearly unrelated: centered cos .13.
- A linear map of W_out[x] is exactly as good as one of E[x] (LOE = LE).
The mechanism probes add that what the entry must produce is a pass-4-like query (P5). A tie would put the token's own input in the lm_head "next-token" coordinates, which belong to the predecessor's state. So we have no evidence that a tie helps beyond what the learned entry map (or v2's training) already achieves. A trained tie would be a new training run and needs the user's OK.

Caveats:
- One model family, frozen weights, teacher forcing on v2's text. That text is off-policy for sft/abl/stock26: their full-depth argmax matches the text token at .91-.96 on GSM8K and .84-.92 on IFEval.
- Attention stats are head-averaged; no head-level attribution.
- The P3/P4 mixed contexts are off-distribution for every model. They show which combinations break, not a causal decomposition.
- Numerical floor: cell(k,k) vs the k-pass forward flips ≤ 1 argmax in 30-53 k positions. The max logit diff is 5e-4 to 0.13 in fp32 on GPU (CPU test exact to 1e-5).

## Data completeness

- GSM8K: v2 300 docs / 30736 positions; sft 300 docs / 30736 positions; abl 300 docs / 30736 positions; stock26 300 docs / 30736 positions (common docs 300; requested 300; missing none). Numerical floor (argmax flips of cell(4,4) vs the full forward, fp32 GPU): v2 0, sft 0, abl 0, stock26 0.
- IFEval: v2 150 docs / 53184 positions; sft 150 docs / 53184 positions; abl 150 docs / 53184 positions; stock26 150 docs / 53184 positions (common docs 150; requested 150; missing none). Numerical floor (argmax flips of cell(4,4) vs the full forward, fp32 GPU): v2 0, sft 0, abl 0, stock26 0.

All four models (v2, sft, abl, stock26) are complete for both tasks: every requested doc is present, `.done` exists, the 4 jobs (one per model, both tasks each) exited rc=0 (logs in `_logs/`). The text is v2's own full-depth generation for every model, so it is off-policy for sft/abl/stock26: the full-depth argmax equals the text token at GSM8K v2 0.995, sft 0.950, abl 0.959, stock26 0.906; IFEval v2 0.988, sft 0.911, abl 0.922, stock26 0.844.

## Key numbers (agreement with the model's own full-depth argmax; 95 % CI over documents)

### GSM8K

| model | 1-pass model | L_1 (embedding at pass 4, deep ctx) | L_2 | L_3 | P3: K deep + V pass 1 | P3: K pass 1 + V deep | P4 best partial (which) | L_1 attn entropy − pass-4 query (nats) | L_1 sink mass − pass-4 query | CKA(L_1 state, pass-4 state) |
|---|---|---|---|---|---|---|---|---|---|---|
| v2 | 0.898 [0.893, 0.902] | 0.962 [0.960, 0.965] | 0.984 | 0.992 | 0.296 | 0.000 | 0.953 (last_64) | 0.07 [0.07, 0.08] | -0.022 [-0.022, -0.022] | 0.961 |
| sft | 0.881 [0.876, 0.886] | 0.699 [0.689, 0.707] | 0.896 | 0.926 | 0.146 | 0.031 | 0.917 (last_64) | 0.94 [0.92, 0.96] | -0.147 [-0.151, -0.144] | 0.312 |
| abl | 0.904 [0.900, 0.908] | 0.722 [0.714, 0.729] | 0.935 | 0.962 | 0.169 | 0.048 | 0.945 (last_64) | 1.00 [0.98, 1.02] | -0.158 [-0.161, -0.154] | 0.273 |
| stock26 | 0.935 [0.932, 0.939] | 0.758 [0.752, 0.764] | 0.902 | 0.929 | 0.084 | 0.029 | 0.948 (last_16) | 0.83 [0.81, 0.85] | -0.127 [-0.130, -0.123] | 0.334 |

P4, GSM8K: single-pass slot L_1 with deep (pass-4) K/V only for the named context positions and pass-1 K/V elsewhere. Row 'v2 share' = fraction of v2's deep-context gain a(4,1) − a(1,1) recovered; the other rows = agreement change vs the shallow slot a(1,1) (= the 1-pass model), in points (the all-deep column is L_1 itself).

| row | all deep (L_1) | last 1 | last 2 | last 8 | last 64 | all but last 1 | all but last 2 | all but last 8 | prompt only | generated only | pos 0 only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 share (gap 6.4 pts) | 1.00 | 0.25 | 0.40 | 0.58 | 0.86 | 0.80 | 0.71 | 0.54 | 0.41 | 0.69 | 0.04 |
| v2 Δ pts | +6.4 | +1.6 | +2.6 | +3.7 | +5.5 | +5.2 | +4.6 | +3.5 | +2.6 | +4.4 | +0.2 |
| sft Δ pts | -18.3 | -0.7 | +1.5 | +2.7 | +3.6 | -12.2 | -10.8 | +0.6 | +1.5 | +2.7 | +0.2 |
| abl Δ pts | -18.2 | +0.7 | +1.5 | +2.7 | +4.1 | -14.5 | -12.9 | -1.1 | +0.7 | +3.0 | +0.0 |
| stock26 Δ pts | -17.7 | -0.1 | +0.7 | +1.1 | +0.4 | -14.0 | -12.8 | -1.6 | -0.1 | +0.5 | -0.1 |

Per-position marginal gain (agreement points per context position, v2, GSM8K): d1-1 1.61, d2-2 0.96, d3-4 0.30, d5-8 0.14, d9-16 0.08, d17-64 0.02, d65- 0.91 (total over ~93 older positions). Mean context per query 157 positions (prompt 93, generation 103).

Collapse of the mixed-pass K/V configs (GSM8K; share of positions with that argmax): v2 K-deep/V-pass-1 → '\n' 0.34, K-pass-1/V-deep → '<|endoftext|>' 1.00; sft K-deep/V-pass-1 → '\n' 0.65, K-pass-1/V-deep → '\n' 0.93; abl K-deep/V-pass-1 → '\n' 0.61, K-pass-1/V-deep → '<|endoftext|>' 0.76; stock26 K-deep/V-pass-1 → '\n' 0.73, K-pass-1/V-deep → '\n' 0.98. For comparison the most frequent argmax of L_1 / the 1-pass model covers v2 0.12/0.11, sft 0.16/0.11, abl 0.17/0.12, stock26 0.14/0.10; L_1 copies x_p at v2 0.027, sft 0.057, abl 0.043, stock26 0.034 (1-pass model 0.028, 0.028, 0.028, 0.030).

### IFEval

| model | 1-pass model | L_1 (embedding at pass 4, deep ctx) | L_2 | L_3 | P3: K deep + V pass 1 | P3: K pass 1 + V deep | P4 best partial (which) | L_1 attn entropy − pass-4 query (nats) | L_1 sink mass − pass-4 query | CKA(L_1 state, pass-4 state) |
|---|---|---|---|---|---|---|---|---|---|---|
| v2 | 0.757 [0.735, 0.780] | 0.887 [0.876, 0.898] | 0.943 | 0.969 | 0.139 | 0.000 | 0.868 (old_1) | 0.13 [0.12, 0.13] | -0.023 [-0.024, -0.022] | 0.961 |
| sft | 0.738 [0.714, 0.762] | 0.519 [0.500, 0.538] | 0.721 | 0.748 | 0.081 | 0.042 | 0.781 (prompt) | 0.84 [0.79, 0.89] | -0.110 [-0.119, -0.103] | 0.339 |
| abl | 0.761 [0.739, 0.784] | 0.593 [0.563, 0.623] | 0.817 | 0.848 | 0.097 | 0.011 | 0.802 (prompt) | 0.90 [0.85, 0.96] | -0.121 [-0.130, -0.112] | 0.365 |
| stock26 | 0.820 [0.802, 0.838] | 0.565 [0.546, 0.584] | 0.724 | 0.768 | 0.085 | 0.045 | 0.841 (prompt) | 0.79 [0.74, 0.84] | -0.103 [-0.111, -0.095] | 0.320 |

P4, IFEval: single-pass slot L_1 with deep (pass-4) K/V only for the named context positions and pass-1 K/V elsewhere. Row 'v2 share' = fraction of v2's deep-context gain a(4,1) − a(1,1) recovered; the other rows = agreement change vs the shallow slot a(1,1) (= the 1-pass model), in points (the all-deep column is L_1 itself).

| row | all deep (L_1) | last 1 | last 2 | last 8 | last 64 | all but last 1 | all but last 2 | all but last 8 | prompt only | generated only | pos 0 only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 share (gap 13.0 pts) | 1.00 | -0.07 | 0.06 | 0.20 | 0.47 | 0.85 | 0.83 | 0.67 | 0.42 | 0.59 | 0.06 |
| v2 Δ pts | +13.0 | -0.9 | +0.7 | +2.6 | +6.1 | +11.1 | +10.8 | +8.8 | +5.4 | +7.7 | +0.8 |
| sft Δ pts | -21.9 | -4.8 | -3.2 | -1.8 | -2.9 | -15.1 | -11.9 | -4.6 | +4.3 | -5.4 | +0.7 |
| abl Δ pts | -16.8 | -3.0 | -1.5 | +0.1 | +0.6 | -12.3 | -9.0 | -2.1 | +4.1 | +0.4 | +0.6 |
| stock26 Δ pts | -25.4 | -5.9 | -5.1 | -4.2 | -6.2 | -19.4 | -16.1 | -8.3 | +2.2 | -9.9 | +0.3 |

Per-position marginal gain (agreement points per context position, v2, IFEval): d1-1 -0.87, d2-2 1.60, d3-4 0.44, d5-8 0.26, d9-16 0.12, d17-64 0.05, d65- 6.91 (total over ~299 older positions). Mean context per query 363 positions (prompt 67, generation 356).

Collapse of the mixed-pass K/V configs (IFEval; share of positions with that argmax): v2 K-deep/V-pass-1 → '\n' 0.67, K-pass-1/V-deep → '<|endoftext|>' 1.00; sft K-deep/V-pass-1 → '\n' 0.83, K-pass-1/V-deep → '\n' 1.00; abl K-deep/V-pass-1 → '\n' 0.84, K-pass-1/V-deep → '<|endoftext|>' 0.83; stock26 K-deep/V-pass-1 → '\n' 0.87, K-pass-1/V-deep → '\n' 1.00. For comparison the most frequent argmax of L_1 / the 1-pass model covers v2 0.04/0.05, sft 0.09/0.05, abl 0.11/0.05, stock26 0.15/0.05; L_1 copies x_p at v2 0.020, sft 0.065, abl 0.055, stock26 0.052 (1-pass model 0.020, 0.020, 0.020, 0.020).

## Entry-probe cross-reference (same slot = cell(4,1) with entry E; entry probe text = each model's OWN generation)

| model / task | E (= L_1) | LE (linear map of E, in-domain fit) | LEH (+ predecessor deep state) | ORA (true pass-4 input) | LE share of E→ORA gap | mech L_1 on v2 text |
|---|---|---|---|---|---|---|
| v2/gsm8k_cot_zeroshot | 0.962 | 0.941 | 0.947 | 0.996 | -0.64 | 0.962 |
| v2/ifeval | 0.887 | 0.763 | 0.835 | 0.983 | -1.29 | 0.887 |
| sft/gsm8k_cot_zeroshot | 0.706 | 0.930 | 0.939 | 0.995 | 0.78 | 0.699 |
| sft/ifeval | 0.501 | 0.719 | 0.824 | 0.979 | 0.46 | 0.519 |
| stock26/gsm8k_cot_zeroshot | 0.751 | 0.919 | 0.949 | 0.992 | 0.70 | 0.758 |
| stock26/ifeval | 0.659 | 0.828 | 0.897 | 0.989 | 0.51 | 0.565 |

# Full tables (mech_analyze.py)


Sequences: v2's own full-depth (ar:4) generations, GSM8K (gsm8k_cot_zeroshot) docs 0-299 and IFEval docs 0-149, prompt = the campaign's materialized chat-template context; positions p = every generated position whose next token is generated text. L_t(p) = a slot that enters at pass T-t+1 with the raw embedding of x_p, reads the FULL-DEPTH pass-k K/V of all positions < p (+ its own key), readout = renormalised gate mixture over its t passes. Agreement = argmax equals the model's own full-depth (4-pass) argmax at p. Numerics: fp32 weights/activations, TF32 off, readouts fp32. 95 % CIs: bootstrap over documents (paired across configs and models). Model names: **v2** = v2 (SFT+block+mid-pass), **sft** = SFT only, **abl** = block, no mid-pass, **stock26** = stock Ouro-2.6B.

## GSM8K

300 docs, 30736 positions (299 docs re-tokenize to the recorded length). Full-depth argmax = the text token at v2 .995, sft .950, abl .959, stock26 .906.

### P1 — agreement with full depth (and NLL of x_{p+1}): late entry L_t vs the t-pass model

| model | L_1 | 1-pass | L_2 | 2-pass | L_3 | 3-pass | full NLL | L_1 NLL | 1-pass NLL | L_2 NLL | 2-pass NLL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 | 0.962 [0.960, 0.965] | 0.898 [0.893, 0.902] | 0.984 [0.982, 0.985] | 0.954 [0.951, 0.957] | 0.992 [0.991, 0.993] | 0.982 [0.980, 0.983] | .137 | .174 | .507 | .149 | .184 |
| sft | 0.699 [0.689, 0.707] | 0.881 [0.876, 0.886] | 0.896 [0.891, 0.901] | 0.950 [0.947, 0.953] | 0.926 [0.921, 0.930] | 0.979 [0.977, 0.981] | .192 | 1.512 | .569 | .514 | .239 |
| abl | 0.722 [0.714, 0.729] | 0.904 [0.900, 0.908] | 0.935 [0.932, 0.938] | 0.954 [0.952, 0.957] | 0.962 [0.960, 0.965] | 0.980 [0.978, 0.982] | .176 | 1.354 | .493 | .312 | .199 |
| stock26 | 0.758 [0.752, 0.764] | 0.935 [0.932, 0.939] | 0.902 [0.896, 0.907] | 0.981 [0.979, 0.982] | 0.929 [0.925, 0.933] | 0.995 [0.994, 0.995] | .289 | 1.264 | .424 | .551 | .296 |

Paired differences L_t − t-pass (agreement, NLL):

- v2: t=1: Δagree 0.064 [0.060, 0.069], ΔNLL -0.333 [-0.357, -0.312]; t=2: Δagree 0.029 [0.026, 0.032], ΔNLL -0.036 [-0.042, -0.029]; t=3: Δagree 0.010 [0.009, 0.012], ΔNLL 0.009 [0.008, 0.010]
- sft: t=1: Δagree -0.183 [-0.192, -0.175], ΔNLL 0.943 [0.901, 0.990]; t=2: Δagree -0.054 [-0.060, -0.049], ΔNLL 0.275 [0.250, 0.303]; t=3: Δagree -0.054 [-0.059, -0.049], ΔNLL 0.282 [0.258, 0.311]
- abl: t=1: Δagree -0.182 [-0.189, -0.175], ΔNLL 0.861 [0.830, 0.895]; t=2: Δagree -0.019 [-0.023, -0.016], ΔNLL 0.113 [0.105, 0.120]; t=3: Δagree -0.018 [-0.020, -0.015], ΔNLL 0.087 [0.082, 0.091]
- stock26: t=1: Δagree -0.177 [-0.183, -0.172], ΔNLL 0.840 [0.814, 0.867]; t=2: Δagree -0.079 [-0.084, -0.074], ΔNLL 0.256 [0.234, 0.279]; t=3: Δagree -0.065 [-0.070, -0.061], ΔNLL 0.216 [0.196, 0.238]

Agreement binned by the full-depth top-1 probability at p (fraction of positions in the bin; L_1 / 1-pass / L_2 / 2-pass):

- v2: p∈[0.00-0.50) .06: .692/.452/.827/.652; p∈[0.50-0.80) .14: .882/.684/.959/.852; p∈[0.80-0.95) .12: .985/.879/.999/.968; p∈[0.95-1.00) .68: .998/.983/1.000/.999
- sft: p∈[0.00-0.50) .07: .358/.395/.624/.606; p∈[0.50-0.80) .14: .543/.671/.812/.871; p∈[0.80-0.95) .12: .678/.863/.912/.973; p∈[0.95-1.00) .67: .769/.977/.939/.997
- abl: p∈[0.00-0.50) .06: .358/.495/.640/.682; p∈[0.50-0.80) .14: .530/.728/.853/.856; p∈[0.80-0.95) .12: .722/.892/.963/.972; p∈[0.95-1.00) .67: .798/.984/.976/.999
- stock26: p∈[0.00-0.50) .07: .381/.532/.556/.779; p∈[0.50-0.80) .14: .591/.809/.793/.971; p∈[0.80-0.95) .13: .719/.969/.921/1.000; p∈[0.95-1.00) .66: .841/.999/.958/1.000

### D × t grid (context run to D passes, slot makes t passes at D-t+1..D): agreement with full depth / NLL

**v2**  (1,1) .898/0.51 · (2,1) .945/0.24 · (2,2) .954/0.18 · (3,1) .962/0.16 · (3,2) .979/0.14 · (3,3) .982/0.13 · (4,1) .962/0.17 · (4,2) .984/0.15 · (4,3) .992/0.14 · (4,4) 1.000/0.14  
**sft**  (1,1) .881/0.57 · (2,1) .670/1.96 · (2,2) .950/0.24 · (3,1) .684/1.69 · (3,2) .893/0.53 · (3,3) .979/0.20 · (4,1) .699/1.51 · (4,2) .896/0.51 · (4,3) .926/0.48 · (4,4) 1.000/0.19  
**abl**  (1,1) .904/0.49 · (2,1) .710/1.79 · (2,2) .954/0.20 · (3,1) .717/1.51 · (3,2) .931/0.34 · (3,3) .980/0.17 · (4,1) .722/1.35 · (4,2) .935/0.31 · (4,3) .962/0.26 · (4,4) 1.000/0.18  
**stock26**  (1,1) .935/0.42 · (2,1) .707/1.81 · (2,2) .981/0.30 · (3,1) .734/1.46 · (3,2) .911/0.52 · (3,3) .995/0.29 · (4,1) .758/1.26 · (4,2) .902/0.55 · (4,3) .929/0.50 · (4,4) 1.000/0.29  

### P2 — representation of L_1's final state

| model | cos(L1,P1) | cos(L1,P2) | cos(L1,P3) | cos(L1,P4) | cos(P1,P4) | cos(P3,P4) | CKA L1~P1 | CKA L1~P4 | CKA P1~P4 |
|---|---|---|---|---|---|---|---|---|---|
| v2 | 0.740 [0.736, 0.744] | 0.857 [0.855, 0.860] | 0.899 [0.898, 0.900] | 0.927 [0.926, 0.928] | 0.683 [0.679, 0.687] | 0.933 [0.932, 0.935] | 0.779 [0.772, 0.787] | 0.961 [0.961, 0.963] | 0.756 [0.749, 0.763] |
| sft | 0.521 [0.514, 0.527] | 0.584 [0.577, 0.590] | 0.612 [0.606, 0.619] | 0.627 [0.621, 0.633] | 0.692 [0.687, 0.696] | 0.953 [0.952, 0.955] | 0.277 [0.272, 0.285] | 0.312 [0.305, 0.322] | 0.785 [0.779, 0.794] |
| abl | 0.470 [0.465, 0.476] | 0.547 [0.541, 0.553] | 0.583 [0.578, 0.589] | 0.647 [0.642, 0.653] | 0.656 [0.652, 0.660] | 0.920 [0.919, 0.922] | 0.252 [0.248, 0.259] | 0.273 [0.270, 0.281] | 0.746 [0.739, 0.751] |
| stock26 | 0.580 [0.575, 0.584] | 0.668 [0.663, 0.672] | 0.681 [0.677, 0.685] | 0.682 [0.678, 0.686] | 0.708 [0.704, 0.711] | 0.985 [0.985, 0.986] | 0.297 [0.294, 0.304] | 0.334 [0.327, 0.344] | 0.848 [0.843, 0.854] |

Logit lens (lm_head(norm(residual)) after each of the 48 layers of ONE pass; agreement with the full-depth argmax). `settle` = first layer from which the lens agrees through layer 48 (positions whose final readout agrees):

| model | query | agree @12 | @24 | @36 | @44 | @48 | settle mean | settle median [p10,p90] |
|---|---|---|---|---|---|---|---|---|
| v2 | L_1 (deep ctx) | 0.002 | 0.144 | 0.181 | 0.798 | 0.962 | 40.6 [40.5, 40.6] | 41 [36, 46] |
| v2 | normal pass 1 | 0.003 | 0.090 | 0.151 | 0.769 | 0.898 | 40.3 [40.2, 40.4] | 41 [36, 46] |
| v2 | normal pass 4 | 0.015 | 0.130 | 0.163 | 0.744 | 0.996 | 41.0 [40.9, 41.1] | 42 [36, 47] |
| sft | L_1 (deep ctx) | 0.000 | 0.013 | 0.066 | 0.469 | 0.699 | 42.5 [42.4, 42.6] | 43 [37, 47] |
| sft | normal pass 1 | 0.003 | 0.075 | 0.154 | 0.723 | 0.881 | 40.7 [40.6, 40.7] | 42 [36, 46] |
| sft | normal pass 4 | 0.015 | 0.100 | 0.165 | 0.708 | 0.993 | 41.3 [41.2, 41.4] | 42 [36, 47] |
| abl | L_1 (deep ctx) | 0.000 | 0.014 | 0.088 | 0.524 | 0.722 | 42.0 [41.9, 42.0] | 42 [37, 47] |
| abl | normal pass 1 | 0.002 | 0.100 | 0.167 | 0.799 | 0.904 | 39.8 [39.7, 39.9] | 40 [35, 45] |
| abl | normal pass 4 | 0.013 | 0.135 | 0.178 | 0.738 | 0.994 | 40.9 [40.8, 41.0] | 42 [36, 47] |
| stock26 | L_1 (deep ctx) | 0.000 | 0.016 | 0.091 | 0.510 | 0.758 | 42.4 [42.3, 42.5] | 43 [37, 47] |
| stock26 | normal pass 1 | 0.003 | 0.086 | 0.193 | 0.777 | 0.935 | 40.4 [40.3, 40.5] | 41 [35, 46] |
| stock26 | normal pass 4 | 0.024 | 0.133 | 0.246 | 0.705 | 0.991 | 40.4 [40.3, 40.5] | 42 [34, 47] |

### P3 — keys vs values (L_1 reading deep/pass-1 K and V): agreement / NLL

| model | K deep + V deep (=L_1) | K deep + V pass-1 | K pass-1 + V deep | both pass-1 (=1-pass) |
|---|---|---|---|---|
| v2 | 0.962 [0.960, 0.965] / 0.174 | 0.296 [0.288, 0.305] / 6.035 | 0.000 [0.000, 0.000] / 6.895 | 0.898 [0.893, 0.902] / 0.507 |
| sft | 0.699 [0.689, 0.707] / 1.512 | 0.146 [0.140, 0.152] / 6.531 | 0.031 [0.028, 0.034] / 6.701 | 0.881 [0.876, 0.886] / 0.569 |
| abl | 0.722 [0.714, 0.729] / 1.354 | 0.169 [0.163, 0.175] / 6.520 | 0.048 [0.045, 0.051] / 6.351 | 0.904 [0.900, 0.908] / 0.493 |
| stock26 | 0.758 [0.752, 0.764] / 1.264 | 0.084 [0.079, 0.088] / 7.202 | 0.029 [0.027, 0.031] / 6.482 | 0.935 [0.932, 0.939] / 0.424 |

### P4 — which context positions need depth (L_1, deep K/V for a subset, pass-1 K/V elsewhere): agreement

| model | variant | m=0 | 1 | 2 | 4 | 8 | 16 | 64 | all |
|---|---|---|---|---|---|---|---|---|---|
| v2 | deep for the LAST m | .898 | .914 | .924 | .930 | .935 | .942 | .953 | .962 |
| v2 | deep for OLDER than m | .962 | .950 | .944 | .938 | .933 | .926 | .912 | .898 |
| sft | deep for the LAST m | .881 | .874 | .896 | .904 | .908 | .915 | .917 | .699 |
| sft | deep for OLDER than m | .699 | .760 | .773 | .850 | .887 | .896 | .895 | .881 |
| abl | deep for the LAST m | .904 | .911 | .919 | .926 | .931 | .937 | .945 | .722 |
| abl | deep for OLDER than m | .722 | .759 | .775 | .857 | .893 | .911 | .915 | .904 |
| stock26 | deep for the LAST m | .935 | .935 | .942 | .945 | .946 | .948 | .940 | .758 |
| stock26 | deep for OLDER than m | .758 | .796 | .808 | .881 | .919 | .934 | .940 | .935 |

Partition variants (deep K/V only for …): v2: prompt .924, generated part .942, position 0 only .900, all but position 0 .926; sft: prompt .896, generated part .908, position 0 only .883, all but position 0 .684; abl: prompt .911, generated part .934, position 0 only .904, all but position 0 .712; stock26: prompt .934, generated part .940, position 0 only .934, all but position 0 .722

### P5 — attention of the query at p (head-averaged, mean over the 48 layers)

| model | query | entropy (nats) | mass on p-7..p | mass on pos 0 | mass on itself |
|---|---|---|---|---|---|
| v2 | L_1 (deep ctx) | 1.56 [1.55, 1.57] | 0.148 [0.147, 0.149] | 0.617 [0.616, 0.618] | 0.035 [0.035, 0.036] |
| v2 | normal pass 1 | 1.72 [1.72, 1.73] | 0.141 [0.141, 0.142] | 0.057 [0.056, 0.057] | 0.031 [0.031, 0.031] |
| v2 | normal pass 4 | 1.48 [1.48, 1.49] | 0.141 [0.140, 0.142] | 0.639 [0.638, 0.640] | 0.044 [0.043, 0.044] |
| sft | L_1 (deep ctx) | 2.44 [2.41, 2.47] | 0.107 [0.105, 0.108] | 0.494 [0.491, 0.498] | 0.035 [0.035, 0.036] |
| sft | normal pass 1 | 1.76 [1.75, 1.76] | 0.149 [0.148, 0.149] | 0.042 [0.042, 0.042] | 0.034 [0.033, 0.034] |
| sft | normal pass 4 | 1.50 [1.49, 1.51] | 0.135 [0.134, 0.136] | 0.642 [0.640, 0.643] | 0.045 [0.045, 0.045] |
| abl | L_1 (deep ctx) | 2.48 [2.46, 2.51] | 0.110 [0.108, 0.111] | 0.483 [0.480, 0.487] | 0.034 [0.033, 0.034] |
| abl | normal pass 1 | 1.71 [1.70, 1.71] | 0.144 [0.144, 0.145] | 0.051 [0.051, 0.052] | 0.032 [0.032, 0.032] |
| abl | normal pass 4 | 1.48 [1.47, 1.49] | 0.141 [0.140, 0.142] | 0.641 [0.640, 0.642] | 0.044 [0.044, 0.044] |
| stock26 | L_1 (deep ctx) | 2.40 [2.38, 2.42] | 0.114 [0.113, 0.115] | 0.496 [0.493, 0.500] | 0.038 [0.038, 0.038] |
| stock26 | normal pass 1 | 1.83 [1.82, 1.84] | 0.148 [0.147, 0.149] | 0.036 [0.036, 0.037] | 0.034 [0.034, 0.034] |
| stock26 | normal pass 4 | 1.57 [1.56, 1.58] | 0.146 [0.145, 0.147] | 0.623 [0.622, 0.624] | 0.048 [0.047, 0.048] |

Paired L_1 − normal pass-4 query (layer mean): v2: entropy 0.07 [0.07, 0.08], last8 0.007 [0.007, 0.007], sink -0.022 [-0.022, -0.022], self -0.008 [-0.008, -0.008]; sft: entropy 0.94 [0.92, 0.96], last8 -0.029 [-0.030, -0.028], sink -0.147 [-0.151, -0.144], self -0.009 [-0.010, -0.009]; abl: entropy 1.00 [0.98, 1.02], last8 -0.032 [-0.033, -0.031], sink -0.158 [-0.161, -0.154], self -0.010 [-0.011, -0.010]; stock26: entropy 0.83 [0.81, 0.85], last8 -0.033 [-0.033, -0.032], sink -0.127 [-0.130, -0.123], self -0.010 [-0.010, -0.010]

Runtime exactness checks (max |logit diff|, cell(k,k) vs k-pass forward, fp32): v2 1.9e-03; sft 1.6e-03; abl 1.1e-03; stock26 5.6e-04

## IFEval

150 docs, 53184 positions (144 docs re-tokenize to the recorded length). Full-depth argmax = the text token at v2 .988, sft .911, abl .922, stock26 .844.

### P1 — agreement with full depth (and NLL of x_{p+1}): late entry L_t vs the t-pass model

| model | L_1 | 1-pass | L_2 | 2-pass | L_3 | 3-pass | full NLL | L_1 NLL | 1-pass NLL | L_2 NLL | 2-pass NLL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 | 0.887 [0.876, 0.898] | 0.757 [0.735, 0.780] | 0.943 [0.937, 0.948] | 0.866 [0.853, 0.880] | 0.969 [0.966, 0.972] | 0.949 [0.944, 0.955] | .400 | .528 | 1.779 | .444 | .753 |
| sft | 0.519 [0.500, 0.538] | 0.738 [0.714, 0.762] | 0.721 [0.695, 0.746] | 0.859 [0.845, 0.874] | 0.748 [0.724, 0.771] | 0.954 [0.949, 0.959] | .450 | 3.058 | 1.875 | 1.637 | .791 |
| abl | 0.593 [0.563, 0.623] | 0.761 [0.739, 0.784] | 0.817 [0.799, 0.835] | 0.870 [0.857, 0.883] | 0.848 [0.832, 0.864] | 0.950 [0.945, 0.955] | .436 | 2.292 | 1.750 | .907 | .729 |
| stock26 | 0.565 [0.546, 0.584] | 0.820 [0.802, 0.838] | 0.724 [0.699, 0.748] | 0.935 [0.928, 0.942] | 0.768 [0.743, 0.790] | 0.984 [0.982, 0.986] | .546 | 2.817 | 1.228 | 1.758 | .633 |

Paired differences L_t − t-pass (agreement, NLL):

- v2: t=1: Δagree 0.130 [0.117, 0.143], ΔNLL -1.251 [-1.382, -1.119]; t=2: Δagree 0.076 [0.067, 0.086], ΔNLL -0.309 [-0.357, -0.264]; t=3: Δagree 0.020 [0.016, 0.023], ΔNLL -0.004 [-0.012, 0.002]
- sft: t=1: Δagree -0.219 [-0.247, -0.191], ΔNLL 1.183 [0.861, 1.590]; t=2: Δagree -0.138 [-0.168, -0.110], ΔNLL 0.847 [0.588, 1.184]; t=3: Δagree -0.206 [-0.231, -0.181], ΔNLL 1.292 [1.074, 1.548]
- abl: t=1: Δagree -0.168 [-0.185, -0.150], ΔNLL 0.542 [0.409, 0.667]; t=2: Δagree -0.053 [-0.062, -0.045], ΔNLL 0.177 [0.132, 0.221]; t=3: Δagree -0.102 [-0.115, -0.091], ΔNLL 0.414 [0.370, 0.461]
- stock26: t=1: Δagree -0.254 [-0.278, -0.232], ΔNLL 1.589 [1.339, 1.931]; t=2: Δagree -0.211 [-0.236, -0.188], ΔNLL 1.125 [0.875, 1.446]; t=3: Δagree -0.216 [-0.241, -0.194], ΔNLL 1.112 [0.904, 1.361]

Agreement binned by the full-depth top-1 probability at p (fraction of positions in the bin; L_1 / 1-pass / L_2 / 2-pass):

- v2: p∈[0.00-0.50) .24: .631/.349/.796/.556; p∈[0.50-0.80) .18: .893/.657/.963/.876; p∈[0.80-0.95) .12: .982/.873/.996/.978; p∈[0.95-1.00) .45: .996/.983/.998/.998
- sft: p∈[0.00-0.50) .24: .288/.321/.521/.541; p∈[0.50-0.80) .19: .490/.626/.734/.865; p∈[0.80-0.95) .11: .616/.843/.829/.972; p∈[0.95-1.00) .46: .628/.977/.794/.997
- abl: p∈[0.00-0.50) .24: .299/.351/.552/.566; p∈[0.50-0.80) .19: .525/.669/.790/.884; p∈[0.80-0.95) .12: .665/.888/.900/.981; p∈[0.95-1.00) .45: .760/.985/.948/.998
- stock26: p∈[0.00-0.50) .20: .287/.428/.503/.727; p∈[0.50-0.80) .19: .499/.741/.722/.955; p∈[0.80-0.95) .12: .643/.920/.810/.993; p∈[0.95-1.00) .49: .688/.988/.796/.999

### D × t grid (context run to D passes, slot makes t passes at D-t+1..D): agreement with full depth / NLL

**v2**  (1,1) .757/1.78 · (2,1) .809/1.27 · (2,2) .866/0.75 · (3,1) .857/0.81 · (3,2) .927/0.47 · (3,3) .949/0.42 · (4,1) .887/0.53 · (4,2) .943/0.44 · (4,3) .969/0.42 · (4,4) 1.000/0.40  
**sft**  (1,1) .738/1.88 · (2,1) .419/4.41 · (2,2) .859/0.79 · (3,1) .485/3.53 · (3,2) .685/1.96 · (3,3) .954/0.47 · (4,1) .519/3.06 · (4,2) .721/1.64 · (4,3) .748/1.76 · (4,4) 1.000/0.45  
**abl**  (1,1) .761/1.75 · (2,1) .542/3.81 · (2,2) .870/0.73 · (3,1) .576/3.06 · (3,2) .795/1.23 · (3,3) .950/0.45 · (4,1) .593/2.29 · (4,2) .817/0.91 · (4,3) .848/0.86 · (4,4) 1.000/0.44  
**stock26**  (1,1) .820/1.23 · (2,1) .445/4.20 · (2,2) .935/0.63 · (3,1) .510/3.55 · (3,2) .715/1.89 · (3,3) .984/0.55 · (4,1) .565/2.82 · (4,2) .724/1.76 · (4,3) .768/1.66 · (4,4) 1.000/0.55  

### P2 — representation of L_1's final state

| model | cos(L1,P1) | cos(L1,P2) | cos(L1,P3) | cos(L1,P4) | cos(P1,P4) | cos(P3,P4) | CKA L1~P1 | CKA L1~P4 | CKA P1~P4 |
|---|---|---|---|---|---|---|---|---|---|
| v2 | 0.769 [0.760, 0.778] | 0.869 [0.863, 0.875] | 0.913 [0.909, 0.916] | 0.930 [0.928, 0.933] | 0.736 [0.727, 0.745] | 0.961 [0.958, 0.964] | 0.693 [0.668, 0.752] | 0.961 [0.959, 0.969] | 0.679 [0.650, 0.750] |
| sft | 0.524 [0.511, 0.537] | 0.584 [0.571, 0.598] | 0.605 [0.592, 0.620] | 0.611 [0.598, 0.625] | 0.730 [0.720, 0.739] | 0.971 [0.969, 0.974] | 0.254 [0.227, 0.354] | 0.339 [0.316, 0.433] | 0.692 [0.658, 0.759] |
| abl | 0.522 [0.507, 0.536] | 0.584 [0.569, 0.600] | 0.618 [0.602, 0.634] | 0.651 [0.634, 0.669] | 0.734 [0.727, 0.742] | 0.957 [0.954, 0.960] | 0.256 [0.219, 0.360] | 0.365 [0.329, 0.471] | 0.679 [0.652, 0.750] |
| stock26 | 0.581 [0.569, 0.594] | 0.623 [0.610, 0.634] | 0.621 [0.609, 0.632] | 0.618 [0.607, 0.628] | 0.781 [0.773, 0.789] | 0.989 [0.988, 0.990] | 0.286 [0.269, 0.357] | 0.320 [0.303, 0.399] | 0.782 [0.763, 0.825] |

Logit lens (lm_head(norm(residual)) after each of the 48 layers of ONE pass; agreement with the full-depth argmax). `settle` = first layer from which the lens agrees through layer 48 (positions whose final readout agrees):

| model | query | agree @12 | @24 | @36 | @44 | @48 | settle mean | settle median [p10,p90] |
|---|---|---|---|---|---|---|---|---|
| v2 | L_1 (deep ctx) | 0.004 | 0.028 | 0.106 | 0.523 | 0.887 | 43.0 [42.8, 43.3] | 44 [37, 48] |
| v2 | normal pass 1 | 0.005 | 0.022 | 0.094 | 0.506 | 0.757 | 42.4 [42.1, 42.6] | 43 [37, 47] |
| v2 | normal pass 4 | 0.033 | 0.047 | 0.108 | 0.500 | 0.983 | 43.6 [43.4, 43.8] | 45 [38, 48] |
| sft | L_1 (deep ctx) | 0.003 | 0.007 | 0.023 | 0.215 | 0.519 | 44.8 [44.6, 45.0] | 46 [39, 48] |
| sft | normal pass 1 | 0.006 | 0.021 | 0.103 | 0.465 | 0.738 | 42.6 [42.3, 42.9] | 44 [37, 47] |
| sft | normal pass 4 | 0.042 | 0.047 | 0.109 | 0.468 | 0.978 | 43.7 [43.5, 43.9] | 45 [38, 48] |
| abl | L_1 (deep ctx) | 0.002 | 0.007 | 0.035 | 0.299 | 0.593 | 44.2 [44.0, 44.4] | 46 [39, 48] |
| abl | normal pass 1 | 0.005 | 0.021 | 0.090 | 0.514 | 0.761 | 42.3 [42.1, 42.6] | 43 [37, 47] |
| abl | normal pass 4 | 0.033 | 0.045 | 0.108 | 0.501 | 0.981 | 43.6 [43.4, 43.8] | 45 [38, 48] |
| stock26 | L_1 (deep ctx) | 0.002 | 0.006 | 0.025 | 0.237 | 0.565 | 44.7 [44.5, 44.8] | 46 [39, 48] |
| stock26 | normal pass 1 | 0.005 | 0.022 | 0.114 | 0.500 | 0.820 | 42.7 [42.4, 43.1] | 44 [37, 47] |
| stock26 | normal pass 4 | 0.044 | 0.051 | 0.112 | 0.482 | 0.980 | 43.5 [43.3, 43.8] | 45 [38, 48] |

### P3 — keys vs values (L_1 reading deep/pass-1 K and V): agreement / NLL

| model | K deep + V deep (=L_1) | K deep + V pass-1 | K pass-1 + V deep | both pass-1 (=1-pass) |
|---|---|---|---|---|
| v2 | 0.887 [0.876, 0.898] / 0.528 | 0.139 [0.123, 0.154] / 7.833 | 0.000 [0.000, 0.000] / 9.027 | 0.757 [0.735, 0.780] / 1.779 |
| sft | 0.519 [0.500, 0.538] / 3.058 | 0.081 [0.071, 0.092] / 8.099 | 0.042 [0.034, 0.051] / 9.115 | 0.738 [0.714, 0.762] / 1.875 |
| abl | 0.593 [0.563, 0.623] / 2.292 | 0.097 [0.085, 0.109] / 8.045 | 0.011 [0.009, 0.014] / 8.818 | 0.761 [0.739, 0.784] / 1.750 |
| stock26 | 0.565 [0.546, 0.584] / 2.817 | 0.085 [0.074, 0.096] / 8.244 | 0.045 [0.037, 0.053] / 8.916 | 0.820 [0.802, 0.838] / 1.228 |

### P4 — which context positions need depth (L_1, deep K/V for a subset, pass-1 K/V elsewhere): agreement

| model | variant | m=0 | 1 | 2 | 4 | 8 | 16 | 64 | all |
|---|---|---|---|---|---|---|---|---|---|
| v2 | deep for the LAST m | .757 | .748 | .764 | .773 | .783 | .792 | .818 | .887 |
| v2 | deep for OLDER than m | .887 | .868 | .864 | .853 | .845 | .836 | .819 | .757 |
| sft | deep for the LAST m | .738 | .690 | .706 | .715 | .720 | .722 | .710 | .519 |
| sft | deep for OLDER than m | .519 | .587 | .619 | .657 | .692 | .719 | .755 | .738 |
| abl | deep for the LAST m | .761 | .731 | .746 | .755 | .762 | .768 | .767 | .593 |
| abl | deep for OLDER than m | .593 | .638 | .671 | .708 | .739 | .763 | .789 | .761 |
| stock26 | deep for the LAST m | .820 | .761 | .768 | .774 | .778 | .780 | .758 | .565 |
| stock26 | deep for OLDER than m | .565 | .626 | .659 | .700 | .737 | .764 | .809 | .820 |

Partition variants (deep K/V only for …): v2: prompt .811, generated part .833, position 0 only .765, all but position 0 .782; sft: prompt .781, generated part .684, position 0 only .745, all but position 0 .467; abl: prompt .802, generated part .765, position 0 only .767, all but position 0 .562; stock26: prompt .841, generated part .721, position 0 only .822, all but position 0 .456

### P5 — attention of the query at p (head-averaged, mean over the 48 layers)

| model | query | entropy (nats) | mass on p-7..p | mass on pos 0 | mass on itself |
|---|---|---|---|---|---|
| v2 | L_1 (deep ctx) | 1.71 [1.68, 1.73] | 0.137 [0.133, 0.141] | 0.631 [0.627, 0.635] | 0.036 [0.035, 0.037] |
| v2 | normal pass 1 | 1.83 [1.81, 1.85] | 0.131 [0.128, 0.134] | 0.051 [0.051, 0.052] | 0.031 [0.030, 0.033] |
| v2 | normal pass 4 | 1.58 [1.55, 1.60] | 0.132 [0.129, 0.135] | 0.653 [0.649, 0.658] | 0.045 [0.044, 0.047] |
| sft | L_1 (deep ctx) | 2.43 [2.37, 2.49] | 0.100 [0.097, 0.103] | 0.547 [0.536, 0.557] | 0.035 [0.034, 0.036] |
| sft | normal pass 1 | 1.80 [1.78, 1.82] | 0.135 [0.131, 0.138] | 0.039 [0.038, 0.039] | 0.033 [0.032, 0.034] |
| sft | normal pass 4 | 1.59 [1.57, 1.62] | 0.127 [0.123, 0.132] | 0.657 [0.652, 0.662] | 0.046 [0.045, 0.048] |
| abl | L_1 (deep ctx) | 2.49 [2.43, 2.55] | 0.104 [0.101, 0.107] | 0.531 [0.519, 0.542] | 0.032 [0.031, 0.033] |
| abl | normal pass 1 | 1.82 [1.80, 1.84] | 0.133 [0.130, 0.136] | 0.047 [0.046, 0.047] | 0.032 [0.030, 0.033] |
| abl | normal pass 4 | 1.59 [1.56, 1.62] | 0.133 [0.130, 0.136] | 0.652 [0.648, 0.656] | 0.046 [0.044, 0.047] |
| stock26 | L_1 (deep ctx) | 2.39 [2.33, 2.45] | 0.106 [0.102, 0.109] | 0.553 [0.541, 0.564] | 0.039 [0.038, 0.040] |
| stock26 | normal pass 1 | 1.88 [1.86, 1.91] | 0.136 [0.132, 0.139] | 0.034 [0.033, 0.034] | 0.034 [0.033, 0.035] |
| stock26 | normal pass 4 | 1.59 [1.57, 1.62] | 0.134 [0.130, 0.140] | 0.656 [0.650, 0.661] | 0.048 [0.046, 0.049] |

Paired L_1 − normal pass-4 query (layer mean): v2: entropy 0.13 [0.12, 0.13], last8 0.005 [0.004, 0.006], sink -0.023 [-0.024, -0.022], self -0.009 [-0.010, -0.009]; sft: entropy 0.84 [0.79, 0.89], last8 -0.028 [-0.030, -0.025], sink -0.110 [-0.119, -0.103], self -0.011 [-0.013, -0.009]; abl: entropy 0.90 [0.85, 0.96], last8 -0.029 [-0.031, -0.027], sink -0.121 [-0.130, -0.112], self -0.014 [-0.015, -0.012]; stock26: entropy 0.79 [0.74, 0.84], last8 -0.029 [-0.031, -0.027], sink -0.103 [-0.111, -0.095], self -0.009 [-0.010, -0.007]

Runtime exactness checks (max |logit diff|, cell(k,k) vs k-pass forward, fp32): v2 4.1e-03; sft 1.3e-01; abl 1.1e-02; stock26 2.0e-03



## Verification addendum (10-27, independent check)

- **Tests:** `test_mech_cpu.py` re-run: ALL PASSED (worst 9.18e-6).
- **Numbers recomputed from the raw pickles** with separate code, all of which hold:
  - P1: L_1 / L_2 / L_3 and the 1-pass model for all four models and both tasks;
  - the (D, t) grid. Every late start (t < D) is below the D-pass model in all three untrained models.
  - P3 agreements and the collapse tokens;
  - P4 shares and the per-position gains;
  - the newest-2-at-pass-1 cost (1.9 / 2.3 pts);
  - the >= .95 and < .5 confidence splits;
  - the attention entropy, sink and local-mass differences;
  - the state cosines and CKA;
  - the off-policy rates.
  Numerical floor: cell(k,k) vs the k-pass model flips at most 1 argmax per configuration. The max logit difference is 0.13 (sft IFEval, cell(4,4)): fp32 drift that compounds over 4 x 48 layers, with no argmax effect.
- **Corrections, made by rerunning mech_finish.py; all numbers unchanged, backup in `_bak/`:**
  - The older-than-64 figure (14 % / 53 %) is the gain from ADDING those positions on top of a deep newest 64. Deep K/V for those positions ALONE recovers 22 % (GSM8K) / 48 % (IFEval).
  - There were 4 jobs (one per model, each running both tasks), not 8.
- **Precision notes on the relayed report (this file was already right):**
  - With pass-1 K + deep V, abl predicts `<|endoftext|>` (76 % GSM8K / 83 % IFEval), like v2. Only sft and stock26 go to `\n`.
  - The deep-prompt-only effect on GSM8K is +1.5 / +0.7 / −0.1 (sft / abl / stock26): stock26 gains nothing.
  - "Deep everywhere except the newest 8 costs about ±1" holds on GSM8K (+0.6 / −1.1 / −1.6). On IFEval it costs −2 to −8.
