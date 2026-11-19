# LoopAhead reproduction plan — Stage 1 (feasibility)

**Paper:** LoopAhead: Speculating Ahead for Faster Decoding in Looped Language Models,
arXiv 2610.21594v1 (2026-10-15). Liu, Xue, Bao, Xu, Cao, Yao, Chen. **No released code.**

**What this is:** OUR best-effort reimplementation, to be measured under the Latecomer
matched-document protocol on this cluster. Every number it produces will be labelled
"our reimplementation of LoopAhead", never presented as theirs. Where the paper
underspecifies, we take the interpretation most favourable to their method; every such
choice is recorded in §4 below and will be echoed in the result files' `meta`.

Everything here was extracted from the arXiv HTML (136,811 chars, read end to end in five
windowed passes on 2026-10-30). Facts are tagged with the paper's section/table.

Status: **Stage 2 complete (2026-11-03): teacher data 9,025+782 assembled & verified; LAD drafter trained (job 07f32e55bf3aab02, 13,536 upd, dev CE 3.519, GSM8K q=[.747,.494,.240]); Stage 3 gated on the Stage-2 report.**

---

## 1. The algorithm, written out

### 1.1 Setting and notation

A looped transformer applies a shared stack repeatedly: Z^(r) = F_θ(Z^(r−1)), r = 1..R
(§2; the paper writes R for recurrence depth). For Ouro-2.6B: 48 shared layers, hidden
2048, FFN 5632, 16 heads × head-dim 128, vocab 49,152, R = 4 (their A.1 — matches the
released checkpoint's config.json exactly). Target model is FROZEN and RELEASED — no
fine-tuning anywhere in the method.

Per-model hyperparameters (their Table 7); the Ouro row is the one we need:

| group | d1 (draft/anchor depth) | d2 (feedback depth) | R | K | B_ret | B_exe | d_guard | stride |
|---|---|---|---|---|---|---|---|---|
| **Ouro (all three)** | **1** | **2** | **4** | **3** | **32** | **16** | **2** | **1** |
| Raven-Llama / OLMo | 4 | 8 | 32 | 2 | 32 | 16 | 1 | 2 |
| Raven-TinyLlama | 2 | 8 | 32 | 3 | 48 | 32 | 1 | 2 |

### 1.2 The speculation tree (§3.2, Appendix D operator list)

State is a **tree of candidate nodes** rooted at the last committed token. Each node
stores: its token, parent, hidden state (the carried inter-pass state), per-depth KV it
has written, and its completed loop depth. Two sources add children:

- **Native proposals:** the target's own readout at an early depth. Greedy anchor
  (their Eq. 2): a_t = argmax_v [W_out h_t^(r)]_v over the FULL target vocab, where
  h_t^(r) is the pre-head state after r loops. For Ouro, r = d1 = 1 (pass-1 readout).
- **LAD drafts:** chains of up to K = 3 tokens from the drafter (§1.3). Admission: after
  a loop-step, at most one native-proposal child whose parent just reached depth d1
  becomes an anchor (preferring nodes nearest the root; the anchor must have a cached
  context window and no existing children). A LAD call happens when retention capacity
  is free; it retains the first min(K, f) drafted tokens, f = free retention slots
  (their A.5). Drafted tokens can themselves anchor later LAD calls ("recursive
  drafting", their App. E).
- Proposals with the same (parent, token) merge into one node.

**Scheduling / suspend-resume** (§3.2): per loop-step, select ≤ B_exe = 16 eligible
nodes and advance each ONE loop (stride 1 for Ouro) in one batched forward. Eligible =
path is conflict-free, OR the node lies within d_guard = 2 token positions of the
committed prefix (the "protected region" — protected nodes are never suspended).
**Conflict (greedy):** a node's token differs from its parent's prediction at the
deepest feedback depth the parent has reached (for Ouro: d2 = 2, a single feedback
depth). A path conflicts if any token on it conflicts. Conflicting paths outside the
protected region are **suspended**: their states and KV are kept, nothing recomputed on
resume; they resume if later feedback resolves the conflict or root advancement brings
them into the protected region. B_ret = 32 caps retained nodes. Execution priority
(Ouro, greedy): nodes closer to the committed root first.

**Verification and commit (greedy):** when a node chain reaches depth R = 4, the
full-depth readout at a position determines the next token (argmax). If a retained
child (native or LAD) matches it, that child becomes the new root, keeps all its
computed states and KV (no recomputation), and can itself commit within the same step
if its full-depth readout is ready; competing subtrees are discarded. If no retained
child matches, the token is still emitted (it is the model's own full-depth argmax —
the "bonus" token) and a fresh node is spawned at depth 0 for it, states computed from
the committed prefix. **Greedy output is therefore exact by construction** — every
committed token is the full-depth argmax given its committed prefix. Their §4.2 states
greedy LoopAhead reproduces AR outputs; no accuracy tables are given (none needed).

Sampling mode exists (native-proposal acceptance + residual correction
(p ⊖ q)(v) = [p(v) − q(v)]₊ / Σ_u [p(u) − q(u)]₊ against the full-depth distribution,
their App. D) — **out of scope here**: our protocol is batch-1 greedy (§5).

### 1.3 The LAD drafter (§3.1, their A.5)

Architecture (≈ 11.1M params for Ouro, stated in their text):

- Input: a causal window of w = 64 cached depth-d1 target states (2048-d) ending at the
  anchor position, plus the frozen embedding of the anchor token a_t.
- The context window is projected to the drafter hidden size 512. For each future
  position j = 1..K, a query is built "combining the anchor embedding, the latest
  projected context state, and a position embedding". All K queries are built BEFORE
  any token is selected — no draft token feeds back into the next prediction (single
  parallel call).
- Backbone: **two causal Transformer layers**, hidden 512, 8 heads, FFN 1536
  (SiLU-gated), RMSNorm. Queries attend to the context and to earlier queries.
- Output: projected back to 2048 and ADDED to the latest early-loop state
  (their Eq. 3): h̃_{t,j} = h_t^(d1) + D_{φ,j}(H_t^(d1), E(a_t)); distribution
  q_{φ,j} = softmax(W_out h̃_{t,j}) through the FROZEN target output head.
- **Draft vocabulary** (their A.5): at inference the drafter's output projection is
  restricted to the 8,192 most frequent tokens of the teacher-generated training
  outputs (ties by token id) + prompt tokens + EOS + committed tokens, per request.
  Draft token selection = argmax over this restricted vocab. Native proposals and
  verification always use the full vocab.
- (Raven also has a +8.5M recursive residual head, their Eqs. 8–9 — Raven-only, not
  needed for Ouro.)

### 1.4 Drafter training (their §4.1 / A.2 / A.5)

Data: the frozen target's **greedy full-depth continuations** on 9,025 mixed prompts
(their Table 6; dev/confirm 782/782):

| source | train | dev | notes |
|---|---|---|---|
| GSM8K train | 2,000 | 128 | |
| CodeAlpaca, Python-filtered | 2,000 | 128 | non-Python examples excluded |
| MATH train (original split) | 2,000 | 128 | |
| BBH custom split | 2,025 | 270 | 75/10/10 per subtask |
| UltraChat train_sft | 1,000 | 128 | first user request only; their responses unused |
| total | 9,025 | 782 | |

Dedup: exact after NFKC/case/whitespace normalisation; near-dup at token 5-gram
Jaccard ≥ 0.9 with ≥ 8 shared 5-grams; BBH 5-grams shared by ≥ 20% of a subtask
ignored; eval questions and few-shot demos protected before splitting.

Objective (their Eq. 5): cross-entropy of q_{φ,j} against the teacher's next K_train = 7
tokens after each anchor, labels past the recorded continuation masked; target model,
embeddings and output head frozen. Optimizer (their A.5): from scratch, 3 epochs, 32
anchor positions sampled per request per epoch, microbatch 8 × grad-accum 8 = 64,
AdamW (0.9, 0.95), wd 0.01, clip 1.0, LR warmup 5% → 2e-4, cosine → 2e-5, seed 42.

### 1.5 Their measurement context (for honest comparison)

Their numbers: 4× NVIDIA A40 (48 GB), 1 request per GPU, bf16 weights/caches, fp32
accumulation in target linears; local PyTorch backend with CUDA graphs (PyTorch 2.6,
FlashAttention 2.8.3); timing excludes prefill/warmup/loading; throughput
ν = Σ N_out / Σ t_decode, first (prefill-produced) token excluded. Their AR baseline =
full-depth standard AR decoding, implementation unspecified. Greedy speedups over their
own AR (their Table 1), **Ouro-2.6B row: GSM8K 5.21×, MATH500 4.37×, HumanEval 4.72×,
MBPP 4.76×, BBH 5.12×** (Ouro-1.4B: 4.27/3.73/4.07/4.01/4.30). Their ablations
(Ouro-1.4B, 428 reqs, GSM8K): no-LAD-no-sched 2.88×, LAD-only 2.89×, LAD+sched
(no susp., no vocab) 3.86×, +suspension 4.13×, full 4.26× — so the tree scheduler,
suspension, AND drafter all carry weight; a partial reimplementation would be unfair.
Their eval: GSM8K 1,319 3-shot/512 max-new; MATH500 500 4-shot Minerva/2048; HumanEval
164 0-shot/1024; MBPP 500 0-shot/1024; BBH 540 (20×27) 3-shot/1024;
add_special_tokens=False.

---

## 2. What the drafter needs from the serving side (all already exposed or cheap)

| need | status in our engine (`decode/{,late_k}/engine.py`) |
|---|---|
| pass-1 normed states (h^(1), what the head reads) per position | `loop_pass(X, t=0, ...)` returns exactly this; cache last 64 per branch (10 MB at S=2560) |
| pass-1 readout for native proposals (head on state, NOT gate mixture) | head matmul on the t=0 output; the gate-zero trap (gate-zero safeguard) says head-on-state is the only safe reading anyway |
| full-depth readout for verification | the engine's gate-mixture accumulation (`exp`), made per-node |
| frozen embed / lm_head | loaded tensors `embed`, `lm_head` |
| per-depth KV with per-row page tables | `attn_paged` already runs FA3 varlen with one page table per sequence, page_size 1 — mixed-depth rows fit by unifying the T per-pass caches into one [T·S] page pool |
| CUDA graphs over a fixed-shape step | existing capture machinery; the LoopAhead step is a FIXED 16-row forward (pad inactive rows), so it captures the same way |

The one genuinely new mechanism vs. our lattice: rows at **different** loop depths in
one forward (our lattice is synchronous — all rows at the same pass). Because the stack
is shared across passes, mixed depths differ only in (a) which cache slice a row
reads/writes — solved by the unified page pool + per-row write slots, and (b) per-row
gate-mixture accumulation state — a per-node persistent buffer (32 × 49,152 fp32 ≈ 6 MB).
Both fit the existing static-buffer CUDA-graph pattern.

---

## 3. Build plan (Stage 2, after approval)

All new files under `<data-root> only. Wrapper
scripts for every job, distinct `-n` captions, `jobs.tsv` here, **no background GPU daemons on any
timing job**, weights staged to node-local /tmp, document counts + `meta.done` verified
in every output (job status is not a completion check; `--docs` filters by doc_id — use
positions and assert n).

1. **`data/` — teacher data generation** (1 job, base-image mars env, HF transformers,
   batched greedy on released Ouro-2.6B staged to /tmp). Build the 9,025+782 prompt mix
   (datasets verified reachable on the hub 2026-10-30: openai/gsm8k ✓cached,
   EleutherAI/hendrycks_math ✓, sahil2801/CodeAlpaca-20k ✓, lukaemon/bbh ✓cached,
   HuggingFaceH4/ultrachat_200k ✓), dedup per their A.2, protect our fair_* eval docs
   and few-shot demos, generate greedy continuations, write
   `data/teacher_{train,dev}.jsonl` + a `MANIFEST.json` with per-source counts and every
   substitution. Also derive the 8,192-token draft vocabulary here.
2. **`drafter/` — LAD model + training** (1 job). Pass-1 windows computed on the fly by
   teacher-forced single-pass forwards (cheap); train per §1.4; save
   `drafter/ckpt/lad_ouro26.pt` + training curve + dev top-1/top-j accuracy per
   position (their B.2 gives q_1 ≈ 64.3% → q_3 ≈ 36.1% on Ouro/GSM8K as a sanity
   anchor for the trained drafter's quality).
3. **`decoder/` — tree decoder on our engine substrate.** `loopahead.py` (tree,
   scheduler, suspension, verification; one sync per loop-step), reusing
   `engine.py`-style weights/ops/graphs with the unified page pool. **CPU test first**
   (`test_cpu_loopahead.py`, pattern of `late_k/test_cpu_latek.py`): fp64, tree decode
   vs. from-scratch full-depth AR — token-identical on ~20 short docs, plus a KV-equality
   check for committed nodes. Then a 1-GPU smoke (few docs, `--audit`-style
   teacher-forced full-depth check of every committed token).
4. **`eval/` — measurement** (Stage 3): the 6 × 120 matched fair_* docs
   (`eval/data/fair_*_prompts.jsonl`), released Ouro-2.6B, one job running
   in-process: engine-AR first, then LoopAhead-repro (same process, same GPU → shared
   denominator, as the protocol requires); separately an HF-transformers AR job
   (ref_hf.py's `ar_scores` path) on the same docs. Report BOTH ratios, labelled:
   repro / our-engine-AR (honest denominator) and repro / HF-AR (what their ratio looks
   like over an HF-grade baseline). Note our protocol's six tasks ≠ their five
   (we have IFEval and MMLU-Pro, no HumanEval); GSM8K/MATH500/MBPP/BBH overlap.

## 3.1 Cost estimate

| item | GPU-h | notes |
|---|---|---|
| teacher generation, 9,807 prompts × ≤512 tok, batched HF greedy | 4–8 | one 1-GPU job, shardable ×2 |
| drafter training (3 ep × 9,025 × 32 anchors; on-the-fly pass-1 feats) | 1–2 | tiny model; target forwards dominate |
| decoder smokes / debug | 3–5 | CPU test costs no GPU |
| measurement: engine-AR + repro, 6 × 120 docs | 3–5 | AR ≈ 63 tok/s dominates |
| measurement: HF AR, 6 × 120 docs | 8–14 | HF ≈ 8–15 tok/s; shard ×2–3 |
| **total** | **≈ 20–35** | |

Engineering: the tree scheduler is the heavy part (≈ 800–1,200 lines + CPU tests);
data/training/eval are routine. Realistic build time: the decoder is the long pole.

---

## 4. Underspecified points and OUR assumptions (most favourable to them)

Recorded here and in every output's `meta.assumptions`. "Favourable" = the choice a
careful author would plausibly have made, erring toward whatever helps their speed.

| # | underspecified | our choice | why favourable / impact |
|---|---|---|---|
| 1 | Drafter input combination ("combines" anchor emb, latest projected context state, position emb) | separate linear 2048→512 for context states; anchor embedding through its own 2048→512; query_j = proj(E(a_t)) + proj(c_last) + learned-PE_j (sum) | sum is the standard reading; param count then matches their stated ≈11.1M |
| 2 | Drafter position encoding over [64 ctx + K queries] | RoPE over the concatenated sequence + learned per-j query embeddings | modern default; nothing in the text contradicts |
| 3 | Output projection form | linear 512→2048, no bias, residual add per Eq. 3 | matches 11.1M budget |
| 4 | Pre/post-norm, dropout | pre-norm RMSNorm, no dropout | standard for small spec heads (EAGLE-style) |
| 5 | Teacher continuation length / stopping | max-new 512, stop at EOS | unstated; 512 matches their GSM8K eval cap; longer adds cost, not fidelity |
| 6 | Teacher label generation method (they don't say how) | OUR engine's exact full-depth AR (Lattice B=0, CUDA graphs, bf16 greedy, until-trimmed) — batched HF generate was tried first and REJECTED: the released modeling code ignores left padding (11/12 batched rows diverge from single-prompt greedy at token 1; job e6aff2c0ae31590a) | exact greedy labels from the same machinery the measurement uses — strictly cleaner than any padded-batch approximation |
| 7 | Early readout for Ouro: raw head on pass-r state vs. partial gate mixture | head on pass-r state (their Eq. 2 literally: W_out h^(r)); also the only safe reading (gate can be exactly 0) | favourable: the mixture reading can silently emit token 0 |
| 8 | Full-depth "readout" for verification on Ouro (plain Z^(R) head vs. Ouro's learned 4-pass gate mixture) | the gate mixture — it IS the released model's output function; anything else would not be lossless w.r.t. the real model | required for a lossless claim on Ouro; assume their backend did the same |
| 9 | Exact node-selection priority beyond "closer to root first" | stable order: distance to root, then native-before-LAD, then creation order | second-order; any stable rule |
| 10 | Precise semantics of d_guard protected region | nodes whose tree position is ≤ 2 tokens past the committed prefix are always eligible and never suspended | direct reading of §3.2 |
| 11 | LAD-call admission detail ("when retention capacity is free", min(K,f)) | call for the selected anchor whenever f ≥ 1 after native children are placed; retain first min(3, f) | their stated Ouro rule, holes filled permissively |
| 12 | BBH custom 75/10/10 split (seed/ordering unknown) | our own per-subtask split, seed 42, eval items + few-shot demos excluded first | unknowable; affects only training data composition |
| 13 | MATH train "original split" | EleutherAI/hendrycks_math train (7,500) | the original MATH train; MATH-500 test items excluded by dedup anyway |
| 14 | CodeAlpaca "Python-filtered" rule | keep examples whose instruction/output contain Python signals (def/import/print/python) and no other-language markers; sample 2,000 seed 42 | their filter unspecified; any reasonable filter yields similar prompts |
| 15 | Algorithm 1 / Figure 13 pseudocode bodies (truncated in the HTML) | reconstructed from the §3.2 prose + App. D operator list (AdvanceBatch, FillNativeChildren, SelectAnchor, Reroot, Spawn, GraftChain, …) | the operator list pins the control flow; bodies are forced by the stated rules |
| 16 | Sampling mode | NOT reproduced (greedy only) | our protocol is greedy batch-1; stated scope limit, not an assumption |
| 17 | Their AR baseline implementation | we report against BOTH our engine AR (strong) and HF AR (weak) | brackets whatever theirs was |
| 18 | Their hardware (A40) vs ours | measure on our GPU, report absolute tok/s + ratios; never mix their ratios with ours | protocol rule |
| 19 | Drafter attention over draft vocab at train time | train over FULL vocab (Eq. 5 is plain CE); restriction is inference-only | their A.5 describes restriction under "Low-overhead execution" |
| 20 | UltraChat "first user request only" prompt framing | the first user turn, formatted with the same chat template as our fair prompts (SmolLM-style `<|im_start|>`), consistent with teacher generation | their prompt template unstated |
| 21 | training anchor: pass-1 argmax vs the teacher's actual token (Eq. 5 writes y_t) | the teacher's actual token (ground-truth anchor) | matches Eq. 5's conditioning and the only case that matters at decode time (a wrong anchor's branch dies at verification); cleaner signal |
| 22 | early readout on Ouro, refined from #7: at depth t we read argmax of (partial gate mixture + remainder x current-pass logits) | one formula for proposals (t=1 it EQUALS the pass-1 head readout, their Eq. 2), feedback (t=2) and verification (t=4 it equals the true mixture) | never degenerate when a gate weight is exactly 0 (gate-zero safeguard); strictly more informed than head-on-state at t=2 |
| 23 | feedback conflicts use the parent's deepest available readout among depths >= d2 (4 if present, else 2) | pred[4] of a deep non-root parent prunes dead children too | favourable: strictly less wasted work; the paper's "deepest available feedback depth" wording |
| 24 | eval protection: items whose question near-dups one of OUR 720 fair eval docs are dropped from the pools by CONTAINMENT (>= 90% of the item's 5-grams inside an eval doc, or verbatim substring), not symmetric Jaccard | their "protected" mechanism is unspecified; plain Jaccard provably misses short BBH questions (caught and fixed 2026-10-30) |

Build-stage facts recorded 2026-10-30: the drafter architecture choices in #1-#4 give EXACTLY 11,051,008 parameters
(= the paper's stated count; learned absolute position embeddings over 64+7 slots, an input RMSNorm(2048), four
2048<->512 bias-free projections, two pre-norm SwiGLU blocks, final RMSNorm(512)). The CPU test
(`decoder/test_cpu_loopahead.py`, 90 cases, fp64+fp32, 3 tiny models x 5 prompts) passes: token-identical to
full-depth AR with no drafter, with an adversarial random drafter, and with an oracle drafter; committed-prefix KV at
all four passes and the pass-1 window states equal a from-scratch forward (fp64 max err ~1e-15); the oracle drafter
accelerates tokens/step 1.6-2.1x over the no-drafter arm; suspension fires. Teacher-generation batching uses HF eager
attention (HF sdpa fails on a non-contiguous mask with left padding - recorded, labels only).

**Known risks (not assumptions):**
- **Host-side scheduler overhead.** On A40 a loop-pass is ~3× slower than on our GPU, so
  Python bookkeeping hides more easily there. We budget one device sync per loop-step and
  vectorised tree ops; if the repro ends host-bound we will say so explicitly alongside
  the number (it would make our measurement of their method conservative, which must be
  stated, not silently accepted).
- **bf16 near-ties:** committed tokens are full-depth argmaxes by construction, but GEMM
  shape changes flip near-ties (our K=4 experience). The audit reports exact
  agreement-with-AR counts rather than claiming token-identity.

---

## 5. Measurement protocol (Stage 3, recap)

Released Ouro-2.6B. 120 matched docs × {gsm8k, ifeval, math500, mbpp, bbh, mmlupro}
(`fair_*_prompts.jsonl`), batch 1, greedy, bf16, one GPU, nothing else on it, isolated GPU.
In one process per task: engine-AR then LoopAhead-repro (shared denominator). Separate
job: HF-transformers AR on the same docs. Deliver per task: absolute tok/s for all
three, repro/engine-AR, repro/HF-AR, tokens-per-loop-step, accept-length stats, and the
full-depth audit (n_diff per doc). Verify n == 120 and `meta.done` in every file.

Context for reading the result (from their Table 1): on *their* backend/AR, Ouro-2.6B
GSM8K is 5.21×. If our repro over our engine-AR lands well below that and over HF-AR
well above it, the gap is the denominator, which is the point of measuring both.

---

## 6. Verdict

**A faithful reproduction is achievable here.** The load-bearing parts are pinned by the
paper: the drafter's size/inputs/outputs/loss (Eqs. 2–5, A.5), the full training recipe
(Table 6 + A.5, optimizer and schedule included), every scheduling hyperparameter for
Ouro (Table 7: d1=1, d2=2, K=3, B_ret=32, B_exe=16, d_guard=2, stride 1), the conflict /
suspension / protected-region rules, and the greedy verification rule (exact by
construction). The datasets are available. The serving substrate (per-pass KV, pass-1
readouts, per-row page tables, CUDA-graph step) already exists in our engine.

What is NOT pinned is listed exhaustively in §4 — mostly second-order architecture
choices inside an 11M-parameter drafter and tie-breaking details in the scheduler, plus
the truncated pseudocode. None of these is the kind of hole that makes a reproduction
dishonest; the honest labelling is "our reimplementation, assumptions as recorded".
The two caveats that must travel with any number we publish: (a) greedy-only scope,
(b) different GPU and different AR denominators than theirs — handled by reporting
absolute tok/s and both ratios.

Estimated cost: ≈ 20–35 GPU-hours plus the decoder engineering (the tree scheduler is
the long pole). No GPU was used for this stage.
