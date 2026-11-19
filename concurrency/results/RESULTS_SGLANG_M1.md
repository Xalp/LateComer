# sglang_late M1: the two-pass late lattice in SGLang 0.5.14, batch 1

2026-10-28. Ouro-2.6B v2 (`ouro26_1p1_ttt/mars_blk16`), bf16, greedy, batch 1, at most 1024 new tokens, lm-eval stop strings,
H100 80GB, CUDA 13 container (sglang 0.5.14, torch 2.11 cu130), isolated GPU, weights staged to /tmp. All GPU work ran in one dev-worker job
(`faa04cf68cb8cf5b`, `late-sgl-m1-dev`; command scripts and logs in `jobs/devq/`). Launch instructions: `../README.md`.

## Verdict

- **(a) Equal to late.py, fp32:** CPU fp32 and fp64 on tiny random Ouro: 660/660 decodes token- and log-identical, every physical K/V entry
  identical (274,080 (layer, loop, position) checks), with identity and shuffled slot maps. GPU fp32 on the real v2: 16/16 identical
  (B=0, 4, 8 and AR, 4 GSM8K prompts x 64 tokens).
- **(b) Equal to late.py, bf16, through the server:** **100/100 GSM8K docs 0-99 token-identical** for B=4, B=8 and AR, so there are no
  near-ties to document. tokens/step equal (3.497 / 4.103 / 1.000). Full sets through the server (late2_full ran on other nodes): **GSM8K 1319/1319** token-identical
  for B=4 and B=8; **IFEval B=4 541/541, B=8 540/541**. The one IFEval doc (369, B=8) is identical up to late.py's stop point but kept the
  tokens after an EOS: its last commit contained the EOS and crossed max_new_tokens = 1024, and SGLang's `update_finish_state` checks the
  length limit before stop tokens on a multi-token run, so it finished by "length" with the EOS and 7 later tokens in the output. Fixed
  (the worker now cuts a commit after the first token SGLang itself stops on); regression run `jobs/devq/012_eosfix.sh` (job 27099a509b29fded): IFEval docs 360-379 B=8 20/20 identical, doc 369 now ends at its EOS (1016 tokens + EOS, finish = matched token 2, text identical to late.py); GSM8K 0-99 B=4 100/100 at K=1 (344.5 tok/s) and K=4 (363.1 tok/s); requests with `ignore_eos` (exactly 256 tokens, prefix = the normal output) and without `stop_token_ids` (SGLang stops on 0) run at K=1 and K=4 without desync (`tests/edge_eos.py`).
- **(c) Decode speed within 10% of late.py, same job, same GPU:** B=4 **345.0 vs 372.4 tok/s = 0.926x**, B=8 385.4 vs 412.1 = 0.935x,
  AR 60.0 vs 63.8 = 0.941x. With `--late-steps-per-call 4`: **0.967x / 0.975x / 0.971x** (360.2 / 401.9 / 61.9 tok/s), still identical.
  The whole gap is SGLang's per-iteration host time (table below).

## Equivalence

| test | where | result |
|---|---|---|
| `tests/test_cpu_core.py` (fp32 + fp64, 3 tiny models x 5 prompts x 40 tokens; ar, b0, latt:2/4/8, sp:5; lkv + old; identity + shuffled slots) | dev box CPU | ALL OK: 660 decodes / 13,356 steps identical (tokens + per-step commit log); 274,080 K/V entries bit-equal to late.py's `kbuf`; 2-token prompt (< B+1) runs finite (`test_cpu_core.json`) |
| `tests/equiv_inproc.py` fp32 (torch attention: FA3 has no fp32 path) | GPU, in process | 16/16 identical (`equiv_m1.json`) |
| `tests/equiv_inproc.py` bf16 (paged FA3, graphs, pad-64 prefill, W = 2560), GSM8K 0-99 | GPU, in process | B=4 100/100, B=8 100/100, AR 0-19 20/20 identical; graph == eager 20/20; decode tok/s 0.995x / 0.996x / 0.995x of late.py |
| server vs same-job late.py (`run_late.py`), GSM8K 0-99 | GPU, HTTP | B=4 / B=8 / AR 100/100 identical (`cmp_m1k1_*.json`; also `m1t3`, `m1k4` runs) |
| server vs late2_full (4 shards, other nodes), GSM8K 1319 | GPU, HTTP | B=4 1319/1319, B=8 1319/1319 (`cmp_t5gsm_*.json`) |
| server vs late2_full, IFEval 541 | GPU, HTTP | B=4 541/541, B=8 540/541 (doc 369: EOS kept inside a run that crossed max_new, fixed since) (`cmp_t5ife_*.json`) |

Accuracy follows from identity: GSM8K 0-99 flex .850 (B=4) / .830 (B=8) / .860 (AR), same as late.py on the same docs (`score_m1t3.json`).
Full GSM8K 1319: B=4 **.846**, B=8 **.839** (late2_full: same; AR .873, sp13 .867) (`score_t5gsm.json`). IFEval 541 prompt-strict: server B=4 .695 / B=8 .697 vs late2_full .697 / .697 (AR .730). The texts are byte-identical
except doc 369 (B=8); docs 19 and 22 have identical texts but score differently between the two files, i.e. IFEval scorer
nondeterminism (langdetect), not a decoder difference (`score_t5ife.json`).

## Speed (GSM8K docs 0-99, same job, late.py first)

Decode tok/s = (tokens to the stop point - 1) / (time from the end of the prefill to the end of the last decode step), the server measured
inside the worker (`--late-stats`), late.py as in late2 (`decode_tok_per_s`).

| decoder | late.py | server K=1 | ratio | server K=4 | ratio |
|---|---|---|---|---|---|
| B=4 `l2:lkv:am:latt:4` | 372.4 | 345.0 | 0.926 | 360.2 | 0.967 |
| B=8 `l2:lkv:am:latt:8` | 412.1 | 385.4 | 0.935 | 401.9 | 0.975 |
| AR | 63.8 | 60.0 | 0.941 | 61.9 | 0.971 |

Full GSM8K 1319 through the server (K=1): B=4 349.1 tok/s, B=8 399.8 tok/s (late2_full on other nodes: 377.7 / 424.6 -> 0.924x / 0.942x).
IFEval 541 through the server (K=1): B=4 228.7, B=8 225.1 tok/s (late2_full 244.8 / 237.4 -> 0.934x / 0.948x; tok/step 2.33 / 2.44).

**Where the time goes (ms per lattice step, mean over all steps of docs 0-99):**

| | late.py step | server step | = worker (executor + host verify) | + SGLang scheduler gap | + prefill -> first decode |
|---|---|---|---|---|---|
| B=4, K=1 | 9.294 | 10.032 | 9.397 | 0.587 | 0.048 |
| B=8, K=1 | 9.852 | 10.537 | 9.900 | 0.575 | 0.062 |
| AR, K=1 | 15.516 | 16.486 | 15.739 | 0.730 | 0.017 |
| B=4, K=4 | 9.294 | 9.609 | 9.334 | 0.213 | 0.053 |
| B=8, K=4 | 9.852 | 10.103 | 9.841 | 0.195 | 0.058 |

- The executor itself is late.py's speed (in-process 0.995x; worker GPU part 9.31 ms = late.py's 9.27 ms GPU part).
- The worker adds ~0.1 ms of host work per step (ids, verification, the packed H2D of the result, result objects).
- **SGLang's per-iteration host time is ~0.6 ms** (0.73 ms for AR): result processing, finish checks, the next batch, `prepare_for_decode`
  (KV reserve kernel), the forward-isolation snapshot. That is 6-7% of a 9.3 ms step. `--late-steps-per-call 4` runs 4 lattice steps per
  scheduler iteration and cuts it to ~0.2 ms per step; overlap scheduling (design M4) would hide it entirely.
- The server's prefill is eager (~65 ms per GSM8K prompt vs late.py's 18.6 ms graph prefill). Prefill is outside the decode metric; it costs
  end-to-end throughput (wall tok/s B=4: 279 server vs 351 late.py).

## Not done in M1

T4 edge-case suite (stop string across a commit, abort, streaming), prefill CUDA graphs, non-greedy abort, batch > 1 (M2), radix cache (M3).

## Review re-run (2026-10-28, job ed68fc8b4cdb0039 `late-sgl-m1-review`, `jobs/devq_review/013_review.sh`)

Post-review code (three fixes below), strict idle leak check on (`SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1`, also the 0.5.14 default).
- CPU `tests/test_cpu_core.py`: full run on the pre-fix code ALL OK (660 decodes, 274,080 K/V entries); `--quick 1` on the fixed code ALL OK.
- `equiv_inproc` (fixed code): fp32 16/16; bf16 B=4 100/100, B=8 100/100, AR 20/20; graph == eager 20/20; executor 0.997x / 1.000x / 0.994x of late.py.
- Server vs same-job late.py, GSM8K 0-99: B=4 / B=8 / AR at K=1 and B=4 / B=8 at K=4 all 100/100 (`cmp_rev_*`, `cmp_revk4_*`), no leak assert.
  Decode tok/s (late.py 368.5 / 406.9 / 63.1): K=1 336.5 (0.913x) / 377.6 (0.928x) / 59.5 (0.942x); K=4 354.7 (0.962x) / 390.8 (0.960x).
  1-2 points below the first M1 job (run-to-run noise of the same configuration).
- The "SGLang scheduler gap" includes our own `--late-stats` writer: B=4 K=1 without stats has client wall 274.9 vs 265.2 tok/s with stats
  (same tokens, 100/100), i.e. up to ~0.47 ms of the ~0.7 ms per-step gap may be the stats writer (bounded by run noise). Stats-on ratios are
  therefore conservative; the stats-off decode ratio could not be measured (no worker timestamps without stats).
- `edge_eos` K=1 / K=4 OK.
- **Fixed: context-limit crash.** SGLang admits P0 + max_new up to context_len - 1, but the executor asserts lc + D + TM < W and, at stride
  > ~8, prepare_for_decode reserves req_to_token[rpi, :lc + 2*stride] past the row (context_len + 4 + stride wide). Pre-fix snapshot
  (`jobs/prefix_snapshot`): an ignore_eos request run to the limit killed the server (`AssertionError: (2548, 2543, 2538, 2560)` at B=4 K=1,
  `(2538, 2529, 2520, 2560)` at B=8 K=4). Fix: `LateWorker._clamp_len` lowers max_new_tokens at prefill to
  min(W - D - TM - P0 + 1, row - 2*stride - P0 + 2) (logged). Post-fix (`tests/edge_limit.py`): B=4 K=1 ends at P0 + n = 2547, B=8 K=4 at 2530,
  AR at 2558 (SGLang's own cap), finish = length, server alive and the next request unchanged.
- **Fixed:** the prefill readout fell back silently to token 0 (an EOS) if the logits were missing; it now raises.
- **Fixed (latent):** the RoPE table is sized for the req_to_token width before capture and frozen afterwards (`Ops.frozen`), so it can never be
  reallocated under the captured graphs.
