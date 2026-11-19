# `generate_mars` — Phase-2 HF rollout generator (Option-4 agentic RL)

A correctness-first, single-sequence MARS generator for the agentic-RL rollout
loop. It returns, per call, everything the GRPO trainer needs for the Option-4
loss mask.

```python
from mars.samplers import generate_mars

token_ids, accept_log = generate_mars(
    model, tokenizer, prefix_ids,
    tau=0.9, B=4, max_new_tokens=512, temperature=1.0,
)
# or, for the loss mask:
out = generate_mars(model, tokenizer, prefix_ids, tau=0.9, B=4, return_dict=True)
out.token_ids        # [N] generated ids (no prefix)
out.accept_log       # [N_1, N_2, ...] tokens accepted per forward pass
out.is_window_start  # [N] bool; True at the first token of each window  ← Option-4 loss mask
out.num_forwards     # == len(accept_log)
```

## Why only `is_window_start` is on-policy
With `right_shift_logits=True` (the convention these Qwen3 MARS checkpoints are
trained with), the logit used for **window position 0 is the raw AR logit of the
last committed token** — conditioned on zero mask tokens — so the window-start
token is sampled from *exactly* the clean-AR distribution. Recomputing its
logprob at train time with a plain causal AR forward gives the same distribution
→ on-policy, no IS correction. Positions `j>0` see `j` masks and come from a
different distribution, so they are excluded (`is_window_start[j]=False`).

At `tau=1.0` (or `tau=None`) every forward accepts exactly one token → every
token is a window start → identical to standard AR greedy decode. This is the
primary correctness invariant and reduces Option-4 exactly to baseline GRPO.

## Control-flow tests (run now, no weights/transformers needed)
```bash
python tests/test_mars_generate.py
```
Verifies, with a position-deterministic mock oracle: `tau=1.0 ≡ AR`,
`accept_log` all-1s at `tau=1.0`, multi-token acceptance at `tau<0.95`,
`is_window_start` boundary alignment, partial-window acceptance, and EOS stop.

## Validating against a real `MARS-DemyAgent-4B` checkpoint (Phase 1 output)
Once a MARS checkpoint exists, add an integration check:
1. Load the MARS model via `dllm.utils.models.get_model` (adds the `<|mask|>`
   token + Qwen3 chat template).
2. `generate_mars(..., tau=1.0, temperature=0.0)` on a held-out prompt set must
   be **token-identical** to vanilla `model.generate(do_sample=False)` from the
   same weights (strict-superset / lossless-at-τ=1 property).
3. `generate_mars(..., tau=0.9)` should show mean `tokens_per_forward > 1` on
   math/code think-blocks.

## Not yet implemented (deliberately — later phases / out of scope for v1)
- **No KV cache**: one full forward per acceptance step. A block-cached variant
  exists in `mars_cached_sampler.py`; wiring it here is a perf-only follow-up.
- **No batching**: single prefix. The RL loop calls this per assistant turn.
- **No vLLM/SGLang**: HF only. This sets the v1 rollout speed ceiling; vLLM
  integration is the documented next optimization.
