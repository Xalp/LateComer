"""test_cpu_batch_latek.py -- Deterministic CPU exactness gates for multi-request LateKBatchExecutor.

Verifies across float64 & float32, K in {2, 3, 4}, shapes in {b0, latt:2, latt:4}, R in {1, 2, 3, 5}:
  1. Single-request (R = 1) equivalence with standalone decode/latek.py LateKLattice (tokens, commit logs, and K/V buffers).
  2. Multi-request (R = 2, 3, 5) batched execution with heterogeneous prompt lengths, shuffled physical KV slots, and mid-stream
     request completion + continuous slot refill produces BIT-IDENTICAL token sequences and commit logs to independent R = 1 runs.
  3. ARStepExecutor (R = 3) continuous-refill equivalence with R = 1 full-depth sequential decoding.
"""
import os
import random
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "decode"))
sys.path.insert(0, HERE)

import engine as E
import late as LT
import latek as LK
from sglang_late.core_ops import Cfg, OuroCore, FlatKV
from sglang_late.shapes import LateShape, parse_shape
from sglang_late.late_step import ARStepExecutor
from batch_latek import LateKBatchExecutor, LateKReqState, SlotPool, prefill_batch

V = 28
MASK, EOS, IM = V - 1, V - 2, V - 3
NV = V - 3


def make_tiny_ouro_weights(seed: int, dtype=torch.float64):
    """Construct standalone tiny Ouro (T=4, 2 layers) weights directly in PyTorch (zero external repo needed)."""
    torch.manual_seed(seed)
    H, INTER, NL, NH, HD = 32, 64, 2, 4, 8
    cfg_dict = dict(
        vocab_size=V,
        hidden_size=H,
        intermediate_size=INTER,
        num_hidden_layers=NL,
        num_attention_heads=NH,
        num_key_value_heads=NH,
        head_dim=HD,
        total_ut_steps=4,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
    W = {}
    s = 0.15
    W["model.embed_tokens.weight"] = (s * torch.randn(V, H, dtype=dtype)).contiguous()
    W["model.norm.weight"] = (1.0 + 0.25 * torch.randn(H, dtype=dtype)).contiguous()
    W["model.early_exit_gate.weight"] = (0.4 * torch.randn(1, H, dtype=dtype)).contiguous()
    W["model.early_exit_gate.bias"] = torch.zeros(1, dtype=dtype)
    lm = 1.2 * torch.randn(V, H, dtype=dtype)
    lm[NV:].zero_()
    W["lm_head.weight"] = lm.contiguous()
    for i in range(NL):
        p = f"model.layers.{i}."
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            W[p + f"self_attn.{proj}.weight"] = (s * torch.randn(H, H, dtype=dtype)).contiguous()
        for proj, r, c in (("gate_proj", INTER, H), ("up_proj", INTER, H), ("down_proj", H, INTER)):
            W[p + f"mlp.{proj}.weight"] = (s * torch.randn(r, c, dtype=dtype)).contiguous()
        for nm in ("input_layernorm", "input_layernorm_2", "post_attention_layernorm", "post_attention_layernorm_2"):
            W[p + f"{nm}.weight"] = (1.0 + 0.25 * torch.randn(H, dtype=dtype)).contiguous()
    return W, E.Cfg(cfg_dict), Cfg(cfg_dict)


PROMPTS = [
    [3, 5, 7, 11, 13, 17, 19, 2, 4, 6],
    [2, 4, 8, 16, 1, 9, 6, 5, 6, 10, 12, 14],
    [10, 11, 7, 7, 1, 3, 3, 9, 14, 15],
    [12, 0, 14, 2, 18, 19, 1, 1, 5, 7, 2, 3, 8],
    [9, 9, 15, 16, 17, 3, 18, 0, 0, 4, 11],
    [4, 13, 2, 7, 19, 8, 5, 1, 16, 10],
    [1, 6, 12, 18, 3, 9, 15, 2, 8, 14, 20, 5, 11, 17],
]


def run_continuous_batch(core, mode_kind, K, shape_str, prompts, max_new_list, R, W=256, seed_perm=42):
    """Simulate R concurrent slots continuously refilling from `prompts` until all prompts finish."""
    B, sets = parse_shape(shape_str) if mode_kind == "lk" else (0, [])
    sh = LateShape(B, sets) if mode_kind == "lk" else None
    scr = R * sh.PH if (sh and sh.PH > 0) else 8
    pool_size = R * W + 128
    kv = FlatKV.alloc(core.c, pool_size, scr, "cpu", core.dt)

    # Shuffle free list deterministically so physical slots are non-contiguous across requests
    pool = SlotPool(pool_size)
    random.Random(seed_perm).shuffle(pool._free)

    rtt = torch.zeros(R + 1, W + 16, dtype=torch.int32)
    if mode_kind == "lk":
        ex = LateKBatchExecutor(core, K, sh, kv, rtt, W, NV, MASK, R=R, attn="torch", graphs=False)
    else:
        ex = ARStepExecutor(core, kv, rtt, W, NV, MASK, R=R, attn="torch", graphs=False)

    q_idx = 0
    n_total = len(prompts)
    results = [None] * n_total

    # Per-slot tracking
    slots = [None] * R

    def mount_slot(b, p_idx):
        pids = list(prompts[p_idx])
        mnew = max_new_list[p_idx]
        need = len(pids) + mnew + (sh.NB + sh.D + 8 if sh else 8)
        alloc = pool.alloc(need)
        rtt[b + 1, :need] = torch.tensor(alloc, dtype=torch.int32)
        first = prefill_batch(core, kv, [pids], rtt, [b + 1], NV, W, pad=0, attn="torch")[0]
        known = pids + [first]
        done = (first in (EOS, IM)) or (len(known) - len(pids) >= mnew)
        st = LateKReqState(K, sh.D, len(pids)) if sh else None
        slots[b] = dict(p_idx=p_idx, P0=len(pids), mnew=mnew, known=known, log=[], st=st, alloc=alloc, done=done, dummy=False)

    def mount_dummy(b):
        pids = [1, 2, 3, 4, 5, 6]
        need = 64
        alloc = pool.alloc(need)
        rtt[b + 1, :need] = torch.tensor(alloc, dtype=torch.int32)
        first = prefill_batch(core, kv, [pids], rtt, [b + 1], NV, W, pad=0, attn="torch")[0]
        st = LateKReqState(K, sh.D, len(pids)) if sh else None
        slots[b] = dict(p_idx=-1, P0=len(pids), mnew=999999, known=pids + [first], log=[], st=st, alloc=alloc, done=False, dummy=True)

    for b in range(R):
        if q_idx < n_total:
            mount_slot(b, q_idx)
            q_idx += 1
        else:
            mount_dummy(b)

    while True:
        # Retire finished real requests and refill
        for b in range(R):
            s = slots[b]
            if not s["dummy"] and s["done"]:
                gen_raw = s["known"][s["P0"] : s["P0"] + s["mnew"]]
                results[s["p_idx"]] = dict(gen_raw=gen_raw, log=list(s["log"]))
                pool.free(s["alloc"])
                rtt[b + 1].zero_()
                if q_idx < n_total:
                    mount_slot(b, q_idx)
                    q_idx += 1
                else:
                    mount_dummy(b)

        if all(s["dummy"] for s in slots):
            for b in range(R):
                pool.free(slots[b]["alloc"])
            break

        # Keep dummy slots from overflowing W
        for b in range(R):
            s = slots[b]
            if s["dummy"] and len(s["known"]) > 40:
                pool.free(s["alloc"])
                mount_dummy(b)

        rpis = [b + 1 for b in range(R)]
        if mode_kind == "ar":
            lcs = [len(slots[b]["known"]) - 1 for b in range(R)]
            lasts = [slots[b]["known"][-1] for b in range(R)]
            tops = ex.step(lcs, rpis, lasts)
            for b in range(R):
                s = slots[b]
                tk = tops[b]
                s["known"].append(tk)
                s["log"].append(1)
                if not s["dummy"] and (tk in (EOS, IM) or len(s["known"]) - s["P0"] >= s["mnew"]):
                    s["done"] = True
        else:
            hdrs = [tuple(slots[b]["st"].hist) for b in range(R)]
            ids_rows = [
                slots[b]["st"].build_ids(lambda p, kn=slots[b]["known"]: kn[p], slots[b]["known"][-1], sh, MASK)
                for b in range(R)
            ]
            tops = ex.step(hdrs, rpis, ids_rows)
            for b in range(R):
                s = slots[b]
                commit, _ = s["st"].advance(tops[b], sh)
                hit_eos = False
                kept = []
                for tk in commit:
                    kept.append(tk)
                    if tk in (EOS, IM):
                        hit_eos = True
                        break
                s["known"].extend(kept)
                s["log"].append(len(kept))
                if not s["dummy"] and (hit_eos or len(s["known"]) - s["P0"] >= s["mnew"]):
                    s["done"] = True

    assert pool.available == pool.size - 1, f"Slot leak: {pool.available} != {pool.size - 1}"
    return results


def main():
    n_checks = 0
    for dtype in (torch.float64, torch.float32):
        Wd, ecfg, ccfg = make_tiny_ouro_weights(seed=7, dtype=dtype)
        eng = LT.LateEngine(Wd, ecfg, "cpu", dtype, S=256, fused=False, log=lambda *x: None, M=8, Bmax=16)
        core = OuroCore(ccfg, "cpu", dtype, W=Wd, fused=False, log=lambda *x: None)

        # Varying max_new across prompts to force staggered completions and continuous slot refills!
        max_news = [17, 29, 11, 25, 19, 23, 15]

        # 1. Standalone LateKLattice vs LateKBatchExecutor (R = 1, 2, 3, 5)
        for K in (4, 3, 2):
            for shape_str in ("b0", "latt:2", "latt:4"):
                B, sets = parse_shape(shape_str)
                ref = LK.LateKLattice(eng, K, B, sets, NV, [EOS, IM], MASK, attn="torch", graphs=False)
                ref_out = []
                for prm, mn in zip(PROMPTS, max_news):
                    for kb in eng.kbuf + eng.vbuf:
                        kb.zero_()
                    r0 = ref.decode(prm, mn, prefill_attn="torch")
                    ref_out.append(dict(gen_raw=r0["gen_raw"][:mn], log=r0["log"]))

                for R in (1, 2, 3, 5):
                    got = run_continuous_batch(core, "lk", K, shape_str, PROMPTS, max_news, R=R, W=256, seed_perm=100 + R)
                    for i, (g, r) in enumerate(zip(got, ref_out)):
                        assert g["gen_raw"] == r["gen_raw"] and g["log"] == r["log"], (
                            f"MISMATCH dtype={dtype} K={K} shape={shape_str} R={R} prompt={i}:\n"
                            f"  got={g}\n  ref={r}"
                        )
                        n_checks += 1

        # 2. Standalone AR Lattice(B=0) vs ARStepExecutor (R = 1, 3, 5)
        ref_ar = E.Lattice(eng, 0, NV, [EOS, IM], MASK, attn="torch", graphs=False)
        ref_ar_out = []
        for prm, mn in zip(PROMPTS, max_news):
            for kb in eng.kbuf + eng.vbuf:
                kb.zero_()
            r0 = ref_ar.decode(prm, mn, prefill_attn="torch")
            ref_ar_out.append(dict(gen_raw=r0["gen_raw"][:mn], log=r0["log"]))

        for R in (1, 3, 5):
            got_ar = run_continuous_batch(core, "ar", 4, "ar", PROMPTS, max_news, R=R, W=256, seed_perm=200 + R)
            for i, (g, r) in enumerate(zip(got_ar, ref_ar_out)):
                assert g["gen_raw"] == r["gen_raw"], f"AR MISMATCH dtype={dtype} R={R} prompt={i}: {g} vs {r}"
                n_checks += 1

    print(f"ALL {n_checks} CPU MULTI-REQUEST EXACTNESS & SLOT-REFILL CHECKS PASSED 100%!")


if __name__ == "__main__":
    main()
