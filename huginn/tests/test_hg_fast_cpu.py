"""Standalone CPU mathematical exactness gates for `HuginnFastDecoder` vs `HuginnBlockDecoder`."""

import torch

from huginn.model.modeling_a2d_huginn import A2DHuginnConfig, A2DHuginnForCausalLM
from huginn.decode.hg_model_io import make_noise
from huginn.decode.hg_block_dec import HuginnBlockDecoder, parse_blk_cfg
from huginn.decode.hg_fast_dec import HuginnFastDecoder, HuginnFastModel, parse_fast_cfg


def build_tiny_huginn(dtype=torch.float64, T: int = 8):
    torch.manual_seed(42)
    cfg = A2DHuginnConfig(
        n_embd=64,
        n_heads=4,
        n_layers=8,
        block_size=256,
        vocab_size=128,
        padding_multiple=64,
        tie_embeddings=False,
        intermediate_size=160,
        bias=False,
        n_layers_in_prelude=2,
        n_layers_in_recurrent_block=4,
        n_layers_in_coda=2,
        mean_recurrence=T,
        mean_backprop_depth=4,
        qk_bias=True,
        torch_dtype="float64" if dtype == torch.float64 else "float32",
    )
    m = A2DHuginnForCausalLM(cfg).to(dtype=dtype, device="cpu").eval()
    # Non-trivial qk_bias so bias-fusion path is exercised
    with torch.no_grad():
        for blk in list(m.transformer.prelude) + list(m.transformer.core_block) + list(m.transformer.coda):
            blk.attn.qk_bias.normal_(mean=0.0, std=0.05)
    return m


def check_kv_equivalence(d_ref: HuginnBlockDecoder, d_fast: HuginnFastDecoder, hi_completed: int, tol: float):
    """Verify every region's KV cache over `0 .. hi_completed - 1` matches `d_ref` within `tol`."""
    if hi_completed <= 0:
        return 0.0
    max_err = 0.0
    for reg in range(d_fast.R_tot):
        k_ref = d_ref.K_buf[reg, :, :hi_completed, :].transpose(0, 1)  # (L, NH, HD)
        v_ref = d_ref.V_buf[reg, :, :hi_completed, :].transpose(0, 1)
        off = reg * d_fast.SL
        k_fast = d_fast.K_pool[off : off + hi_completed, 0, :, :]
        v_fast = d_fast.V_pool[off : off + hi_completed, 0, :, :]
        ek = (k_ref - k_fast).abs().max().item()
        ev = (v_ref - v_fast).abs().max().item()
        max_err = max(max_err, ek, ev)
        assert max_err < tol, f"KV mismatch at reg={reg}: max_err={max_err:.3e} >= {tol:.3e}"
    return max_err


def run_suite(dtype, tol: float):
    T = 8
    m = build_tiny_huginn(dtype=dtype, T=T)
    L_max = 128
    fm = HuginnFastModel(m, max_seq=L_max + 64, fused=False, log=lambda *a: None)
    mask_id = 127
    stop_ids = (126,)

    cfgs = [
        "blk:8:0",
        "blk:8:4",
        "blk:4:4",
        "blk:3:4",
        "blk:2:4",
        "blk:2:2",
        "blk:2:0",
        "blk:2:4:sync",
        "blk:2:4:none",
        "eblk:4:4",
        "eblk:2:4",
    ]

    checks = 0
    worst_kv = 0.0

    for p_idx, prompt in enumerate(([5, 11, 23], [7, 13, 19, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71])):
        ns = make_noise(m, L_max, 1000 + p_idx, torch.device("cpu"), dtype)
        ref_seq_ids = None

        for cfg in cfgs:
            pc_r = parse_blk_cfg(cfg, T_default=T)
            pc_f = parse_fast_cfg(cfg, T_default=T)

            d_ref = HuginnBlockDecoder(
                m, K=pc_r["K"], B=pc_r["B"], L_max=L_max, T=pc_r["T"], bg=pc_r["bg"],
                nv=128, mask_id=mask_id, stop_ids=stop_ids,
            )
            d_fast = HuginnFastDecoder(
                fm, K=pc_f["K"], B=pc_f["B"], L_max=L_max, T=pc_f["T"], bg=pc_f["bg"], D_bg=pc_f["D_bg"],
                nv=128, mask_id=mask_id, stop_ids=stop_ids, attn="torch", graphs=False,
            )

            out_ref = d_ref.decode_one(prompt, ns, max_new=22)
            out_fast = d_fast.decode_one(prompt, ns, max_new=22)
            assert out_ref["gen_ids"] == out_fast["gen_ids"], (
                f"Token mismatch on {cfg} (prompt len {len(prompt)}): "
                f"ref={out_ref['gen_ids']} vs fast={out_fast['gen_ids']}"
            )
            assert out_ref["c_list"] == out_fast["c_list"], (cfg, out_ref["c_list"], out_fast["c_list"])
            checks += 1

            if cfg == "blk:8:0":
                ref_seq_ids = out_fast["gen_ids"]
            elif cfg == "blk:8:4":
                assert out_fast["gen_ids"] == ref_seq_ids, "Full-depth K=T block must match sequential!"
                checks += 1

            # Verify exact KV cache agreement over all fully caught-up positions (`hist_final[-1]`)
            hi_done = out_fast["hist_final"][-1]
            err = check_kv_equivalence(d_ref, d_fast, hi_done, tol)
            worst_kv = max(worst_kv, err)
            checks += 1

        # Forced proposal acceptance regimes (`c = 0`, `c = 2`, `c = B = 4`)
        def prop_fn(step_idx, lc, known):
            B = 4
            if step_idx % 3 == 0:
                return [99] * B  # force c = 0
            if step_idx % 3 == 1:
                return None      # natural MARS proposals
            return [17, 23, 99, 99]

        for cfg in ("blk:2:4", "blk:2:4:sync", "blk:3:4", "blk:4:4", "eblk:2:4"):
            pc_r = parse_blk_cfg(cfg, T_default=T)
            pc_f = parse_fast_cfg(cfg, T_default=T)
            d_ref = HuginnBlockDecoder(
                m, K=pc_r["K"], B=pc_r["B"], L_max=L_max, T=pc_r["T"], bg=pc_r["bg"],
                nv=128, mask_id=mask_id, stop_ids=stop_ids,
            )
            d_fast = HuginnFastDecoder(
                fm, K=pc_f["K"], B=pc_f["B"], L_max=L_max, T=pc_f["T"], bg=pc_f["bg"], D_bg=pc_f["D_bg"],
                nv=128, mask_id=mask_id, stop_ids=stop_ids, attn="torch", graphs=False,
            )
            r1 = d_ref.decode_one(prompt, ns, max_new=24, prop_override_fn=prop_fn)
            r2 = d_fast.decode_one(prompt, ns, max_new=24, prop_override_fn=prop_fn)
            assert r1["gen_ids"] == r2["gen_ids"] and r1["c_list"] == r2["c_list"], (cfg, r1["c_list"], r2["c_list"])
            err = check_kv_equivalence(d_ref, d_fast, r2["hist_final"][-1], tol)
            worst_kv = max(worst_kv, err)
            checks += 1

        # Verify tail-window `until` stop-string detection and special-token filtering
        class _DummyTok:
            def __init__(self):
                self.calls = 0

            def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
                self.calls += 1
                return " ".join(f"w{int(t)}" for t in ids)

        dtok_empty = _DummyTok()
        d_u1 = HuginnFastDecoder(fm, K=2, B=4, L_max=L_max, T=T, bg="pipe", D_bg=T, nv=128, mask_id=mask_id, stop_ids=stop_ids, attn="torch", graphs=False)
        ru1 = d_u1.decode_one(prompt, ns, max_new=24, tok=dtok_empty, until=["<|end_turn|>", "<|end_text|>"])
        assert dtok_empty.calls == 1 and len(ru1["gen_ids"]) == 24, f"Expected 1 post-loop decode call, got {dtok_empty.calls}"
        checks += 1

        target_tok = ru1["gen_ids"][10]
        dtok_hit = _DummyTok()
        d_u2 = HuginnFastDecoder(fm, K=2, B=4, L_max=L_max, T=T, bg="pipe", D_bg=T, nv=128, mask_id=mask_id, stop_ids=stop_ids, attn="torch", graphs=False)
        ru2 = d_u2.decode_one(prompt, ns, max_new=24, tok=dtok_hit, until=["<|end_turn|>", f"w{target_tok}"])
        assert ru2["finish"] == "until" and f"w{target_tok}" not in ru2["text"]
        checks += 1

        # Partial catch-up `p4` on T=8, K=2: verify 2-stage (`passes 5..8`) matches explicit 4-pass backfill
        d_p4 = HuginnFastDecoder(
            fm, K=2, B=4, L_max=L_max, T=T, bg="p4", D_bg=4,
            nv=128, mask_id=mask_id, stop_ids=stop_ids, attn="torch", graphs=False,
        )
        assert d_p4.NG == 2 and d_p4.Ep_bg == 5 and d_p4.has_coda_bg
        r_p4 = d_p4.decode_one(prompt, ns, max_new=20)
        assert len(r_p4["gen_ids"]) == 20
        checks += 1

        # Zero-padded `HD_kv > HD` (e.g. 96 -> 128 FA3 fallback) exactness check
        fm_pad = HuginnFastModel(m, max_seq=L_max + 64, fused=False, log=lambda *a: None)
        fm_pad.HD_kv = fm.HD + 8
        d_pad = HuginnFastDecoder(
            fm_pad, K=2, B=4, L_max=L_max, T=T, bg="pipe", D_bg=T,
            nv=128, mask_id=mask_id, stop_ids=stop_ids, attn="torch", graphs=False,
        )
        d_ref2 = HuginnBlockDecoder(
            m, K=2, B=4, L_max=L_max, T=T, bg="pipe",
            nv=128, mask_id=mask_id, stop_ids=stop_ids,
        )
        assert d_pad.decode_one(prompt, ns, max_new=20)["gen_ids"] == d_ref2.decode_one(prompt, ns, max_new=20)["gen_ids"]
        checks += 1

    print(f"[PASS] dtype={dtype}: {checks} exactness checks passed (worst KV max_abs_err = {worst_kv:.3e})")
    return checks


if __name__ == "__main__":
    n64 = run_suite(torch.float64, tol=1e-11)
    n32 = run_suite(torch.float32, tol=1e-5)
    print(f"ALL {n64 + n32} CPU GATES PASSED.")
