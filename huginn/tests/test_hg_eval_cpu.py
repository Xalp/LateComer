"""CPU exactness unit tests for `arch_explore.huginn_eval` (`eval_hg_single.py` + `hg_block_dec.py`)."""

import torch

from huginn.model.modeling_a2d_huginn import A2DHuginnConfig, A2DHuginnForCausalLM
from huginn.decode.hg_model_io import make_noise
from huginn.decode.eval_hg_single import SftHuginnDecoder, TeacherForcedProbe
from huginn.decode.hg_block_dec import HuginnBlockDecoder, parse_blk_cfg


def build_tiny_model(T: int = 8, nv: int = 96):
    torch.manual_seed(42)
    cfg = A2DHuginnConfig(
        n_embd=64,
        n_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        intermediate_size=128,
        n_layers_in_prelude=1,
        n_layers_in_recurrent_block=2,
        n_layers_in_coda=1,
        n_layers=4,
        mean_recurrence=T,
        mean_backprop_depth=4,
        vocab_size=nv,
        padded_vocab_size=nv,
        block_size=256,
        qk_bias=True,
    )
    cfg._attn_implementation = "sdpa"
    m = A2DHuginnForCausalLM(cfg).float().eval()
    # Resize embeddings to add <|mask|> = nv and pad to 128 (mimicking Stage-2 65536 -> 65600)
    m.resize_token_embeddings(nv + 1, pad_to_multiple_of=32)
    assert m.config.padded_vocab_size == 128
    # Boost weights above nv so any bug failing to slice `:nv` immediately emits >= nv!
    with torch.no_grad():
        m.lm_head.weight[nv:] += 50.0
    return m


def test_all():
    T = 8
    nv = 96
    mask_id = 96
    m = build_tiny_model(T=T, nv=nv)
    dev = torch.device("cpu")
    dt = torch.float32

    prompt = [3, 14, 15, 9, 26, 53, 58, 7, 11, 42]
    L_max = 96
    ns = make_noise(m, L_max, seed=12345, device=dev, dtype=dt)

    # ---- Test 1: SftHuginnDecoder r:8 vs fresh causal TeacherForcedProbe & never emits >= nv ----
    sdec = SftHuginnDecoder(m, f"r:{T}", B=1, L=L_max, nv=nv)
    sdec.keep_logits = True
    first_tok, lg0 = sdec.prefill(0, prompt, ns)
    assert 0 <= first_tok < nv, first_tok
    gen_single = [first_tok]
    active = torch.tensor([True])
    for _ in range(11):
        out = sdec.step(torch.tensor([gen_single[-1]]), active)
        nt = int(out["next"][0].item())
        assert 0 <= nt < nv, nt
        gen_single.append(nt)

    probe = TeacherForcedProbe(m, nv=nv)
    full_seq = prompt + gen_single[:-1]
    ctx = probe.build_clean_context(full_seq, ns, D=T, record_states=True)
    max_lg_diff = float((ctx["logits"][len(prompt) - 1] - lg0).abs().max().item())
    pred_from_ctx = ctx["logits"][len(prompt) - 1 :].argmax(-1).tolist()
    assert pred_from_ctx == gen_single, (pred_from_ctx, gen_single)
    print(f"[eval-cpu test 1] SftHuginnDecoder r:{T} exact match (12 tokens, prefill logit diff={max_lg_diff:.2e})")

    # Probe exactness check
    S0 = len(prompt)
    lg_ora = probe.eval_trajectory(ctx, S0, ctx["states"][5][S0:], [6, 7, 8])
    ora_diff = float((lg_ora - ctx["logits"][S0:]).abs().max().item())
    assert ora_diff < 1e-5, ora_diff
    print(f"[eval-cpu test 1b] TeacherForcedProbe oralate({T},3) max diff = {ora_diff:.2e}")

    # ---- Test 2: HuginnBlockDecoder at full depth K=T, B=0 and B=4 matches r:T bit-for-bit ----
    bdec0 = HuginnBlockDecoder(m, K=T, B=0, L_max=L_max, T=T, nv=nv, mask_id=mask_id, stop_ids=())
    r0 = bdec0.decode_one(prompt, ns, max_new=12)
    assert r0["gen_ids"] == gen_single, (r0["gen_ids"], gen_single)

    bdec4 = HuginnBlockDecoder(m, K=T, B=4, L_max=L_max, T=T, nv=nv, mask_id=mask_id, stop_ids=())
    r4_nat = bdec4.decode_one(prompt, ns, max_new=12)
    assert r4_nat["gen_ids"] == gen_single, (r4_nat["gen_ids"], gen_single)

    # Force partial/full oracle proposals (c = 3, 0, 4, 1, ...) to exercise multi-token commits at K=T!
    oracle_full = prompt + gen_single
    def prop_fn(step_idx, lc, known):
        rem = oracle_full[lc + 1 : lc + 5]
        if len(rem) < 4:
            rem = rem + [0] * (4 - len(rem))
        if step_idx == 0:
            # 3 valid proposals + 1 wrong
            return [rem[0], rem[1], rem[2], (rem[3] + 1) % nv]
        if step_idx == 1:
            # all 4 valid proposals!
            return rem
        return [(rem[0] + 1) % nv, rem[1], rem[2], rem[3]]  # c=0

    r4_ora = bdec4.decode_one(prompt, ns, max_new=12, prop_override_fn=prop_fn)
    assert r4_ora["gen_ids"] == gen_single, (r4_ora["gen_ids"], gen_single)
    assert r4_ora["c_list"][0] == 3 and r4_ora["c_list"][1] == 4, r4_ora["c_list"]
    print(f"[eval-cpu test 2] HuginnBlockDecoder K={T}, B=4 lossless exact match (c_list={r4_ora['c_list']})")

    # ---- Test 3: Pipelined KV catch-up (`bg=pipe`) & `bg=sync` exact KV verification for K=3 (non-divisor) and K=4 ----
    for K_test in (3, 4):
        for bg_mode in ("pipe", "sync"):
            bdec = HuginnBlockDecoder(m, K=K_test, B=3, L_max=L_max, T=T, bg=bg_mode, nv=nv, mask_id=mask_id, stop_ids=())
            def mixed_props(step_idx, lc, known):
                # Deterministic varying acceptance lengths
                return [(lc + step_idx + j) % nv for j in range(3)]
            res = bdec.decode_one(prompt, ns, max_new=16, prop_override_fn=mixed_props)
            full_committed = prompt + res["gen_ids"]
            # Up to `p_exact`, background catch-up has finished all T Core steps + Coda!
            if bg_mode == "sync":
                p_exact = res["lc_final"]
            else:
                # Group NG finished step T on the previous step, so `hist_final[-1]` has completed both Core T and Coda!
                p_exact = res["hist_final"][-1]
            assert p_exact > len(prompt) + 4, (K_test, bg_mode, p_exact, res["hist_final"])

            ctx_ref = probe.build_clean_context(full_committed[:p_exact], ns, D=T, record_states=False)
            # ctx_ref['Kc'] is (S_slots, p_exact, H, hd); bdec.K_buf is (S_slots, H, L_tot, hd)
            K_got = bdec.K_buf[:, :, :p_exact, :].permute(0, 2, 1, 3)
            V_got = bdec.V_buf[:, :, :p_exact, :].permute(0, 2, 1, 3)
            k_err = float((K_got - ctx_ref["Kc"]).abs().max().item())
            v_err = float((V_got - ctx_ref["Vc"]).abs().max().item())
            assert k_err < 1e-5 and v_err < 1e-5, (K_test, bg_mode, k_err, v_err)
            print(
                f"[eval-cpu test 3] K={K_test} (T={T}), bg={bg_mode:4s}: exact KV verified across all "
                f"{bdec.S_slots} slots x {p_exact} tokens (max K err={k_err:.2e}, V err={v_err:.2e})"
            )

    # ---- Test 4: Early-Exit block mode `eblk:<K>:<B>` matches single-token `r:K` bit-for-bit ----
    for K_ee in (3, 4):
        pc_ee = parse_blk_cfg(f"eblk:{K_ee}:4", T_default=T)
        assert pc_ee["kind"] == "eblk" and pc_ee["T"] == K_ee and pc_ee["K"] == K_ee and pc_ee["bg"] == "none"
        sdec_ee = SftHuginnDecoder(m, f"r:{K_ee}", B=1, L=L_max, nv=nv)
        f_ee, _ = sdec_ee.prefill(0, prompt, ns)
        gen_ee_ref = [f_ee]
        for _ in range(11):
            o_ee = sdec_ee.step(torch.tensor([gen_ee_ref[-1]]), torch.tensor([True]))
            gen_ee_ref.append(int(o_ee["next"][0].item()))

        ebdec = HuginnBlockDecoder(m, K=pc_ee["K"], B=pc_ee["B"], L_max=L_max, T=pc_ee["T"], bg=pc_ee["bg"], nv=nv, mask_id=mask_id, stop_ids=())
        r_ee = ebdec.decode_one(prompt, ns, max_new=12)
        assert r_ee["gen_ids"] == gen_ee_ref, (K_ee, r_ee["gen_ids"], gen_ee_ref)
        print(f"[eval-cpu test 4] Early-Exit eblk:{K_ee}:4 exact match with r:{K_ee} (12 tokens, 0 mismatches)")

    # ---- Test 5: Dynamic RoPE extension when L_tot > m.freqs_cis.shape[1] (regression test for MBPP doc 482) ----
    orig_fc = m.freqs_cis
    m.freqs_cis = orig_fc[:, :14]  # artificially shrink buffer below prompt(10) + max_new(12)
    try:
        bdec_ext = HuginnBlockDecoder(m, K=T, B=4, L_max=L_max, T=T, nv=nv, mask_id=mask_id, stop_ids=())
        r_ext = bdec_ext.decode_one(prompt, ns, max_new=12)
        assert r_ext["gen_ids"] == gen_single, (r_ext["gen_ids"], gen_single)
        print(f"[eval-cpu test 5] Dynamic freqs_cis extension (14 -> {bdec_ext.freqs.shape[1]}) exact match!")
    finally:
        m.freqs_cis = orig_fc


if __name__ == "__main__":
    test_all()
