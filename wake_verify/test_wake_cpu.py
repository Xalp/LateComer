"""Adversarial CPU mathematical exactness suite for Wake Verification (`ouro_wake.py` & `hg_wake.py`).

Runs on pure `torch` (zero `transformers` dependency) in exact `float64` arithmetic across
untrained random models (where late entry disagrees on >50% of tokens, triggering dozens of
multi-block rollbacks and premature EOS traps per sequence):
  1. `tau = 0.0` Wake Verification outputs the EXACT same token sequence bit-for-bit as sequential
     full-depth `T`-pass greedy decoding across all `K < T`, `B in {0, 2, 4}`, `T in {4, 6, 8, 12}`,
     and `suffix in {False, True}`.
  2. `tau = inf` (`suffix = False`) matches existing `LateKLattice` / `HuginnFastDecoder` bit-for-bit.
"""

import math
import os
import sys
import types
import torch

torch.set_num_threads(int(os.environ.get("NT", "2")))

# Provide lightweight stub if `transformers` is not installed in bare CPU container
if "transformers" not in sys.modules:
    try:
        import transformers  # noqa: F401
    except ImportError:
        class _DummyBase:
            @classmethod
            def register_for_auto_class(cls, *a, **kw):
                pass
            @classmethod
            def register(cls, *a, **kw):
                pass
        class _DummyMod(types.ModuleType):
            def __init__(self, name):
                super().__init__(name)
                self.__path__ = []
            def __getattr__(self, name):
                if name.startswith("__"):
                    raise AttributeError(name)
                return _DummyBase
        for mname in ("transformers", "transformers.modeling_outputs", "transformers.cache_utils",
                      "transformers.utils", "transformers.generation", "transformers.generation.utils"):
            sys.modules[mname] = _DummyMod(mname)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEK_DIR = os.path.join(REPO_ROOT, "decode")
for p in (LATEK_DIR, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import engine as E
import late as L2
import latek as LK
from huginn.decode.hg_fast_dec import (
    HuginnFastDecoder,
    HuginnFastModel,
    HuginnFastOps,
    _DirectCfgStub,
    _DirectModelStub,
)
from huginn.decode.hg_model_io import make_noise
from wake_verify.ouro_wake import WakeKLattice
from wake_verify.hg_wake import HuginnWakeDecoder


def make_random_ouro(seed: int, V: int, H: int = 32, INTER: int = 64, NL: int = 2, NH: int = 4,
                     gate_scale: float = 4.0, eos_id: int = 22, mask_id: int = 23, eos_bias: bool = False):
    torch.manual_seed(seed)
    dt = torch.float64
    s = H ** -0.5
    W = {
        "model.embed_tokens.weight": torch.randn(V, H, dtype=dt) * 0.25,
        "model.norm.weight": 1.0 + 0.2 * torch.randn(H, dtype=dt),
        "model.early_exit_gate.weight": torch.randn(1, H, dtype=dt) * (gate_scale * s),
        "model.early_exit_gate.bias": torch.zeros(1, dtype=dt),
        "lm_head.weight": torch.randn(V, H, dtype=dt) * (5.0 * s),
    }
    W["lm_head.weight"][mask_id].zero_()
    if eos_bias:
        W["lm_head.weight"][eos_id].mul_(1.2)
    for i in range(NL):
        p = f"model.layers.{i}."
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            W[p + f"self_attn.{proj}.weight"] = torch.randn(H, H, dtype=dt) * s
        W[p + "mlp.gate_proj.weight"] = torch.randn(INTER, H, dtype=dt) * s
        W[p + "mlp.up_proj.weight"] = torch.randn(INTER, H, dtype=dt) * s
        W[p + "mlp.down_proj.weight"] = torch.randn(H, INTER, dtype=dt) * (INTER ** -0.5)
        for nm in ("input_layernorm", "input_layernorm_2", "post_attention_layernorm", "post_attention_layernorm_2"):
            W[p + f"{nm}.weight"] = 1.0 + 0.2 * torch.randn(H, dtype=dt)
    cfg = E.Cfg({
        "hidden_size": H, "num_attention_heads": NH, "num_key_value_heads": NH,
        "intermediate_size": INTER, "num_hidden_layers": NL, "vocab_size": V,
        "total_ut_steps": 4, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
    })
    return W, cfg


def make_random_huginn_fast(seed: int, T: int, V: int = 64, E_dim: int = 48, NH: int = 4, INTER: int = 128,
                            mask_id: int = 63, stop_id: int = 62, max_seq: int = 256):
    torch.manual_seed(seed)
    dev = torch.device("cpu")
    dt = torch.float64
    hd = E_dim // NH
    cfg = _DirectCfgStub({
        "n_embd": E_dim, "num_attention_heads": NH, "intermediate_size": INTER,
        "norm_eps": 1e-6, "rope_base": 10000.0, "qk_bias": True,
        "padded_vocab_size": V, "pad_token_id": mask_id,
        "n_layers_in_prelude": 2, "n_layers_in_recurrent_block": 4, "n_layers_in_coda": 2,
        "mean_recurrence": T,
    })
    pos = torch.arange(max_seq, dtype=torch.float32)
    inv = 1.0 / (cfg.rope_base ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    freqs = torch.outer(pos, inv)
    fc = torch.stack([freqs.cos(), freqs.sin()], dim=-1)[None, :, None, :, :]  # [1, S, 1, hd/2, 2]
    stub = _DirectModelStub(cfg, fc)

    def rand_layer():
        s = E_dim ** -0.5
        return dict(
            n1=1.0 + 0.15 * torch.randn(E_dim, dtype=dt),
            qkv_w=torch.randn(3 * E_dim, E_dim, dtype=dt) * s,
            qkv_b=torch.cat([0.05 * torch.randn(2 * E_dim, dtype=dt), torch.zeros(E_dim, dtype=dt)]),
            o_w=torch.randn(E_dim, E_dim, dtype=dt) * s,
            n2=1.0 + 0.15 * torch.randn(E_dim, dtype=dt),
            n3=1.0 + 0.15 * torch.randn(E_dim, dtype=dt),
            fc_w=torch.randn(2 * INTER, E_dim, dtype=dt) * s,
            down_w=torch.randn(E_dim, INTER, dtype=dt) * (INTER ** -0.5),
            n4=1.0 + 0.15 * torch.randn(E_dim, dtype=dt),
        )

    fm = object.__new__(HuginnFastModel)
    fm.m = stub
    fm.dev, fm.dt = dev, dt
    fm.E, fm.NH, fm.HD, fm.HD_kv = E_dim, NH, hd, hd
    fm.P, fm.Lc_core, fm.C = 2, 4, 2
    fm.emb_scale = stub.emb_scale
    ops = HuginnFastOps.__new__(HuginnFastOps)
    ops.dev, ops.dt = dev, dt
    ops.E, ops.NH, ops.HD, ops.INTER, ops.eps, ops.log = E_dim, NH, hd, INTER, cfg.norm_eps, lambda *a: None
    ops.freqs = fc
    ops.cos_sin32 = torch.cat([fc[0, :, 0, :, 0].float(), fc[0, :, 0, :, 1].float()], dim=-1).contiguous()
    ops.chosen = {"rms": "torch", "add_rms": "torch", "silu": "torch", "rope": "torch"}
    ops._norm_mods = {}
    ops.rms, ops.add_rms, ops.silu_mul, ops.rope = ops.t_rms, ops.t_add_rms, ops.t_silu_mul, ops.t_rope
    fm.ops = ops

    fm.wte = torch.randn(V, E_dim, dtype=dt) * 0.2
    fm.adap_w = torch.randn(E_dim, 2 * E_dim, dtype=dt) * ((2 * E_dim) ** -0.5)
    fm.adap_b = torch.zeros(E_dim, dtype=dt)
    fm.ln_f = 1.0 + 0.15 * torch.randn(E_dim, dtype=dt)
    fm.lm_head = torch.randn(V, E_dim, dtype=dt) * (4.0 * (E_dim ** -0.5))
    fm.lm_head[mask_id].fill_(-50.0)
    fm.lm_head[stop_id].mul_(1.2)
    fm.pre_layers = [rand_layer() for _ in range(2)]
    fm.core_layers = [rand_layer() for _ in range(4)]
    fm.coda_layers = [rand_layer() for _ in range(2)]
    return fm


def test_ouro_wake():
    V = 24
    MASK, EOS = V - 1, V - 2
    NV = V - 1
    prompts = [
        [3, 5, 7, 11, 13, 17, 19, 2, 4, 6],
        [2, 4, 8, 16, 1, 9, 6, 5, 6, 10, 12, 4, 8],
    ]
    checks = 0
    total_rb = 0

    for seed, gate_scale, eos_bias in [(0, 4.0, False), (7, 8.0, True)]:
        W, c = make_random_ouro(seed, V, gate_scale=gate_scale, eos_id=EOS, mask_id=MASK, eos_bias=eos_bias)
        eng = L2.LateEngine(W, c, torch.device("cpu"), torch.float64, S=256, fused=False, log=lambda *a: None, M=8, Bmax=8)
        dec_ar = E.Lattice(eng, 0, NV, [EOS], MASK, attn="torch", graphs=False)

        for pr in prompts:
            ref = dec_ar.decode(pr, max_new=16, prefill_attn="torch")
            ref_raw = ref["gen_raw"][:16]

            for K in (1, 2, 3):
                for shape in ("b0", "latt:2", "latt:4"):
                    B, sets = L2.parse_shape(shape)

                    d_old = LK.LateKLattice(eng, K, B, sets, NV, [EOS], MASK, attn="torch", graphs=False)
                    r_old = d_old.decode(pr, max_new=16, prefill_attn="torch")

                    d_inf = WakeKLattice(eng, K, B, sets, NV, [EOS], MASK, tau=float("inf"), suffix=False, attn="torch", graphs=False)
                    r_inf = d_inf.decode(pr, max_new=16, prefill_attn="torch")
                    assert r_inf["gen_raw"] == r_old["gen_raw"][:16], (
                        f"Ouro tau=inf mismatch K={K} {shape}: {r_inf['gen_raw']} vs {r_old['gen_raw'][:16]}"
                    )
                    checks += 1

                    for sfx in (False, True):
                        d_wk = WakeKLattice(eng, K, B, sets, NV, [EOS], MASK, tau=0.0, suffix=sfx, attn="torch", graphs=False)
                        r_wk = d_wk.decode(pr, max_new=16, prefill_attn="torch")
                        assert r_wk["gen_raw"] == ref_raw, (
                            f"Ouro Wake tau=0 EXACTNESS FAIL (seed={seed}, K={K}, {shape}, sfx={sfx}):\n"
                            f"  got : {r_wk['gen_raw']} (rb={r_wk['n_rollbacks']}, drain={r_wk['n_drain_steps']})\n"
                            f"  want: {ref_raw}"
                        )
                        total_rb += r_wk["n_rollbacks"]
                        checks += 1

    print(f"[PASS] Ouro Wake Verification CPU fp64 exactness: {checks} checks, {total_rb} rollbacks verified cleanly!")


def test_huginn_wake():
    checks = 0
    total_rb = 0
    mask_id = 63
    stop_id = 62

    for T, K_list in [(6, [2, 3, 4]), (8, [2, 4, 6]), (12, [4, 6])]:
        L_max = 128
        fm = make_random_huginn_fast(100 + T, T=T, mask_id=mask_id, stop_id=stop_id, max_seq=L_max + 64)
        prompts = [[4, 9, 15, 21, 9, 15], [7, 13, 19, 29, 31, 37, 13, 19, 29]]

        for p_idx, pr in enumerate(prompts):
            ns = make_noise(fm.m, L_max, 5000 + T * 10 + p_idx, torch.device("cpu"), torch.float64)
            d_seq = HuginnFastDecoder(
                fm, K=T, B=0, L_max=L_max, T=T, bg="none", D_bg=0,
                nv=63, mask_id=mask_id, stop_ids=(stop_id,), attn="torch", graphs=False,
            )
            ref_ids = d_seq.decode_one(pr, ns, max_new=20)["gen_ids"]

            d_full_sfx = HuginnWakeDecoder(
                fm, K=T, B=4, L_max=L_max, T=T, bg="none", D_bg=0, tau=float("inf"), suffix=True,
                nv=63, mask_id=mask_id, stop_ids=(stop_id,), attn="torch", graphs=False,
            )
            assert d_full_sfx.decode_one(pr, ns, max_new=20)["gen_ids"] == ref_ids
            checks += 1

            for K in K_list:
                for B in (0, 2, 4):
                    d_fast = HuginnFastDecoder(
                        fm, K=K, B=B, L_max=L_max, T=T, bg="pipe", D_bg=T,
                        nv=63, mask_id=mask_id, stop_ids=(stop_id,), attn="torch", graphs=False,
                    )
                    d_winf = HuginnWakeDecoder(
                        fm, K=K, B=B, L_max=L_max, T=T, bg="pipe", D_bg=T, tau=float("inf"), suffix=False,
                        nv=63, mask_id=mask_id, stop_ids=(stop_id,), attn="torch", graphs=False,
                    )
                    assert d_fast.decode_one(pr, ns, max_new=20)["gen_ids"] == d_winf.decode_one(pr, ns, max_new=20)["gen_ids"]
                    checks += 1

                    for sfx in (False, True):
                        d_wk = HuginnWakeDecoder(
                            fm, K=K, B=B, L_max=L_max, T=T, bg="pipe", D_bg=T, tau=0.0, suffix=sfx,
                            nv=63, mask_id=mask_id, stop_ids=(stop_id,), attn="torch", graphs=False,
                        )
                        out_wk = d_wk.decode_one(pr, ns, max_new=20)
                        assert out_wk["gen_ids"] == ref_ids, (
                            f"Huginn Wake tau=0 EXACTNESS FAIL (T={T}, K={K}, B={B}, sfx={sfx}):\n"
                            f"  got : {out_wk['gen_ids']} (rb={out_wk['n_rollbacks']})\n"
                            f"  want: {ref_ids}"
                        )
                        total_rb += out_wk["n_rollbacks"]
                        checks += 1

    print(f"[PASS] Huginn Wake Verification + T-Scaling CPU fp64 exactness: {checks} checks, {total_rb} rollbacks verified cleanly!")


if __name__ == "__main__":
    test_ouro_wake()
    test_huginn_wake()
    print("ALL WAKE VERIFICATION & DEPTH-SCALING CPU EXACTNESS GATES PASSED!")
