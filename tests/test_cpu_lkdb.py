"""test_cpu_lkdb.py -- CPU gates for the deep-bonus (lkdb) finish path (fixed-shape, graph-ready version).

  PYTHONNOUSERSITE=1 NT=8 python test_cpu_lkdb.py [--n 48]

(1) identity: theta-impossible gate (LK_DB_THETA=0 -> never fires): lkdb decode == lk decode, tokens and per-step log.
    Validates that the extra db row (scratch-slot writes) never perturbs the block path.
(2) forced fire (LK_DB_THETA=2 -> fires at every c==0 with proposals): at every fire, the deep readout must equal the
    reference full-depth gate-mixture argmax at the fired position (RefLate.full_depth of the committed prefix),
    fp64-exact on tiny random models. Exercises the fixed-shape finish batch incl. short groups (pad rows in front).
"""
import argparse, os, sys, time
import torch

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=48); a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("NT", "8"))); T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "decode")); sys.path.insert(0, os.path.join(HERE, "..", "train"))
import engine as E
import latek as LK
import ref_late_k as RK
from dllm.pipelines.a2d import A2DOuroConfig, A2DOuroLMHeadModel
def P(*x): print(f"[{time.time() - T0:6.0f}s]", *x, flush=True)

V = 24; MASK, EOS, IM = V - 1, V - 2, V - 3; NV = V - 3


def tiny(seed, head_scale, gate_scale=1.0, layers=2, dtype=torch.float64):
    torch.manual_seed(seed)
    cfg = A2DOuroConfig(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=4,
                        total_ut_steps=4, max_position_embeddings=1024, use_cache=True, tie_word_embeddings=False, rope_theta=10000.0)
    cfg.model_type = "a2d-ouro"; cfg._attn_implementation = "eager"; m = A2DOuroLMHeadModel(cfg).eval()
    with torch.no_grad():
        m.lm_head.weight.mul_(head_scale); m.lm_head.weight[NV:].zero_(); m.model.early_exit_gate.weight.mul_(gate_scale)
        for n, p in m.named_parameters():
            if "layernorm" in n or n.endswith("norm.weight"): p.add_(0.3 * torch.randn_like(p))
    return m.to(dtype)


PROMPTS = [[3, 5, 7, 11, 13, 17, 19, 2, 4, 6], [2, 4, 8, 16, 1, 9, 6, 5, 6, 10, 12], [10, 11, 7, 7, 1, 3, 3, 9, 14, 15],
           [12, 0, 14, 2, 18, 19, 1, 1, 5, 7, 2, 3], [9, 9, 15, 16, 17, 3, 18, 0, 0, 4]]

FIRES = []
_ORIG = LK.LateKLattice._db_finish
def _patched(self, lc, lo):
    d = _ORIG(self, lc, lo); FIRES.append((lc, d)); return d
LK.LateKLattice._db_finish = _patched

fails = 0
for name, seed, hs, gs in [("s0 head x6", 0, 6.0, 1.0), ("s2 head x10 gate x8", 2, 10.0, 8.0)]:
    m = tiny(seed, hs, gs)
    Wd, cfg = E.weights_from_hf(m)
    eng = LK.LateEngine(Wd, cfg, "cpu", torch.float64, S=512, fused=False, log=lambda *x: None, M=8, Bmax=16)
    ref = RK.RefLateK(m, NV, [EOS, IM], MASK)
    for K in (2, 3):
        B = 2; sets = [(aa, B) for aa in range(B + 1)]
        os.environ["LK_DB_THETA"] = "0"
        lk = LK.LateKLattice(eng, K, B, sets, NV, [EOS, IM], MASK, attn="torch", graphs=False)
        db0 = LK.LateKLattice(eng, K, B, sets, NV, [EOS, IM], MASK, attn="torch", graphs=False, db=True)
        ident = True
        for pi, pr in enumerate(PROMPTS):
            r1 = lk.decode(pr, a.n); r2 = db0.decode(pr, a.n)
            if r1["gen_raw"] != r2["gen_raw"] or r1["log"] != r2["log"]:
                ident = False; fails += 1; P("FAIL identity", name, "K", K, "prompt", pi)
        P(f"identity {name} K={K}: {ident}")
        os.environ["LK_DB_THETA"] = "2"
        db1 = LK.LateKLattice(eng, K, B, sets, NV, [EOS, IM], MASK, attn="torch", graphs=False, db=True)
        nchk = nok = 0
        for pi, pr in enumerate(PROMPTS):
            FIRES.clear()
            r = db1.decode(pr, a.n)
            known = list(pr) + r["gen_raw"]
            for lc, deep in FIRES[:6]:
                _, refa = ref.full_depth(known[: lc + 1])
                nchk += 1
                if refa == deep: nok += 1
                else: fails += 1; P("FAIL forced-fire", name, "K", K, "prompt", pi, "lc", lc, "deep", deep, "ref", refa)
        P(f"forced-fire {name} K={K}: {nok}/{nchk}")
P("FAILS", fails)
sys.exit(1 if fails else 0)
