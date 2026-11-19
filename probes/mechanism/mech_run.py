"""mech_run.py -- GPU driver of the teacher-forced late-entry mechanism probes (10-23). One model per invocation.

Sequences: v2's own full-depth (ar:4) generations, GSM8K (gsm8k_cot_zeroshot) docs 0-299 and IFEval docs 0-149, i.e. the materialized
lm-eval prompt (chat template exactly as the campaign: lmeval_ouro/<v2>/_tasks/<task>.jsonl, tokenized add_special_tokens=False) +
the generated text (re-tokenized, as lmeval_ouro/tf4_posthoc.py does; rt_ok flags docs whose re-tokenization does not have the
recorded token count). The same token sequences are used for every model (v2's tokenizer). Positions p = P0 .. N-2 (x_p and x_{p+1}
both generated text).

Per doc and model (all fp32, TF32 off; readouts/NLL fp32):
  full / ar1..ar{T-1}          normal forward, gate mixture over all / the first k passes
  cell_D_t (D=1..T, t=1..D)    late-entry slot with t passes at pass indices D-t+1..D, context run to D passes (P1, D x t grid)
  p3_KdVs / p3_KsVd            L_1 reading deep keys + pass-1 values / pass-1 keys + deep values (P3; both deep = cell_T_1,
                               both shallow = cell_1_1)
  p4_last_m / p4_old_m         L_1 reading deep K/V only for the last m context positions / only for positions older than m, pass-1
                               K/V elsewhere (P4, m in M_LIST); p4_prompt / p4_gen / p4_sink / p4_nosink: deep only for the prompt /
                               the generated part / position 0 / everything but position 0
  P2: cosine of L_1's final normed state with the full forward's pass-1..T states at p (+ pass-k vs pass-T), linear CKA (doc-group
      accumulators, bootstrap over doc groups, computed here), logit lens agreement per layer for L_1, normal pass 1, normal pass T
  P5: attention entropy / mass on p-7..p / on position 0 / on itself, per layer, head-averaged, for L_1, normal pass-1 and pass-T
      queries at p (sums split by whether L_1 agrees with full depth)
Runtime checks per doc: max |logit diff| of cell_k_k vs ar_k and cell_T_T vs full (should be ~1e-5 in fp32).

    python mech_run.py --model <dir> --model-tag v2 --out <dir> [--tasks "gsm8k_cot_zeroshot:0:300 ifeval:0:150"] [--limit n]
"""
import argparse, glob, json, os, pickle, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mech_core as mc  # noqa: E402

LMO = "eval/data"
V2_DIR = "Xalphinions/ouro-2.6b-late-entry"   # lmo_common.MODELS['v2'] (text tokenizer)
M_LIST = (1, 2, 4, 8, 16, 64)
EXTRAS = ("prompt", "gen", "sink", "nosink")
CKA_PAIRS = None   # filled by CKAAcc (names L1, P1..PT)


def load_docs(task, lo, hi, tok, src_tag="v2", root=LMO):
    ctx = {}
    for l in open(f"{root}/{src_tag}/_tasks/{task}.jsonl"):
        d = json.loads(l); ctx[d["doc_id"]] = d["context"]
    recs = {}
    for f in sorted(glob.glob(f"{root}/{src_tag}/{task}/ar_4/shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if lo <= r["doc_id"] < hi and r.get("finish") != "error": recs.setdefault(r["doc_id"], r)
    out = []
    for d in sorted(recs):
        r = recs[d]; c = tok(ctx[d], add_special_tokens=False)["input_ids"]; g = tok(r.get("text_raw") or r["text"], add_special_tokens=False)["input_ids"]
        out.append(dict(doc_id=d, ctx=c, gen=g, rt_ok=len(g) == r["tokens"], finish=r["finish"]))
    missing = sorted(set(range(lo, hi)) - set(recs))
    return out, missing


def doc_configs(T, S, N, P0, dev):
    """[(name, specs, want_stats_lens)] -- see the module docstring"""
    C = mc.causal_strict(S, N, dev); j = torch.arange(N, device=dev); dist = S[:, None] - j[None]; d, s = T - 1, 0
    cf = []
    for D in range(1, T + 1):
        for t in range(1, D + 1): cf.append((f"cell_{D}_{t}", mc.cell_specs(D, t, C), (D, t) == (T, 1)))
    cf += [("p3_KdVs", [[(d, s, C)]], False), ("p3_KsVd", [[(s, d, C)]], False)]
    for m in M_LIST:
        near = dist <= m
        cf.append((f"p4_last_{m}", [[(d, d, C & near), (s, s, C & ~near)]], False))
        cf.append((f"p4_old_{m}", [[(d, d, C & ~near), (s, s, C & near)]], False))
    sel = dict(prompt=(j < P0)[None], gen=(j >= P0)[None], sink=(j == 0)[None], nosink=(j > 0)[None])
    for e in EXTRAS:
        A = sel[e].expand_as(C); cf.append((f"p4_{e}", [[(d, d, C & A), (s, s, C & ~A)]], False))
    return cf


@torch.no_grad()
def process_doc(probe, ids, P0, dev):
    """ids: list of token ids (prompt + generation), P0 = prompt length. Returns (record, feats) where feats = [L1 state, pass-1..T states]
    at S (for CKA). record: per-config argmax / NLL, cosines, lens agreement, attention-stat sums, runtime checks."""
    T, L = probe.T, probe.L; N = len(ids); ids_t = torch.tensor(ids, device=dev)
    S = torch.arange(P0, N - 1, device=dev); tgt = ids_t[S + 1]
    FF = probe.full_forward(ids_t, S, stats_passes=(0, T - 1), lens_passes=(0, T - 1))
    res, checks = {}, {}
    full = mc.mixture(FF["lg"], FF["g"]); res["full"] = mc.summarize(full, tgt, extra=True); full_am = full.argmax(-1)
    ar = {T: full}
    for k in range(1, T):
        ar[k] = mc.mixture(FF["lg"][:k], FF["g"][:k]); res[f"ar{k}"] = mc.summarize(ar[k], tgt)
    L1 = None
    for name, specs, want in doc_configs(T, S, N, P0, dev):
        out = probe.slot(ids_t, S, FF, specs, stats=want, lens=want); res[name] = mc.summarize(out["mix"], tgt)
        D, t = (int(x) for x in name.split("_")[1:]) if name.startswith("cell_") else (0, 0)
        if D and D == t: checks[f"{name}-ar{D}"] = float((out["mix"] - ar[D]).abs().max())
        if want: L1 = out
        del out
    # ---- P2: representation
    hL = L1["h"]; hs = FF["hs"]
    cos = [torch.nn.functional.cosine_similarity(hL, hs[k], dim=-1) for k in range(T)] + [torch.nn.functional.cosine_similarity(hs[k], hs[T - 1], dim=-1) for k in range(T - 1)]
    lens = dict(L1=(L1["lens"] == full_am[None]).cpu().numpy(), pass1=(FF["lens"][0] == full_am[None]).cpu().numpy(), passT=(FF["lens"][T - 1] == full_am[None]).cpu().numpy())
    # ---- P5: attention stats, sums split by L_1 agreement
    agree = torch.tensor(res[f"cell_{T}_1"]["am"], device=dev) == full_am.int()
    st = torch.stack([L1["stats"], FF["stats"][0], FF["stats"][T - 1]])            # [3, L, nq, 4]
    attn = torch.stack([st[:, :, agree].sum(2), st[:, :, ~agree].sum(2)], 1).permute(0, 1, 3, 2)   # [3 qtype, 2 (agree, disagree), 4 stat, L]
    rec = dict(N=N, P0=P0, nq=int(S.shape[0]), tgt=tgt.int().cpu().numpy(), res=res, checks=checks,
               cos=torch.stack(cos, 1).float().cpu().numpy(), lens=lens, attn=attn.double().cpu().numpy(),
               attn_n=np.array([int(agree.sum()), int((~agree).sum())]))
    feats = [hL] + [hs[k] for k in range(T)]
    del FF, L1
    return rec, feats


class CKAAcc:
    """linear CKA between L_1's state and the full forward's pass-k states, pooled over positions; doc-group accumulators so a bootstrap
    over doc groups gives CIs (CKA(X,Y) = ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F), Xc centered over positions)"""

    def __init__(self, T, d, dev):
        self.names = ["L1"] + [f"P{k + 1}" for k in range(T)]; F_ = len(self.names)
        self.pairs = [(a, a) for a in range(F_)] + [(0, k) for k in range(1, F_)] + [(k, F_ - 1) for k in range(1, F_ - 1)]
        self.d, self.dev, self.groups = d, dev, []; self._new()

    def _new(self):
        F_ = len(self.names); self.n = 0
        self.s = torch.zeros(F_, self.d, dtype=torch.float64, device=self.dev); self.C = torch.zeros(len(self.pairs), self.d, self.d, dtype=torch.float64, device=self.dev)

    def add(self, feats):
        X = [f.double() for f in feats]; self.n += X[0].shape[0]
        for a, x in enumerate(X): self.s[a] += x.sum(0)
        for pi, (a, b) in enumerate(self.pairs): self.C[pi] += X[a].T @ X[b]

    def close(self):
        if self.n: self.groups.append((self.n, self.s.cpu(), self.C.cpu()))
        self._new()

    def result(self, B=1000, seed=0):
        self.close(); del self.C; torch.cuda.empty_cache() if self.dev != "cpu" else None
        G = len(self.groups); n = torch.tensor([g[0] for g in self.groups], dtype=torch.float64, device=self.dev)
        s = torch.stack([g[1] for g in self.groups]).to(self.dev); C = torch.stack([g[2] for g in self.groups]).to(self.dev)
        idx = {p: i for i, p in enumerate(self.pairs)}

        def cka(w):
            nn_ = w @ n; ss = torch.einsum("g,gfd->fd", w, s); CC = torch.einsum("g,gpij->pij", w, C)
            fro = torch.stack([(CC[i] - torch.outer(ss[a], ss[b]) / nn_).norm() for i, (a, b) in enumerate(self.pairs)])
            return {f"{self.names[a]}~{self.names[b]}": float(fro[idx[(a, b)]] ** 2 / (fro[idx[(a, a)]] * fro[idx[(b, b)]])) for (a, b) in self.pairs if a != b}

        point = cka(torch.ones(G, dtype=torch.float64, device=self.dev))
        rng = np.random.default_rng(seed); boots = []
        for _ in range(B):
            w = torch.tensor(np.bincount(rng.integers(0, G, G), minlength=G), dtype=torch.float64, device=self.dev); boots.append(cka(w))
        out = {k: dict(point=v, lo=float(np.percentile([b[k] for b in boots], 2.5)), hi=float(np.percentile([b[k] for b in boots], 97.5))) for k, v in point.items()}
        del s, C; self.groups = []
        return dict(groups=G, positions=int(n.sum()), bootstrap=B, pairs=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--model-tag", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="gsm8k_cot_zeroshot:0:300:20 ifeval:0:150:15")   # task:lo:hi:cka_groups
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--cka-boot", type=int, default=1000)
    ap.add_argument("--text-tok", default=V2_DIR)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    mc.strict_fp32(); dev = a.device if torch.cuda.is_available() else "cpu"; t0 = time.time()
    m, tok_m = mc.load_model(a.model, dev); nv = len(tok_m); probe = mc.Probe(m, nv)
    tok = AutoTokenizer.from_pretrained(a.text_tok)
    os.makedirs(a.out, exist_ok=True)
    meta = dict(model=a.model, model_tag=a.model_tag, dtype="float32 (TF32 off), readout/NLL fp32", T=probe.T, L=probe.L, nv=nv, vocab_rows=int(m.lm_head.weight.shape[0]),
                M_LIST=list(M_LIST), EXTRAS=list(EXTRAS), stats=list(mc.Probe.STATS), torch=torch.__version__, host=os.uname().nodename,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", text_tok=a.text_tok)
    print(f"[mech] {a.model_tag}: loaded {a.model} T={probe.T} L={probe.L} nv={nv} rows={meta['vocab_rows']} on {meta['gpu']} ({time.time() - t0:.0f}s)", flush=True)
    for spec in a.tasks.split():
        task, lo, hi, ng = spec.split(":"); lo, hi, ng = int(lo), int(hi), int(ng)
        docs, missing = load_docs(task, lo, hi, tok)
        if a.limit: docs = docs[: a.limit]
        print(f"[mech] {task}: {len(docs)} docs (missing {missing}), rt_ok {sum(d['rt_ok'] for d in docs)}", flush=True)
        cka = CKAAcc(probe.T, m.config.hidden_size, dev); recs = []; t1 = time.time(); ntok = 0; worst = {}
        for i, d in enumerate(docs):
            ids = d["ctx"] + d["gen"]
            assert max(ids) < nv, f"token id {max(ids)} >= nv {nv}"
            if len(d["gen"]) < 2: print(f"[mech] skip doc {d['doc_id']} (gen {len(d['gen'])})", flush=True); continue
            rec, feats = process_doc(probe, ids, len(d["ctx"]), dev)
            rec.update(doc_id=d["doc_id"], task=task, rt_ok=d["rt_ok"], finish=d["finish"]); recs.append(rec); ntok += rec["nq"]
            cka.add(feats); del feats
            if (i + 1) * ng // len(docs) != i * ng // len(docs): cka.close()
            for k, v in rec["checks"].items(): worst[k] = max(worst.get(k, 0.0), v)
            if i % 25 == 0 or i == len(docs) - 1:
                el = time.time() - t1
                print(f"[mech] {task} {i + 1}/{len(docs)} doc {d['doc_id']} N={rec['N']} nq={rec['nq']} | {ntok} pos, {el:.0f}s ({ntok / max(el, 1e-9):.0f} pos/s) "
                      f"| L1 agree {np.mean(rec['res'][f'cell_{probe.T}_1']['am'] == rec['res']['full']['am']):.3f} | worst checks {max(worst.values()):.2e}", flush=True)
            if dev != "cpu": torch.cuda.empty_cache()
        ck = cka.result(B=a.cka_boot) if recs else None; del cka
        meta_t = dict(meta, task=task, lo=lo, hi=hi, docs=len(recs), positions=ntok, missing=missing, worst_checks=worst, sec=time.time() - t1, cka=ck,
                      written=time.strftime("%Y-%m-%d %H:%M:%S"))
        tmp = f"{a.out}/raw_{task}.pkl.tmp"
        with open(tmp, "wb") as f: pickle.dump(dict(meta=meta_t, docs=recs), f, protocol=4)
        os.replace(tmp, f"{a.out}/raw_{task}.pkl")
        json.dump(meta_t, open(f"{a.out}/meta_{task}.json", "w"), indent=1)
        print(f"[mech] {task} DONE {len(recs)} docs {ntok} positions in {time.time() - t1:.0f}s; worst checks {worst}; CKA {json.dumps(ck['pairs'] if ck else None)[:600]}", flush=True)
    print(f"[mech] ALL DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
