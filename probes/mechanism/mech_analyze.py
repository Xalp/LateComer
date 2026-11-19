"""mech_analyze.py -- aggregate the teacher-forced late-entry probes (mech_run.py outputs) into mech_summary.{json,md}. CPU, numpy only.

Every rate / mean is a ratio of per-document sums; 95 % CIs are percentile bootstraps over DOCUMENTS (B=2000, the same resampling
weights for every metric, model and config of a task, so differences between configs or models are paired). CKA CIs come from
mech_run.py (bootstrap over doc groups).
    python mech_analyze.py [--root probes/mechanism/results] [--models v2 sft abl stock26]
"""
import argparse, json, os, pickle, time

import numpy as np

ROOT = "probes/mechanism/results"
TASKS = [("gsm8k_cot_zeroshot", "GSM8K"), ("ifeval", "IFEval")]
MODELS = [("v2", "v2 (SFT+block+mid-pass)"), ("sft", "SFT only"), ("abl", "block, no mid-pass"), ("stock26", "stock Ouro-2.6B")]
STATS = ("entropy", "last8", "sink", "self")
QTYPES = ("L1", "pass1", "passT")
B = 2000


class Boot:
    def __init__(self, ndocs, seed):
        rng = np.random.default_rng(seed)
        self.W = np.stack([np.bincount(rng.integers(0, ndocs, ndocs), minlength=ndocs) for _ in range(B)]).astype(np.float64)   # [B, ndocs]

    def ratio(self, num, den):
        """num, den: [ndocs] or [ndocs, k] per-document sums -> dict(point, lo, hi) (lists for k)"""
        num = np.asarray(num, np.float64); den = np.asarray(den, np.float64)
        pt = num.sum(0) / den.sum(0); bs = (self.W @ num) / (self.W @ den if den.ndim == num.ndim else (self.W @ den)[:, None])
        lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
        f = (lambda x: x.tolist()) if np.ndim(pt) else float
        return dict(point=f(pt), lo=f(lo), hi=f(hi))

    def diff(self, num_a, num_b, den):
        """paired difference of two ratios with the same denominator"""
        return self.ratio(np.asarray(num_a, np.float64) - np.asarray(num_b, np.float64), den)


def load(root, tag, task):
    p = f"{root}/{tag}/raw_{task}.pkl"
    if not os.path.exists(p): return None
    with open(p, "rb") as f: return pickle.load(f)


def per_doc(docs, fn):
    return np.array([fn(d) for d in docs], np.float64)


def analyze_model(docs, meta, bt):
    T = meta["T"]; L = meta["L"]; out = {}
    n = per_doc(docs, lambda d: d["nq"]); full = lambda d: d["res"]["full"]["am"]
    cfgs = [c for c in docs[0]["res"]]
    ag = {c: per_doc(docs, lambda d, c=c: (d["res"][c]["am"] == full(d)).sum()) for c in cfgs}
    tk = {c: per_doc(docs, lambda d, c=c: (d["res"][c]["am"] == d["tgt"]).sum()) for c in cfgs}
    nl = {c: per_doc(docs, lambda d, c=c: d["res"][c]["nll"].astype(np.float64).sum()) for c in cfgs}
    out["configs"] = {c: dict(agree=bt.ratio(ag[c], n), tok_acc=bt.ratio(tk[c], n), nll=bt.ratio(nl[c], n)) for c in cfgs}
    ar = lambda k: "full" if k == T else f"ar{k}"
    # P1: L_t vs the k-pass model, paired
    out["P1"] = {f"t{t}": dict(L=f"cell_{T}_{t}", ar=ar(t), d_agree=bt.diff(ag[f"cell_{T}_{t}"], ag[ar(t)], n), d_nll=bt.diff(nl[f"cell_{T}_{t}"], nl[ar(t)], n)) for t in range(1, T)}
    # P1b: agreement binned by the full-depth top-1 probability at p (are disagreements near-ties?)
    bins = [0.0, 0.5, 0.8, 0.95, 1.01]; out["P1_bins"] = {}
    for bi in range(len(bins) - 1):
        sel = lambda d, bi=bi: (d["res"]["full"]["p1"] >= bins[bi]) & (d["res"]["full"]["p1"] < bins[bi + 1])
        nb = per_doc(docs, lambda d: sel(d).sum())
        out["P1_bins"][f"{bins[bi]:.2f}-{min(bins[bi + 1], 1):.2f}"] = dict(
            frac=bt.ratio(nb, n), **{c: bt.ratio(per_doc(docs, lambda d, c=c: ((d["res"][c]["am"] == full(d)) & sel(d)).sum()), nb) for c in (f"cell_{T}_1", "ar1", f"cell_{T}_2", "ar2")})
    # D x t grid, also vs the context depth's own AR (ar_D)
    out["grid_vs_ownAR"] = {f"cell_{D}_{t}": bt.ratio(per_doc(docs, lambda d, D=D, t=t: (d["res"][f"cell_{D}_{t}"]["am"] == d["res"][ar(D)]["am"]).sum()), n)
                            for D in range(1, T + 1) for t in range(1, D + 1)}
    # P2: cosines, lens
    cosn = ["L1~P%d" % (k + 1) for k in range(T)] + ["P%d~P%d" % (k + 1, T) for k in range(T - 1)]
    out["cos"] = dict(zip(cosn, [bt.ratio(per_doc(docs, lambda d, i=i: d["cos"][:, i].astype(np.float64).sum()), n) for i in range(len(cosn))]))
    lens = {}
    for q in ("L1", "pass1", "passT"):
        curve = bt.ratio(np.stack([d["lens"][q].sum(1) for d in docs]).astype(np.float64), n)
        fin = per_doc(docs, lambda d, q=q: d["lens"][q][-1].sum())
        def settle(a):                                                  # [L, nq] bool -> 1-based layer from which it agrees to the end (L+1 = never)
            return L + 1 - np.cumprod(a[::-1], 0).sum(0)                # trailing run of agreeing layers
        def first(a):
            return np.where(a.any(0), a.argmax(0) + 1, L + 1)
        st_sum = per_doc(docs, lambda d, q=q: settle(d["lens"][q])[d["lens"][q][-1]].sum())
        fi_sum = per_doc(docs, lambda d, q=q: first(d["lens"][q])[d["lens"][q][-1]].sum())
        st_all = np.concatenate([settle(d["lens"][q])[d["lens"][q][-1]] for d in docs]); fi_all = np.concatenate([first(d["lens"][q])[d["lens"][q][-1]] for d in docs])
        lens[q] = dict(curve=curve, final_agree=bt.ratio(fin, n), settle_mean=bt.ratio(st_sum, fin), first_mean=bt.ratio(fi_sum, fin),
                       settle_median=float(np.median(st_all)) if len(st_all) else None, first_median=float(np.median(fi_all)) if len(fi_all) else None,
                       settle_q=[float(np.percentile(st_all, q_)) for q_ in (10, 25, 50, 75, 90)] if len(st_all) else None)
    out["lens"] = lens
    # P5: attention stats per layer [qtype][stat] -> curve; also L1 split by agreement
    A = np.stack([d["attn"] for d in docs]); An = np.stack([d["attn_n"] for d in docs]).astype(np.float64)   # [nd, 3, 2, 4, L], [nd, 2]
    out["attn"] = {QTYPES[qi]: {STATS[si]: bt.ratio(A[:, qi, :, si, :].sum(1), An.sum(1)) for si in range(4)} for qi in range(3)}
    out["attn_mean"] = {QTYPES[qi]: {STATS[si]: bt.ratio(A[:, qi, :, si, :].sum(1).mean(-1), An.sum(1)) for si in range(4)} for qi in range(3)}   # layer-averaged
    out["attn_split"] = {QTYPES[qi]: {f"{STATS[si]}_{w}": bt.ratio(A[:, qi, wi, si, :].mean(-1), An[:, wi]) for si in range(4) for wi, w in enumerate(("agree", "disagree"))} for qi in range(3)}
    out["attn_d_L1_minus"] = {q: {STATS[si]: bt.diff(A[:, 0, :, si, :].sum(1).mean(-1), A[:, qi, :, si, :].sum(1).mean(-1), An.sum(1)) for si in range(4)} for qi, q in ((1, "pass1"), (2, "passT"))}
    # data hygiene
    out["docs"] = len(docs); out["positions"] = int(n.sum()); out["rt_ok_docs"] = int(sum(d["rt_ok"] for d in docs))
    out["full_tok_acc"] = out["configs"]["full"]["tok_acc"]; out["worst_checks"] = meta.get("worst_checks"); out["cka"] = meta.get("cka")
    out["meta"] = {k: meta[k] for k in ("model", "dtype", "T", "L", "nv", "vocab_rows", "gpu", "torch", "sec") if k in meta}
    return out


def fmt(r, p=3):
    if r is None: return "n/a"
    s = f"{{:.{p}f}}"
    return f"{s.format(r['point'])} [{s.format(r['lo'])}, {s.format(r['hi'])}]"


def short(r, p=3):
    s = f"{{:.{p}f}}"
    return s.format(r["point"]).lstrip("0") if r["point"] >= 0 else "-" + s.format(-r["point"]).lstrip("0")


def write_md(S, path, models_present):
    T = 4; lines = [f"# Late-entry mechanism probes (teacher forced) — {S['written']}", "",
                    "Sequences: v2's own full-depth (ar:4) generations, GSM8K (gsm8k_cot_zeroshot) docs 0-299 and IFEval docs 0-149, prompt = the "
                    "campaign's materialized chat-template context; positions p = every generated position whose next token is generated text. "
                    "L_t(p) = a slot that enters at pass T-t+1 with the raw embedding of x_p, reads the FULL-DEPTH pass-k K/V of all positions < p "
                    "(+ its own key), readout = renormalised gate mixture over its t passes. Agreement = argmax equals the model's own full-depth "
                    "(4-pass) argmax at p. Numerics: fp32 weights/activations, TF32 off, readouts fp32. 95 % CIs: bootstrap over documents "
                    "(paired across configs and models). Model names: " + ", ".join(f"**{t}** = {n}" for t, n in MODELS if t in models_present) + ".", ""]
    for task, tname in TASKS:
        R = S["tasks"].get(task)
        if not R: continue
        lines += [f"## {tname}", ""]
        any_ = next(iter(R.values()))
        lines += [f"{any_['docs']} docs, {any_['positions']} positions ({any_['rt_ok_docs']} docs re-tokenize to the recorded length). "
                  f"Full-depth argmax = the text token at " + ", ".join(f"{m} {short(R[m]['full_tok_acc'])}" for m in R) + ".", ""]
        lines += ["### P1 — agreement with full depth (and NLL of x_{p+1}): late entry L_t vs the t-pass model", "",
                  "| model | L_1 | 1-pass | L_2 | 2-pass | L_3 | 3-pass | full NLL | L_1 NLL | 1-pass NLL | L_2 NLL | 2-pass NLL |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m, r in R.items():
            c = r["configs"]
            lines.append(f"| {m} | {fmt(c['cell_4_1']['agree'])} | {fmt(c['ar1']['agree'])} | {fmt(c['cell_4_2']['agree'])} | {fmt(c['ar2']['agree'])} | "
                         f"{fmt(c['cell_4_3']['agree'])} | {fmt(c['ar3']['agree'])} | {short(c['full']['nll'])} | {short(c['cell_4_1']['nll'])} | "
                         f"{short(c['ar1']['nll'])} | {short(c['cell_4_2']['nll'])} | {short(c['ar2']['nll'])} |")
        lines += ["", "Paired differences L_t − t-pass (agreement, NLL):", ""]
        for m, r in R.items():
            lines.append(f"- {m}: " + "; ".join(f"t={t}: Δagree {fmt(r['P1'][f't{t}']['d_agree'])}, ΔNLL {fmt(r['P1'][f't{t}']['d_nll'])}" for t in (1, 2, 3)))
        lines += ["", "Agreement binned by the full-depth top-1 probability at p (fraction of positions in the bin; L_1 / 1-pass / L_2 / 2-pass):", ""]
        for m, r in R.items():
            lines.append(f"- {m}: " + "; ".join(f"p∈[{k}) {short(v['frac'], 2)}: {short(v['cell_4_1'])}/{short(v['ar1'])}/{short(v['cell_4_2'])}/{short(v['ar2'])}"
                                                for k, v in r["P1_bins"].items()))
        lines += ["", "### D × t grid (context run to D passes, slot makes t passes at D-t+1..D): agreement with full depth / NLL", ""]
        for m, r in R.items():
            c = r["configs"]; lines.append(f"**{m}**  " + " · ".join(f"({D},{t}) {short(c[f'cell_{D}_{t}']['agree'])}/{c[f'cell_{D}_{t}']['nll']['point']:.2f}"
                                                                for D in range(1, T + 1) for t in range(1, D + 1)) + "  ")
        lines += ["", "### P2 — representation of L_1's final state", "",
                  "| model | cos(L1,P1) | cos(L1,P2) | cos(L1,P3) | cos(L1,P4) | cos(P1,P4) | cos(P3,P4) | CKA L1~P1 | CKA L1~P4 | CKA P1~P4 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for m, r in R.items():
            cs = r["cos"]; ck = (r.get("cka") or {}).get("pairs", {})
            g = lambda k: fmt(ck[k]) if k in ck else "n/a"
            lines.append(f"| {m} | {fmt(cs['L1~P1'])} | {fmt(cs['L1~P2'])} | {fmt(cs['L1~P3'])} | {fmt(cs['L1~P4'])} | {fmt(cs['P1~P4'])} | {fmt(cs['P3~P4'])} | {g('L1~P1')} | {g('L1~P4')} | {g('P1~P4')} |")
        lines += ["", "Logit lens (lm_head(norm(residual)) after each of the 48 layers of ONE pass; agreement with the full-depth argmax). "
                  "`settle` = first layer from which the lens agrees through layer 48 (positions whose final readout agrees):", "",
                  "| model | query | agree @12 | @24 | @36 | @44 | @48 | settle mean | settle median [p10,p90] |", "|---|---|---|---|---|---|---|---|---|"]
        for m, r in R.items():
            for q, qn in (("L1", "L_1 (deep ctx)"), ("pass1", "normal pass 1"), ("passT", "normal pass 4")):
                lz = r["lens"][q]; cv = lz["curve"]["point"]
                lines.append(f"| {m} | {qn} | {cv[11]:.3f} | {cv[23]:.3f} | {cv[35]:.3f} | {cv[43]:.3f} | {cv[47]:.3f} | {fmt(lz['settle_mean'], 1)} | "
                             f"{lz['settle_median']:.0f} [{lz['settle_q'][0]:.0f}, {lz['settle_q'][4]:.0f}] |" if lz["settle_q"] else f"| {m} | {qn} | - |")
        lines += ["", "### P3 — keys vs values (L_1 reading deep/pass-1 K and V): agreement / NLL", "", "| model | K deep + V deep (=L_1) | K deep + V pass-1 | K pass-1 + V deep | both pass-1 (=1-pass) |", "|---|---|---|---|---|"]
        for m, r in R.items():
            c = r["configs"]; f2 = lambda k: f"{fmt(c[k]['agree'])} / {c[k]['nll']['point']:.3f}"
            lines.append(f"| {m} | {f2('cell_4_1')} | {f2('p3_KdVs')} | {f2('p3_KsVd')} | {f2('cell_1_1')} |")
        Ms = [1, 2, 4, 8, 16, 64]
        lines += ["", "### P4 — which context positions need depth (L_1, deep K/V for a subset, pass-1 K/V elsewhere): agreement", "",
                  "| model | variant | m=0 | 1 | 2 | 4 | 8 | 16 | 64 | all |", "|---|---|---|---|---|---|---|---|---|---|"]
        for m, r in R.items():
            c = r["configs"]
            lines.append(f"| {m} | deep for the LAST m | {short(c['cell_1_1']['agree'])} | " + " | ".join(short(c[f'p4_last_{x}']['agree']) for x in Ms) + f" | {short(c['cell_4_1']['agree'])} |")
            lines.append(f"| {m} | deep for OLDER than m | {short(c['cell_4_1']['agree'])} | " + " | ".join(short(c[f'p4_old_{x}']['agree']) for x in Ms) + f" | {short(c['cell_1_1']['agree'])} |")
        lines += ["", "Partition variants (deep K/V only for …): " + "; ".join(
            f"{m}: prompt {short(r['configs']['p4_prompt']['agree'])}, generated part {short(r['configs']['p4_gen']['agree'])}, position 0 only {short(r['configs']['p4_sink']['agree'])}, all but position 0 {short(r['configs']['p4_nosink']['agree'])}"
            for m, r in R.items()), ""]
        lines += ["### P5 — attention of the query at p (head-averaged, mean over the 48 layers)", "",
                  "| model | query | entropy (nats) | mass on p-7..p | mass on pos 0 | mass on itself |", "|---|---|---|---|---|---|"]
        for m, r in R.items():
            for q, qn in (("L1", "L_1 (deep ctx)"), ("pass1", "normal pass 1"), ("passT", "normal pass 4")):
                a = r["attn_mean"][q]; lines.append(f"| {m} | {qn} | {fmt(a['entropy'], 2)} | {fmt(a['last8'])} | {fmt(a['sink'])} | {fmt(a['self'])} |")
        lines += ["", "Paired L_1 − normal pass-4 query (layer mean): " + "; ".join(
            f"{m}: entropy {fmt(r['attn_d_L1_minus']['passT']['entropy'], 2)}, last8 {fmt(r['attn_d_L1_minus']['passT']['last8'])}, sink {fmt(r['attn_d_L1_minus']['passT']['sink'])}, self {fmt(r['attn_d_L1_minus']['passT']['self'])}"
            for m, r in R.items()), ""]
        lines += ["Runtime exactness checks (max |logit diff|, cell(k,k) vs k-pass forward, fp32): " + "; ".join(f"{m} {max(r['worst_checks'].values()):.1e}" for m, r in R.items() if r.get("worst_checks")), ""]
    if S.get("reading"): lines += ["## Reading", ""] + S["reading"]
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=ROOT); ap.add_argument("--models", nargs="*", default=[m for m, _ in MODELS])
    ap.add_argument("--reading", default=None, help="optional text file with the reading paragraphs appended to the md")
    a = ap.parse_args(); S = dict(written=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), tasks={}, models=dict(MODELS))
    present = set()
    for ti, (task, _) in enumerate(TASKS):
        data = {m: load(a.root, m, task) for m in a.models}; data = {m: v for m, v in data.items() if v is not None}
        if not data: continue
        ids = [tuple(d["doc_id"] for d in v["docs"]) for v in data.values()]
        common = sorted(set.intersection(*[set(x) for x in ids])); bt = Boot(len(common), seed=1000 + ti)
        R = {}
        for m, v in data.items():
            by = {d["doc_id"]: d for d in v["docs"]}; docs = [by[i] for i in common]; R[m] = analyze_model(docs, v["meta"], bt); present.add(m)
        # cross-model paired differences at the same positions
        X = {}
        if "v2" in data:
            n = np.array([d["nq"] for d in [{x["doc_id"]: x for x in data["v2"]["docs"]}[i] for i in common]], np.float64)
            def hits(m, c):
                by = {d["doc_id"]: d for d in data[m]["docs"]}
                return np.array([(by[i]["res"][c]["am"] == by[i]["res"]["full"]["am"]).sum() for i in common], np.float64)
            for m in data:
                if m == "v2": continue
                X[f"v2-{m}"] = {c: bt.diff(hits("v2", c), hits(m, c), n) for c in ("cell_4_1", "cell_4_2", "ar1", "ar2")}
        S["tasks"][task] = R; S.setdefault("cross", {})[task] = X; S.setdefault("common_docs", {})[task] = len(common)
        print(f"[analyze] {task}: models {list(data)} on {len(common)} common docs", flush=True)
    if a.reading and os.path.exists(a.reading): S["reading"] = open(a.reading).read().strip().split("\n")
    json.dump(S, open(f"{a.root}/mech_summary.json", "w"), indent=1)
    write_md(S, f"{a.root}/mech_summary.md", present)
    print(f"[analyze] wrote {a.root}/mech_summary.{{json,md}}", flush=True)


if __name__ == "__main__":
    main()
