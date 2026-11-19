"""entry_analyze.py -- tables + bootstrap CIs for the entry-signal probe (10-27). CPU only (numpy).

Input : <root>/<model>/raw_<task>.pkl (entry_run.py), meta_<task>.json
Output: <root>/entry_summary.json, <root>/entry_summary.md (the "Answer" section is read from <root>/entry_answer.md if present)

Metrics per (model, task, entry), pooled over slot positions, 95 % CIs = percentile bootstrap over DOCUMENTS (B = 2000, seed 0):
  agree    argmax of the late slot == argmax of the full-depth gate mixture at q (what the model actually predicts)
  agree4   argmax == the full forward's pass-T readout at q (ceiling of any pass-T-only readout; ORA = 1 by construction)
  nll      NLL of x_{q+1} under the late slot's pass-T readout (nats/token)
  cos_out  cosine(late slot's normed pass-T output state, true pass-T output state)
  cos_in   cosine(entry state, true pass-T input) (input-space distance; ORA = 1)
  closed   share of the E -> ORA agreement gap closed: (agree_X - agree_E) / (agree_ORA - agree_E)
Splits (soft-embedding diagnostics): CC = predecessor argmax == x_q and p(x_q) >= .9 ("confident-correct"), UC = argmax right but p < .9,
W = predecessor argmax wrong; NC = UC + W.
    python entry_analyze.py [--root DIR] [--models v2,sft,stock26] [--tasks gsm8k_cot_zeroshot,ifeval] [--boot 2000]
"""
import argparse, json, os, pickle, time

import numpy as np

ROOT = "probes/entry/results"
MECH = "probes/mechanism/results"
ENTRIES = [("E", "tok"), ("NE", "tok"), ("LE", "tok"), ("TM", "tok"), ("OE", "tok"), ("LOE", "tok"),
           ("SE", "soft"), ("SE10", "soft"), ("NSE", "soft"), ("LSE", "soft"), ("OSE", "soft"), ("OSE10", "soft"),
           ("MU", "ctx"), ("H", "ctx"), ("EH", "ctx"), ("NEH", "ctx"), ("SEH", "ctx"), ("OEH", "ctx"), ("LEH", "ctx"),
           ("ORA", "ora")]
NAMES = [n for n, _ in ENTRIES]
EXTRA = {"TM", "MU", "NEH"}
DESC = {"E": "embedding E[x_q] (current late entry)", "NE": "self.norm(E) rescaled to c", "LE": "ridge map W E + b",
        "TM": "per-token mean of the true pass-4 input (fit set)", "OE": "output row W_out[x_q] rescaled to c (tied, zero-shot)",
        "LOE": "ridge map of W_out[x_q]", "SE": "soft embedding sum_v p(v) E[v]", "SE10": "soft embedding, top-10",
        "NSE": "self.norm(SE) rescaled to c", "LSE": "ridge map of SE", "OSE": "soft output embedding sum_v p(v) W_out[v], rescaled",
        "OSE10": "soft output embedding, top-10, rescaled", "MU": "constant: mean true pass-4 input", "H": "predecessor's normed pass-4 state",
        "EH": "E + H", "NEH": "NE + H", "SEH": "SE + H", "OEH": "OE + H", "LEH": "ridge map of [E; H]", "ORA": "true pass-4 input (oracle)"}
MODELS = ["v2", "sft", "stock26"]
TASKS = ["gsm8k_cot_zeroshot", "ifeval"]
TNAME = {"gsm8k_cot_zeroshot": "GSM8K", "ifeval": "IFEval"}
REFS = ["p4", "ar1", "ar2", "ar3"]


def load(root, model, task):
    p = f"{root}/{model}/raw_{task}.pkl"
    if not os.path.exists(p): return None
    with open(p, "rb") as f: return pickle.load(f)


def per_doc(docs, mask_fn=None):
    """per-doc sums. Returns dict metric -> {name: [G] sums}, n [G] counts"""
    G = len(docs); n = np.zeros(G); S = {m: {e: np.zeros(G) for e in NAMES + REFS} for m in ("agree", "agree4", "nll", "cos_out", "cos_in", "copy")}
    for i, d in enumerate(docs):
        msk = np.ones(d["nq"], bool) if mask_fn is None else mask_fn(d)
        n[i] = msk.sum()
        if not n[i]: continue
        full = d["ref"]["full"]["am"][msk]; p4 = d["ref"]["p4"]["am"][msk]; xq = d["x"][msk]
        for e in NAMES:
            r = d["ent"][e]; am = r["am"][msk]
            S["agree"][e][i] = (am == full).sum(); S["agree4"][e][i] = (am == p4).sum(); S["copy"][e][i] = (am == xq).sum(); S["nll"][e][i] = r["nll"][msk].astype(np.float64).sum()
            S["cos_out"][e][i] = r["cos_out"][msk].astype(np.float64).sum(); S["cos_in"][e][i] = r["cos_in"][msk].astype(np.float64).sum()
        for k in REFS:
            r = d["ref"][k]; S["agree"][k][i] = (r["am"][msk] == full).sum(); S["agree4"][k][i] = (r["am"][msk] == p4).sum(); S["copy"][k][i] = (r["am"][msk] == xq).sum()
            S["nll"][k][i] = r["nll"][msk].astype(np.float64).sum()
    return S, n


def boot_weights(G, B, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(B)]).astype(np.float64)   # [B, G]


def stat(sums, n, Wb):
    pt = sums.sum() / max(n.sum(), 1); bs = (Wb @ sums) / np.maximum(Wb @ n, 1)
    return dict(point=float(pt), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5))), bs


def analyse(docs, Wb, mask_fn=None, closure=True):
    S, n = per_doc(docs, mask_fn); out = {"n": int(n.sum())}; bsa = {}; bsn = {}
    for e in NAMES + REFS:
        o = {}
        for m in ("agree", "agree4", "nll", "cos_out", "cos_in", "copy"):
            if e in REFS and m in ("cos_out", "cos_in"): continue
            o[m], bs = stat(S[m][e], n, Wb)
            if m == "agree": bsa[e] = bs
            if m == "nll": bsn[e] = bs
        out[e] = o
    if closure and n.sum():
        aE, aO = out["E"]["agree"]["point"], out["ORA"]["agree"]["point"]
        for e in NAMES:
            den = aO - aE; dbs = bsa["ORA"] - bsa["E"]
            pt = (out[e]["agree"]["point"] - aE) / den if abs(den) > 1e-9 else float("nan")
            bs = (bsa[e] - bsa["E"]) / np.where(np.abs(dbs) > 1e-9, dbs, np.nan)
            out[e]["closed"] = dict(point=float(pt), lo=float(np.nanpercentile(bs, 2.5)), hi=float(np.nanpercentile(bs, 97.5)))
            d = bsa[e] - bsa["E"]
            out[e]["d_agree_vs_E"] = dict(point=float(out[e]["agree"]["point"] - aE), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
            nE, nO = out["E"]["nll"]["point"], out["ORA"]["nll"]["point"]; dn = bsn["E"] - bsn["ORA"]
            bs = (bsn["E"] - bsn[e]) / np.where(np.abs(dn) > 1e-9, dn, np.nan)
            out[e]["closed_nll"] = dict(point=float((nE - out[e]["nll"]["point"]) / (nE - nO)) if abs(nE - nO) > 1e-9 else float("nan"),
                                        lo=float(np.nanpercentile(bs, 2.5)), hi=float(np.nanpercentile(bs, 97.5)))
    return out


def soft_stats(docs, Wb):
    pk = np.concatenate([d["pred_ok"] for d in docs]); px = np.concatenate([d["pxq"] for d in docs]); sc = np.concatenate([d["se_e_cos"] for d in docs])
    cc = pk & (px >= 0.9)
    G = len(docs); n = np.array([d["nq"] for d in docs], float)
    def b(fn):
        s = np.array([fn(d).sum() for d in docs], float); return stat(s, n, Wb)[0]
    return dict(pred_ok=b(lambda d: d["pred_ok"]), pxq_mean=b(lambda d: d["pxq"].astype(np.float64)), se_e_cos_mean=b(lambda d: d["se_e_cos"].astype(np.float64)),
                share_cc=b(lambda d: d["pred_ok"] & (d["pxq"] >= 0.9)), share_wrong=b(lambda d: ~d["pred_ok"]),
                pxq_median=float(np.median(px)), se_e_cos_median=float(np.median(sc)), se_e_cos_cc=float(sc[cc].mean()) if cc.any() else None,
                se_e_cos_nc=float(sc[~cc].mean()) if (~cc).any() else None, tm_seen=float(np.concatenate([d["tm_seen"] for d in docs]).mean()),
                oe_cos_posweighted=float(np.concatenate([d["oe_cos"] for d in docs]).mean()), n=int(n.sum()), G=G)


def mech_crosscheck(docs, model, task):
    """E entry vs the 10-23 mech probe's cell_4_1 (same code path, v2 text for every model -> only v2 shares the sequences)"""
    p = f"{MECH}/{model}/raw_{task}.pkl"
    if model != "v2" or not os.path.exists(p): return None
    with open(p, "rb") as f: M = pickle.load(f)
    md = {d["doc_id"]: d for d in M["docs"]}; same = tot = docs_ok = 0; ag_e = ag_m = 0
    for d in docs:
        m = md.get(d["doc_id"])
        if m is None or m["N"] != d["N"] or m["P0"] != d["P0"]: continue
        # mech rows S = P0..N-2 == ours
        a, b = d["ent"]["E"]["am"], m["res"]["cell_4_1"]["am"]
        if len(a) != len(b): continue
        same += int((a == b).sum()); tot += len(a); docs_ok += 1
        ag_e += int((a == d["ref"]["full"]["am"]).sum()); ag_m += int((b == m["res"]["full"]["am"]).sum())
    return dict(docs=docs_ok, positions=tot, argmax_identical=same / max(tot, 1), agree_ours=ag_e / max(tot, 1), agree_mech=ag_m / max(tot, 1))


def fmt(s, d=3):
    if s is None: return "-"
    f = f"{{:.{d}f}}"
    return f"{f.format(s['point'])} [{f.format(s['lo'])}, {f.format(s['hi'])}]"


def fmt_s(s, d=3):
    return "-" if s is None else f"{s['point']:.{d}f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=ROOT); ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--tasks", default=",".join(TASKS)); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(); models = a.models.split(","); tasks = a.tasks.split(",")
    J = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), root=a.root, boot=a.boot, entries=ENTRIES, extra=sorted(EXTRA), desc=DESC, results={}, meta={})
    for mo in models:
        for t in tasks:
            R = load(a.root, mo, t)
            if R is None: print(f"missing {mo} {t}"); continue
            docs = R["docs"]; meta = R["meta"]; G = len(docs); Wb = boot_weights(G, a.boot)
            key = f"{mo}/{t}"; print(f"[analyse] {key}: {G} docs, {sum(d['nq'] for d in docs)} positions", flush=True)
            res = dict(all=analyse(docs, Wb))
            nn = np.array([d["nq"] for d in docs], float)
            res["all"]["full_copy"] = stat(np.array([(d["ref"]["full"]["am"] == d["x"]).sum() for d in docs], float), nn, Wb)[0]
            cc = lambda d: d["pred_ok"] & (d["pxq"] >= 0.9)
            res["split"] = dict(CC=analyse(docs, Wb, cc), NC=analyse(docs, Wb, lambda d: ~cc(d)),
                                UC=analyse(docs, Wb, lambda d: d["pred_ok"] & (d["pxq"] < 0.9), closure=False),
                                W=analyse(docs, Wb, lambda d: ~d["pred_ok"], closure=False))
            res["soft"] = soft_stats(docs, Wb); res["mech_crosscheck"] = mech_crosscheck(docs, mo, t)
            res["rt_ok_docs"] = int(sum(bool(d["rt_ok"]) for d in docs)); res["docs"] = G
            J["results"][key] = res
            J["meta"][key] = {k: meta.get(k) for k in ("model", "T", "L", "nv", "vocab_rows", "tie_word_embeddings", "lm_head_is_embed", "cos_Wout_E_centered",
                                                        "fit", "worst_checks", "missing", "docs", "positions", "gpu", "host", "torch", "sec", "written")}
    # fit-domain check: the same probe with the maps / token table / c fit on other data (job 2: fit_ifeval, fit_pooled)
    J["fitdomain"], J["fitdomain_meta"] = {}, {}
    for vname, vdir in [("gsm8k", a.root)] + [(v, f"{a.root}/fit_{v}") for v in ("ifeval", "pooled")]:
        for mo in models:
            for t in tasks:
                Rv = J["results"].get(f"{mo}/{t}") if vname == "gsm8k" else None
                if Rv is None:
                    D = load(vdir, mo, t)
                    if D is None: continue
                    Rv = dict(all=analyse(D["docs"], boot_weights(len(D["docs"]), a.boot)))
                    J["fitdomain_meta"].setdefault(vname, {})[mo] = {k: D["meta"]["fit"].get(k) for k in ("task", "specs", "scale_c", "LE", "LSE", "LOE", "LEH", "TM", "MU_val_r2", "positions", "docs")}
                    print(f"[analyse] fit {vname}: {mo}/{t}", flush=True)
                elif mo in [k.split("/")[0] for k in J["meta"]]:
                    J["fitdomain_meta"].setdefault(vname, {})[mo] = {k: J["meta"][f"{mo}/{t}"]["fit"].get(k) for k in ("task", "scale_c", "LE", "LSE", "LOE", "LEH", "TM", "MU_val_r2", "positions", "docs")}
                J["fitdomain"].setdefault(vname, {})[f"{mo}/{t}"] = {e: {m: Rv["all"][e].get(m) for m in ("agree", "closed", "nll", "closed_nll")} for e in NAMES}
    json.dump(J, open(f"{a.root}/entry_summary.json.tmp", "w"), indent=1, default=float); os.replace(f"{a.root}/entry_summary.json.tmp", f"{a.root}/entry_summary.json")
    write_md(J, a.root, models, tasks)
    print(f"wrote {a.root}/entry_summary.{{json,md}}")


def write_md(J, root, models, tasks):
    R, M = J["results"], J["meta"]; L = []
    L.append("# Entry-signal probe: why does a token entering at the last pass fail without training?\n")
    L.append(f"_written {J['written']} by entry_analyze.py; raw = {root}/<model>/raw_<task>.pkl; code = MARS/integration/probes/entry_probe/_\n")
    ans = f"{root}/entry_answer.md"
    if os.path.exists(ans): L.append(open(ans).read().rstrip() + "\n")
    L.append("## Setup\n")
    L.append("Ouro-2.6B, T = 4 passes of one 48-layer stack; teacher forcing on prompt + the model's OWN full-depth (ar:4) generation "
             "(campaign shards, campaign chat template). For every generated position q (token x_q, successor x_{q+1} also generated) a "
             "1-pass LATE slot runs at pass index 4 and reads the IDEAL deep context: the true pass-4 keys/values of all positions < q from "
             "the normal full-depth forward, plus its own key. Only its entry state (its input to pass 4) changes between rows below. "
             "Readout = its pass-4 logits (the gate mixture renormalised to the one pass it ran: weight 1). fp32, TF32 off. Linear maps "
             "(ridge, standardised features, intercept unpenalised, lambda chosen on held-out docs 550-599, refit on 300-599) are fit on "
             "GSM8K docs 300-599 to predict the TRUE pass-4 input of the same position; the model is frozen. c = median norm of real "
             "pass-4 inputs on the fit set. Verified in modeling_ouro.py: each pass ends with `hidden = self.norm(hidden)`; that normed "
             "state is the next pass's input AND what lm_head / the exit gate read (no further norm); tie_word_embeddings = False, so "
             "lm_head rows W_out[v] live in the loop-input space while embedding rows E[v] do not.\n")
    L.append("Metrics: **agree** = argmax equals the full-depth (gate-mixture) prediction at q; **agree4** = equals the full forward's "
             "pass-4 readout (ORA = 1 by construction); **NLL** of x_{q+1} (nats); **cos_out** = cosine of the slot's normed pass-4 output "
             "with the true pass-4 output state; **cos_in** = cosine of the entry with the true pass-4 input. 95 % CIs: bootstrap over "
             f"documents (B = {J['boot']}). **closed** = share of the E -> ORA agreement gap closed. Entries marked * are controls added beyond the "
             "brief.\n")
    L.append("| entry | definition |\n|---|---|")
    for n, fam in ENTRIES: L.append(f"| {n}{'*' if n in EXTRA else ''} | {DESC[n]} ({fam}) |")
    L.append("")
    FD = J.get("fitdomain", {})
    L.append("## Headline: share of the E -> ORA gap each entry closes (in-domain fits)\n")
    L.append("Cell = agreement with full depth, then (share of the agreement gap closed / share of the NLL gap closed). Fitted entries use "
             "the in-domain fit: GSM8K maps for GSM8K, IFEval maps for IFEval (when job 2's fit_ifeval run exists; otherwise GSM8K maps). "
             "Negative = worse than E.\n")
    sel = ["E", "NE", "OE", "MU", "LE", "TM", "LOE", "SE", "LSE", "H", "NEH", "LEH", "ORA"]
    for t in tasks:
        v = "ifeval" if (t == "ifeval" and "ifeval" in FD) else "gsm8k"
        ks = [f"{mo}/{t}" for mo in models if f"{mo}/{t}" in FD.get(v, {})]
        if not ks: continue
        L.append(f"**{TNAME.get(t, t)}** (fit on {'IFEval 150-449' if v == 'ifeval' else 'GSM8K 300-599'})\n")
        L.append("| entry | " + " | ".join(k.split("/")[0] for k in ks) + " |"); L.append("|---|" + "---|" * len(ks))
        for e in sel:
            cells = []
            for k in ks:
                r = FD[v][k][e]; c1, c2 = r.get("closed"), r.get("closed_nll")
                cells.append(f"{r['agree']['point']:.3f} ({c1['point']:+.2f} / {c2['point']:+.2f})" if (c1 and c2) else f"{r['agree']['point']:.3f}")
            L.append(f"| {e}{'*' if e in EXTRA else ''} | " + " | ".join(cells) + " |")
        L.append("")
    for t in tasks:
        ks = [f"{mo}/{t}" for mo in models if f"{mo}/{t}" in R]
        if not ks: continue
        L.append(f"## {TNAME.get(t, t)}: main table (agreement with full depth, NLL, output cosine; fitted entries: GSM8K 300-599 fit)\n")
        hdr = "| entry | " + " | ".join(f"{k.split('/')[0]} agree [95% CI] | NLL | cos_out | closed" for k in ks) + " |"
        L.append(hdr); L.append("|---|" + "---|---|---|---|" * len(ks))
        for n, fam in ENTRIES:
            row = [f"{n}{'*' if n in EXTRA else ''}"]
            for k in ks:
                r = R[k]["all"][n]; row += [fmt(r["agree"]), fmt_s(r["nll"], 2), fmt_s(r["cos_out"], 3), fmt_s(r.get("closed"), 2)]
            L.append("| " + " | ".join(row) + " |")
        for ref, lab in (("p4", "ref: full forward's pass-4 readout"), ("ar1", "ref: 1-pass model"), ("ar2", "ref: 2-pass model"), ("ar3", "ref: 3-pass model")):
            row = [lab]
            for k in ks:
                r = R[k]["all"][ref]; row += [fmt(r["agree"]), fmt_s(r["nll"], 2), "", ""]
            L.append("| " + " | ".join(row) + " |")
        row = ["ref: full depth (mixture)"]
        for k in ks:
            docs_nll = None
            row += ["1", "", "", ""]
        L.append("| " + " | ".join(row) + " |")
        L.append("\npositions / docs: " + "; ".join(f"{k.split('/')[0]} {R[k]['all']['n']} / {R[k]['docs']}" for k in ks) + "\n")
        L.append(f"### {TNAME.get(t, t)}: agree4 (vs pass-4 readout), input-space cosine, copy rate (slot argmax == its own token x_q)\n")
        L.append("| entry | " + " | ".join(f"{k.split('/')[0]} agree4 | cos_in | copy" for k in ks) + " |"); L.append("|---|" + "---|---|---|" * len(ks))
        for n, fam in ENTRIES:
            row = [n]
            for k in ks: r = R[k]["all"][n]; row += [fmt(r["agree4"]), fmt_s(r["cos_in"], 3), fmt_s(r["copy"], 3)]
            L.append("| " + " | ".join(row) + " |")
        L.append("| ref: full depth | " + " | ".join(f"1 | | {fmt_s(R[k]['all']['full_copy'], 3)}" for k in ks) + " |")
        L.append("")
        L.append(f"### {TNAME.get(t, t)}: soft embedding vs E (the predecessor's distribution)\n")
        L.append("| model | predecessor argmax = x_q | mean p(x_q) | median p(x_q) | share CC (argmax ok & p >= .9) | share argmax wrong | mean cos(SE, E) (CC / rest) |")
        L.append("|---|---|---|---|---|---|---|")
        for k in ks:
            s = R[k]["soft"]
            L.append(f"| {k.split('/')[0]} | {fmt(s['pred_ok'])} | {fmt(s['pxq_mean'])} | {s['pxq_median']:.3f} | {fmt(s['share_cc'])} | {fmt(s['share_wrong'])} | "
                     f"{s['se_e_cos_mean']['point']:.3f} ({(s['se_e_cos_cc'] or float('nan')):.3f} / {(s['se_e_cos_nc'] or float('nan')):.3f}) |")
        L.append("")
        L.append(f"### {TNAME.get(t, t)}: agreement split by the predecessor (CC = confident-correct; NC = the rest; UC / W = argmax right but p < .9 / argmax wrong)\n")
        sel = ["E", "NE", "LE", "OE", "SE", "SE10", "NSE", "LSE", "OSE", "OSE10", "H", "SEH", "OEH", "LEH", "ORA"]
        L.append("| entry | " + " | ".join(f"{k.split('/')[0]} CC | NC | UC | W" for k in ks) + " |"); L.append("|---|" + "---|---|---|---|" * len(ks))
        for n in sel:
            row = [n]
            for k in ks:
                sp = R[k]["split"]; row += [fmt_s(sp[s][n]["agree"], 3) if sp[s]["n"] else "-" for s in ("CC", "NC", "UC", "W")]
            L.append("| " + " | ".join(row) + " |")
        L.append("| n positions | " + " | ".join(" | ".join(str(R[k]["split"][s]["n"]) for s in ("CC", "NC", "UC", "W")) for k in ks) + " |")
        L.append("| ref: full forward pass-4 readout | " + " | ".join(" | ".join(fmt_s(R[k]["split"][s]["p4"]["agree"], 3) if R[k]["split"][s]["n"] else "-" for s in ("CC", "NC", "UC", "W")) for k in ks) + " |")
        L.append("")
    # fit / tied / checks
    FD = J.get("fitdomain", {})
    if len(FD) > 1:
        L.append("## Fit-domain check: where the fitted entries (LE, TM, LOE, LSE, LEH, MU) were fit\n")
        L.append("Job 1 fit the maps on GSM8K docs 300-599 only. Job 2 refit them on IFEval docs 150-449 and on both pooled; everything "
                 "else is identical (same eval docs, no overlap with any fit doc). Cells: agreement with full depth [95% CI] (closed share of the "
                 "E -> ORA gap). E, H, SE, OE and ORA do not depend on the fit (NE/OE/OSE only through the scale c).\n")
        vs = [v for v in ("gsm8k", "ifeval", "pooled") if v in FD]
        for t in tasks:
            L.append(f"### {TNAME.get(t, t)}\n")
            L.append("| model | entry | " + " | ".join(f"fit on {v}" for v in vs) + " |"); L.append("|---|---|" + "---|" * len(vs))
            for mo in models:
                k = f"{mo}/{t}"
                if not all(k in FD[v] for v in vs): continue
                for e in ("E", "MU", "LE", "TM", "LOE", "LSE", "LEH", "ORA"):
                    cells = []
                    for v in vs:
                        r = FD[v][k][e]; cl = r.get("closed")
                        cells.append(fmt(r["agree"]) + ("" if (cl is None or e in ("E", "ORA")) else f" ({cl['point']:.2f})"))
                    L.append(f"| {mo} | {e} | " + " | ".join(cells) + " |")
            L.append("")
        L.append("| model | fit | c | held-out R^2 LE / TM / LOE / LSE / LEH / MU | fit positions |")
        L.append("|---|---|---|---|---|")
        for v in vs:
            for mo, f in J.get("fitdomain_meta", {}).get(v, {}).items():
                if not f or f.get("LE") is None: continue
                L.append(f"| {mo} | {v} ({f.get('task')}) | {f['scale_c']:.2f} | {f['LE']['val_r2']:.3f} / {f['TM']['val_r2']:.3f} / {f['LOE']['val_r2']:.3f} / "
                         f"{f['LSE']['val_r2']:.3f} / {f['LEH']['val_r2']:.3f} / {f.get('MU_val_r2', float('nan')):.3f} | {f.get('positions')} |")
        L.append("")
    L.append("## Fit set (GSM8K docs 300-599), scales, tied embeddings, checks\n")
    L.append("| model | c = median norm of true pass-4 input | median norm E / self.norm(E) / W_out[x] / H / SE | ridge lambda / held-out R^2: LE, LSE, LOE, LEH | TM held-out R^2 (seen) | MU held-out R^2 | cos(W_out[v], E[v]) centered: vocab mean / median / p90 ; position-weighted |")
    L.append("|---|---|---|---|---|---|---|")
    for mo in models:
        k = next((f"{mo}/{t}" for t in tasks if f"{mo}/{t}" in M), None)
        if k is None: continue
        f = M[k]["fit"]; mn = f["median_norm"]; cw = M[k]["cos_Wout_E_centered"]
        rr = ", ".join(f"{n} {f[n]['lam']:g} / {f[n]['val_r2']:.3f}" for n in ("LE", "LSE", "LOE", "LEH"))
        pw = R[k]["soft"]["oe_cos_posweighted"] if k in R else float("nan")
        L.append(f"| {mo} | {f['scale_c']:.2f} | {mn['E']:.2f} / {mn['nE']:.1f} / {mn['O']:.2f} / {mn['H']:.1f} / {mn['SE']:.2f} | {rr} | {f['TM']['val_r2']:.3f} ({f['TM']['val_seen']:.3f}) | "
                 f"{f.get('MU_val_r2', float('nan')):.3f} | {cw['mean']:.3f} / {cw['median']:.3f} / {cw['p90']:.3f} ; {pw:.3f} |")
    L.append("")
    L.append("Runtime checks (max over docs; fp32 on GPU): ORA logits vs the full forward's pass-4 readout, ORA output state vs the true pass-4 "
             "state, and (first doc of each task) E vs mech_core cell(4,1) logits.\n")
    L.append("| model / task | docs | positions | ORA logits | ORA state | E vs mech cell(4,1) | cross-check vs 10-23 mech raw (v2 text): docs, argmax identical, agree ours / mech |")
    L.append("|---|---|---|---|---|---|---|")
    for k in R:
        w = M[k]["worst_checks"] or {}; mx = R[k]["mech_crosscheck"]
        mxs = "-" if not mx else f"{mx['docs']} docs, {mx['argmax_identical']:.4f}, {mx['agree_ours']:.4f} / {mx['agree_mech']:.4f}"
        L.append(f"| {k} | {R[k]['docs']} | {R[k]['all']['n']} | {w.get('ORA_logits_vs_passT', float('nan')):.2e} | {w.get('ORA_state_vs_passT', float('nan')):.2e} | "
                 f"{w.get('E_vs_mech_cell_T_1', float('nan')):.2e} | {mxs} |")
    L.append("")
    open(f"{root}/entry_summary.md.tmp", "w").write("\n".join(L) + "\n"); os.replace(f"{root}/entry_summary.md.tmp", f"{root}/entry_summary.md")


if __name__ == "__main__":
    main()
