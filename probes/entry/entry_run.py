"""entry_run.py -- GPU driver of the entry-signal probe (10-27). One or more models per invocation (sequential), fp32 / TF32 off.

Per model:
  sequences  prompt (the campaign's materialized chat-template context, lmeval_ouro/<tag>/_tasks/<task>.jsonl, add_special_tokens=False)
             + the model's OWN full-depth (ar:4) generation (lmeval_ouro/<tag>/<task>/ar_4/shard_*.jsonl), re-tokenized with the model's
             tokenizer exactly as mech_run.load_docs (rt_ok flags docs whose re-tokenization changes the recorded token count)
  FIT        GSM8K docs 300-599 (train 300-549, lambda selection on 550-599, then refit on all): ridge maps LE / LSE / LOE / LEH to the
             true pass-T input, per-token mean table TM, constant MU, scale c = median ||true pass-T input||. Model frozen.
  EVAL       GSM8K docs 0-299 (+ IFEval 0-149): per slot position q (x_q generated, x_{q+1} generated): every entry of entry_core.ENTRIES
             through one late pass at pass index T reading the ideal pass-T context K/V; per entry: argmax, NLL of x_{q+1}, cosine of
             the output state with the true pass-T output state, cosine of the entry with the true pass-T input. Refs: full-depth
             mixture / pass-T readout / k-pass models (argmax + NLL), predecessor stats (argmax ok, p(x_q)), cos(W_out[x], E[x]).
Runtime checks per doc: ORA logits vs the full forward's pass-T readout and ORA output vs the true pass-T state (max |diff|); on the
first doc of each task: E vs mech_core cell(T, 1) (max |logit diff|).

    python entry_run.py --models "v2=/tmp/m/v2 sft=/tmp/m/sft" --out <dir> [--tasks "gsm8k_cot_zeroshot:0:300 ifeval:0:150"]
                        [--fit gsm8k_cot_zeroshot:300:600:550] [--limit n] [--fit-limit n]
"""
import argparse, json, os, pickle, sys, time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import entry_core as ec  # noqa: E402  (puts mech_probes on sys.path)
import mech_core as mc  # noqa: E402
import mech_run as mr  # noqa: E402

LMO = "eval/data"
NAMES = [n for n, _ in ec.ENTRIES]


def _np(t, dt=np.float32):
    return t.detach().cpu().numpy().astype(dt)


@torch.no_grad()
def fit_phase(pr, docs, val_lo, log=print):
    """docs: list from mech_run.load_docs (fit range). Returns fit dict (GPU tensors) + JSON-able info"""
    dev, T, d, nv = pr.dev, pr.T, pr.m.config.hidden_size, pr.nv
    keys = sorted(set(ec.MAPS.values())); D = {"E": d, "SE": d, "O": d, "EH": 2 * d}
    accs = {k: (ec.RidgeAcc(D[k], d, dev), ec.RidgeAcc(D[k], d, dev)) for k in keys}
    tm_tr, tm_all = ec.TokMean(nv, d, dev), ec.TokMean(nv, d, dev)
    ynorm, norms, val_xEY, t0, npos = [], {k: [] for k in ("E", "nE", "O", "H", "SE", "Y")}, [], time.time(), 0
    for i, dd in enumerate(docs):
        ids = dd["ctx"] + dd["gen"]; P0 = len(dd["ctx"]); N = len(ids)
        if len(dd["gen"]) < 2: continue
        ids_t = torch.tensor(ids, device=dev); Sx = torch.arange(P0 - 1, N - 1, device=dev); x = ids_t[Sx[1:]]
        FF = pr.forward_lean(ids_t, Sx); f = ec.features(pr, x, FF); tr = (not dd["val"]) if "val" in dd else dd["doc_id"] < val_lo
        for k in keys: accs[k][0 if tr else 1].add(ec.map_feature(f, k), f["Y"])
        tm_all.add(x, f["Y"])
        if tr: tm_tr.add(x, f["Y"])
        else: val_xEY.append((x, f["E"], f["Y"]))
        ynorm.append(f["Y"].norm(dim=-1))
        for k, v in (("E", f["E"]), ("nE", pr.mm.norm(f["E"].to(pr.dt)).float()), ("O", f["O"]), ("H", f["H"]), ("SE", f["SE"]), ("Y", f["Y"])):
            norms[k].append(v.norm(dim=-1))
        npos += x.shape[0]; del FF, f
        if i % 50 == 0 or i == len(docs) - 1:
            log(f"[entry] fit {i + 1}/{len(docs)} doc {dd['doc_id']} | {npos} pos | {time.time() - t0:.0f}s")
    maps, info = {}, {}
    for name, key in ec.MAPS.items():
        r = ec.ridge_fit(*accs[key]); maps[name] = (r["W"], r["b"])
        info[name] = dict(feature=key, lam=r["lam"], val_r2=r["val_r2"], all_r2=r["all_r2"], val=r["val"])
        log(f"[entry] ridge {name} ({key}): lambda {r['lam']} val R2 {r['val_r2']:.4f} (fit-all R2 {r['all_r2']:.4f})")
    # token-mean table: held-out R2 with the train-only table (unseen val tokens fall back to the LE map), then the table on all fit docs
    tab, seen = tm_tr.table(); sse = 0.0; nval = 0; nseen = 0
    for x, E, Y in val_xEY:
        pred = torch.where(seen[x.long()][:, None], tab[x.long()], E @ maps["LE"][0] + maps["LE"][1])
        sse += float(((Y - pred) ** 2).sum()); nval += Y.shape[0]; nseen += int(seen[x.long()].sum())
    allv = accs["E"][1]; var = float(allv.syy / allv.n - ((allv.sy / allv.n) ** 2).sum())
    info["TM"] = dict(val_r2=1 - (sse / max(nval, 1)) / var, val_seen=nseen / max(nval, 1))
    tm, tm_seen = tm_all.table(); al = accs["E"][0].merged(accs["E"][1]); mu = (al.sy / al.n).float()
    c = float(torch.cat(ynorm).median())
    info["scale_c"] = c; info["positions"] = npos; info["docs"] = len(docs)
    info["median_norm"] = {k: float(torch.cat(v).median()) for k, v in norms.items()}
    mtr = (accs["E"][0].sy / accs["E"][0].n).double(); vs = accs["E"][1]
    info["MU_val_r2"] = 1 - float(vs.syy / vs.n - 2 * (mtr * vs.sy).sum() / vs.n + (mtr * mtr).sum()) / var
    log(f"[entry] TM val R2 {info['TM']['val_r2']:.4f} (seen {info['TM']['val_seen']:.3f}); c = {c:.3f}; median norms {json.dumps(info['median_norm'])}")
    return dict(c=c, maps=maps, tm=tm, tm_seen=tm_seen, mu=mu), info


@torch.no_grad()
def eval_doc(pr, ids, P0, fit, cosv, names=NAMES, chunk=4096, check_cell=False):
    """one document. Returns a record of numpy arrays (per slot position) + runtime checks"""
    dev, T = pr.dev, pr.T; N = len(ids); ids_t = torch.tensor(ids, device=dev)
    Sx = torch.arange(P0 - 1, N - 1, device=dev); S = Sx[1:]; x = ids_t[S]; tgt = ids_t[S + 1]; nq = S.shape[0]
    FF = pr.forward_lean(ids_t, Sx, kv_pass=T - 1)
    f = ec.features(pr, x, FF); Z = ec.entries(pr, f, x, fit, names)
    lsm = lambda lg: torch.log_softmax(lg.float(), -1)
    refs = {}
    mix = mc.mixture(FF["lg"], FF["g"])[1:]; refs["full"] = mix
    refs["p4"] = FF["lg"][T - 1][1:]
    for k in range(1, T): refs[f"ar{k}"] = mc.mixture(FF["lg"][:k], FF["g"][:k])[1:]
    rec = dict(N=N, P0=P0, nq=nq, x=_np(x, np.int32), tgt=_np(tgt, np.int32), pxq=_np(f["pxq"]), pred_ok=_np(f["pred_ok"], bool),
               se_e_cos=_np(f["se_e_cos"]), oe_cos=_np(cosv[x.long()]), tm_seen=_np(fit["tm_seen"][x.long()], bool), ref={}, ent={}, checks={})
    for k, lg in refs.items():
        l = lsm(lg); rec["ref"][k] = dict(am=_np(l.argmax(-1), np.int32), nll=_np(-l.gather(1, tgt[:, None])[:, 0]))
    true4 = FF["hs"][T - 1][1:].float()
    rest = [n for n in names if n != "ORA"]
    big = torch.cat([Z[n] for n in rest]); pos = S.repeat(len(rest)); tg = tgt.repeat(len(rest))
    out = pr.slot_from(big, pos, FF, tgt=tg, chunk=chunk, keep_logits=False)
    for j, n in enumerate(rest):
        sl = slice(j * nq, (j + 1) * nq); h = out["h"][sl].float()
        rec["ent"][n] = dict(am=_np(out["am"][sl], np.int32), nll=_np(out["nll"][sl]),
                             cos_out=_np(torch.nn.functional.cosine_similarity(h, true4, dim=-1)),
                             cos_in=_np(torch.nn.functional.cosine_similarity(Z[n].float(), f["Y"], dim=-1)))
    del out, big
    if "ORA" in names:
        o = pr.slot_from(Z["ORA"], S, FF, tgt=tgt, chunk=chunk, keep_logits=True); h = o["h"].float()
        rec["ent"]["ORA"] = dict(am=_np(o["am"], np.int32), nll=_np(o["nll"]), cos_out=_np(torch.nn.functional.cosine_similarity(h, true4, dim=-1)),
                                 cos_in=np.ones(nq, np.float32))
        rec["checks"]["ORA_logits_vs_passT"] = float((o["logits"] - FF["lg"][T - 1][1:]).abs().max())
        rec["checks"]["ORA_state_vs_passT"] = float((h - true4).abs().max())
        del o
    if check_cell:   # E entry vs mech_core's cell(T, 1) (independent code path)
        FFm = dict(K={T - 1: FF["K"]}, V={T - 1: FF["V"]}, N=N)
        cm = pr.slot(ids_t, S, FFm, mc.cell_specs(T, 1, mc.causal_strict(S, N, dev)))["mix"]
        oE = pr.slot_from(Z["E"], S, FF, chunk=chunk, keep_logits=True)
        rec["checks"]["E_vs_mech_cell_T_1"] = float((oE["logits"] - cm).abs().max())
        del cm, oE
    del FF, f, Z
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help='"tag=dir tag=dir ..."'); ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="gsm8k_cot_zeroshot:0:300 ifeval:0:150"); ap.add_argument("--fit", default="gsm8k_cot_zeroshot:300:600:550")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--fit-limit", type=int, default=0); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=4096); ap.add_argument("--lmo-root", default=LMO)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    mc.strict_fp32(); dev = a.device if torch.cuda.is_available() else "cpu"; T0 = time.time()
    for spec in a.models.split():
        tag, mdir = spec.split("=", 1); t0 = time.time(); od = f"{a.out}/{tag}"; os.makedirs(od, exist_ok=True)
        log = lambda s: print(s, flush=True)
        m, tok = mc.load_model(mdir, dev); nv = len(tok); pr = ec.EntryProbe(m, nv)
        tie = bool(getattr(m.config, "tie_word_embeddings", False)); same = m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr()
        cosv = ec.centered_cos(m.model.embed_tokens.weight[:nv], m.lm_head.weight[:nv])
        meta = dict(model=mdir, model_tag=tag, T=pr.T, L=pr.L, nv=nv, vocab_rows=int(m.lm_head.weight.shape[0]), hidden=m.config.hidden_size,
                    tie_word_embeddings=tie, lm_head_is_embed=same, dtype="float32 (TF32 off); readout/NLL fp32; ridge fp64",
                    torch=torch.__version__, host=os.uname().nodename, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                    entries=ec.ENTRIES, extra=sorted(ec.EXTRA), topk=ec.TOPK, lams=list(ec.LAMS),
                    cos_Wout_E_centered=dict(mean=float(cosv.mean()), median=float(cosv.median()), p10=float(cosv.quantile(.1)), p90=float(cosv.quantile(.9)),
                                             abs_mean=float(cosv.abs().mean())))
        log(f"[entry] {tag}: loaded {mdir} T={pr.T} L={pr.L} nv={nv} rows={meta['vocab_rows']} tie={tie} same_ptr={same} on {meta['gpu']}; "
            f"cos(W_out,E) centered mean {meta['cos_Wout_E_centered']['mean']:.4f} ({time.time() - t0:.0f}s)")
        # ---------------- fit
        fdocs, fmiss, fspecs = [], {}, []
        for fs in a.fit.split(","):                      # several fit sets are pooled (each with its own held-out tail)
            ft, flo, fhi, fval = fs.split(":"); fd, fm = mr.load_docs(ft, int(flo), int(fhi), tok, src_tag=tag, root=a.lmo_root)
            for x in fd: x["val"] = x["doc_id"] >= int(fval); x["fit_task"] = ft
            if a.fit_limit:
                fd = [x for x in fd if not x["val"]][: a.fit_limit] + [x for x in fd if x["val"]][: max(1, a.fit_limit // 5)]
            fdocs += fd; fmiss[ft] = fm; fspecs.append(dict(task=ft, lo=int(flo), hi=int(fhi), val_lo=int(fval), docs=len(fd)))
            log(f"[entry] {tag} fit set {ft} {flo}-{fhi} (val >= {fval}): {len(fd)} docs, missing {fm}")
        fit, finfo = fit_phase(pr, fdocs, None, log)
        meta["fit"] = dict(task="+".join(x["task"] for x in fspecs), specs=fspecs, missing=fmiss, **finfo)
        torch.save({k: (v[0].cpu(), v[1].cpu()) for k, v in fit["maps"].items()} | {"c": fit["c"], "mu": fit["mu"].cpu()}, f"{od}/fit_maps.pt")
        # ---------------- eval
        for tspec in a.tasks.split():
            task, lo, hi = tspec.split(":"); lo, hi = int(lo), int(hi)
            docs, missing = mr.load_docs(task, lo, hi, tok, src_tag=tag, root=a.lmo_root)
            if a.limit: docs = docs[: a.limit]
            log(f"[entry] {tag} {task}: {len(docs)} docs (missing {missing}), rt_ok {sum(x['rt_ok'] for x in docs)}")
            recs, worst, t1, npos = [], {}, time.time(), 0
            for i, dd in enumerate(docs):
                ids = dd["ctx"] + dd["gen"]
                assert max(ids) < nv, f"token id {max(ids)} >= nv {nv}"
                if len(dd["gen"]) < 2: log(f"[entry] skip doc {dd['doc_id']} (gen {len(dd['gen'])})"); continue
                rec = eval_doc(pr, ids, len(dd["ctx"]), fit, cosv, chunk=a.chunk, check_cell=(i == 0))
                rec.update(doc_id=dd["doc_id"], task=task, rt_ok=dd["rt_ok"], finish=dd["finish"]); recs.append(rec); npos += rec["nq"]
                for k, v in rec["checks"].items(): worst[k] = max(worst.get(k, 0.0), v)
                if i % 25 == 0 or i == len(docs) - 1:
                    el = time.time() - t1; full = rec["ref"]["full"]["am"]
                    ag = " ".join(f"{n} {np.mean(rec['ent'][n]['am'] == full):.2f}" for n in ("E", "NE", "LE", "OE", "SE", "H", "LEH", "ORA"))
                    log(f"[entry] {tag} {task} {i + 1}/{len(docs)} doc {dd['doc_id']} N={rec['N']} nq={rec['nq']} | {npos} pos {el:.0f}s "
                        f"({npos / max(el, 1e-9):.0f} pos/s) | agree {ag} | worst {json.dumps(worst)}")
                if dev != "cpu": torch.cuda.empty_cache()
            mt = dict(meta, task=task, lo=lo, hi=hi, docs=len(recs), positions=npos, missing=missing, worst_checks=worst, sec=time.time() - t1,
                      written=time.strftime("%Y-%m-%d %H:%M:%S"))
            tmp = f"{od}/raw_{task}.pkl.tmp"
            with open(tmp, "wb") as fh: pickle.dump(dict(meta=mt, docs=recs), fh, protocol=4)
            os.replace(tmp, f"{od}/raw_{task}.pkl")
            json.dump({k: v for k, v in mt.items()}, open(f"{od}/meta_{task}.json", "w"), indent=1, default=str)
            log(f"[entry] {tag} {task} DONE {len(recs)} docs {npos} positions {time.time() - t1:.0f}s; worst checks {worst}")
        del m, pr, fit
        if dev != "cpu": torch.cuda.empty_cache()
        open(f"{od}/.done", "w").close()
        log(f"[entry] {tag} ALL DONE {time.time() - t0:.0f}s")
    print(f"[entry] ALL MODELS DONE {time.time() - T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
