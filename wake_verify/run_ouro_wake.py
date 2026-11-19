"""run_ouro_wake.py -- Full-dataset GPU evaluation runner for Ouro-2.6B Wake Verification & Suffix Orthogonality.

Supports all `ar`, `latt:B`, `lk:K:shape`, `lks:K:shape`, `wk:K:shape:tau`, and `wks:K:shape:tau` modes,
recording exact rollback statistics, both end-to-end (`tok_eff_per_s`) and pure-decode (`decode_tok_per_s`)
throughputs, CUDA-graph vs eager verification gates, and full-depth committed-token fidelity audits.
"""

import argparse
import json
import os
import socket
import statistics
import sys
import time
import traceback
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEK_DIR = os.path.join(REPO_ROOT, "decode")
for p in (LATEK_DIR, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import engine as E
import sparse as SP
import late as LT
import latek as LK
from wake_verify.ouro_wake import WakeKLattice, parse_wake_mode


def is_wake_like(mode: str) -> bool:
    return mode.startswith(("wk:", "wks:", "lks:"))


def make_any_decoder(eng, mode: str, nv: int, eos, mask_id: int, **kw):
    if is_wake_like(mode):
        pm = parse_wake_mode(mode)
        for k_ in ("margins", "switch_L", "dump", "check", "bg"):
            kw.pop(k_, None)
        at = kw.pop("attn", "paged")
        at = "torch" if at == "torch" else "paged"
        return WakeKLattice(
            eng, pm["K"], pm["B"], pm["sets"], nv, eos, mask_id,
            tau=pm["tau"], suffix=pm["suffix"], attn=at, **kw,
        )
    return LK.make_decoder(eng, mode, nv, eos, mask_id, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--meta", default="")
    ap.add_argument("--docs", default="0:999999")
    ap.add_argument("--modes", default="ar,lk:4:latt:8,wk:3:latt:8:0,lk:3:latt:8,wk:2:latt:8:0,wk:2:latt:8:0.25,lk:2:latt:8")
    ap.add_argument("--attn", default="paged")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--until", type=int, default=1)
    ap.add_argument("--graphs", type=int, default=1)
    ap.add_argument("--fused", type=int, default=1)
    ap.add_argument("--S", type=int, default=2816)
    ap.add_argument("--burn-s", type=float, default=15.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--prefill-graph", type=int, default=1)
    ap.add_argument("--save-gen", type=int, default=1)
    ap.add_argument("--graph-vs-eager", type=int, default=1)
    ap.add_argument("--audit", type=int, default=1)
    ap.add_argument("--roofline-bench", type=int, default=0)
    a = ap.parse_args()

    T0 = time.time()
    def log(*x):
        print(f"[ouro-wake {time.time() - T0:7.1f}s]", *x, flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if os.path.exists(a.out):
        try:
            prev = json.load(open(a.out))
            if prev.get("meta", {}).get("done") is True:
                log(f"[skip] {a.out} already marked done=True")
                return
        except Exception:
            pass
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT = dict(
        meta=dict(
            host=socket.gethostname(), argv=sys.argv, torch=torch.__version__,
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if dev == "cuda" else "cpu",
        ),
        modes={}, summary={}, roofline=[], checks=[], audit={},
    )

    def save():
        if os.path.exists(a.out) and not OUT["meta"].get("done"):
            try:
                if json.load(open(a.out)).get("meta", {}).get("done") is True:
                    return
            except Exception:
                pass
        tmp = f"{a.out}.{socket.gethostname()}.{os.getpid()}.tmp"
        with open(tmp, "w") as fp:
            json.dump(OUT, fp)
        os.replace(tmp, a.out)

    meta_path = a.meta or os.path.join(
        os.path.dirname(a.prompts), "meta_" + os.path.basename(a.prompts).replace(".jsonl", ".json")
    )
    PM = json.load(open(meta_path))
    NV, EOS, MASK = PM["nv"], PM["eos"], PM["mask_id"]
    docs = [json.loads(l) for l in open(a.prompts)]
    d0, d1 = (int(x) for x in a.docs.split(":"))
    docs = [d for d in docs if d0 <= d["doc_id"] < d1]
    MODES = [m.strip() for m in a.modes.split(",") if m.strip()]
    log(f"{len(docs)} prompts, nv={NV} eos={EOS} mask={MASK}, modes={MODES}")

    DT = getattr(torch, a.dtype)
    W, cfg = E.load_weights(a.model, dev, DT)
    eng = LT.LateEngine(W, cfg, dev, DT, S=a.S, fused=bool(a.fused), log=log, M=8, Bmax=16)
    del W
    OUT["meta"]["ops"] = eng.ops.chosen

    tokzr = None
    if a.until:
        from tokenizers import Tokenizer
        tokzr = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))

    def has_stop(ids, until):
        return any(s in tokzr.decode(ids[-64:], skip_special_tokens=False) for s in until)

    def eff_len(gen, until):
        if not until or tokzr is None:
            return len(gen)
        for k in range(1, len(gen) + 1):
            if has_stop(gen[:k], until):
                return k
        return len(gen)

    if a.prefill_graph and dev == "cuda":
        eng.prefill_graphs = True
        for Pb in sorted({-(-len(d["ids"]) // eng.pg_bucket) * eng.pg_bucket for d in docs}):
            eng._pgraph(Pb, NV)
        log(f"prefill graphs captured for {len(eng.pgraphs)} buckets")

    if dev == "cuda" and a.burn_s > 0:
        x = torch.randn(512, 2048, device=dev, dtype=torch.bfloat16)
        w = eng.layers[0]["gu"]
        te = time.time() + a.burn_s
        while time.time() < te:
            for _ in range(20):
                F.linear(x, w)
            torch.cuda.synchronize()
        del x
        torch.cuda.empty_cache()

    # Optional Pillar 3 hardware roofline curve `t_pass(N_rows)` across N_rows in [1 .. 2048]
    if a.roofline_bench and dev == "cuda":
        for Lc in (128, 384):
            for nrows in (1, 2, 4, 8, 16, 25, 32, 49, 64, 81, 99, 128, 198, 256, 384, 512, 768, 1024, 1536, 2048):
                m = E.Meta()
                m.cu_q = torch.tensor([0, nrows], device=dev, dtype=torch.int32)
                m.max_q = nrows
                m.pos = torch.full((nrows,), Lc, dtype=torch.int64, device=dev)
                m.wslot = torch.full((nrows,), eng.DUMMY, dtype=torch.int64, device=dev)
                m.seqlens = torch.tensor([Lc + nrows], dtype=torch.int32, device=dev)
                pt = torch.zeros((1, eng.S), dtype=torch.int32, device=dev)
                pt[0, : Lc + nrows] = torch.arange(Lc + nrows, dtype=torch.int32, device=dev) % Lc
                m.page_table = pt
                X = torch.randn((nrows, cfg.H), device=dev, dtype=DT)
                for _ in range(5):
                    eng.late_pass(X, m, "paged")
                torch.cuda.synchronize()
                evs = [torch.cuda.Event(enable_timing=True) for _ in range(41)]
                evs[0].record()
                for r_i in range(40):
                    eng.late_pass(X, m, "paged")
                    evs[r_i + 1].record()
                torch.cuda.synchronize()
                ts = sorted(evs[i].elapsed_time(evs[i + 1]) for i in range(40))
                med_ms = ts[len(ts) // 2]
                OUT["roofline"].append(dict(Lc=Lc, rows=nrows, ms_pass_median=round(med_ms, 4)))
                log(f"[roofline] Lc={Lc:4d} rows={nrows:4d} -> {med_ms:.3f} ms/pass")
        save()

    DEC = {}
    def get_dec(mode, attn):
        key = (mode, attn)
        if key not in DEC:
            L = make_any_decoder(eng, mode, NV, EOS, MASK, attn=attn, graphs=(bool(a.graphs) and dev == "cuda"))
            L.capture()
            DEC[key] = L
            log(f"captured {mode}@{attn} (N={L.N})")
        return DEC[key]

    PREFILL = "auto" if dev == "cuda" else "torch"

    # Graph vs eager identity check on first 2 docs
    if a.graph_vs_eager and dev == "cuda" and a.graphs:
        for mode in [m for m in MODES if m.startswith(("lk:", "wk:", "wks:", "lks:"))]:
            Lg = get_dec(mode, "paged")
            Le = make_any_decoder(eng, mode, NV, EOS, MASK, attn="paged", graphs=False)
            for d in docs[:2]:
                rg = Lg.decode(d["ids"], 96, prefill_attn=PREFILL)
                re_ = Le.decode(d["ids"], 96, prefill_attn=PREFILL)
                same = rg["gen_raw"] == re_["gen_raw"]
                OUT["checks"].append(dict(kind="graph_vs_eager", mode=mode, doc=d["doc_id"], identical=same))
                log(f"[gate] graph vs eager {mode} doc {d['doc_id']}: identical={same} (rb={rg.get('n_rollbacks', 0)})")
                assert same, f"CUDA graph vs eager mismatch on {mode} doc {d['doc_id']}"
            del Le
            torch.cuda.empty_cache()
        save()

    # Main timed benchmark loop
    for mode in MODES:
        at = "paged" if dev == "cuda" else "torch"
        L = get_dec(mode, at)
        L.decode(docs[0]["ids"], 32, prefill_attn=PREFILL)  # warm-up

        recs = []
        key = f"{mode}@{at}"
        for idx, d in enumerate(docs):
            until = list(d.get("until") or []) if a.until else []
            sf = (lambda g, u=until: has_stop(g, u)) if until else None
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = L.decode(d["ids"], a.max_new, stop_fn=sf, prefill_attn=PREFILL)
            if dev == "cuda":
                torch.cuda.synchronize()
            sec = time.perf_counter() - t0
            gen = r["gen"]
            te = min(eff_len(gen, until), a.max_new)
            rec = dict(
                doc_id=d["doc_id"], prompt_tokens=len(d["ids"]), tokens=len(gen), tokens_eff=te,
                steps=r["steps"], sec=sec, prefill_ms=r["prefill_ms"],
                n_acc=r.get("n_acc", 0),
                n_rollbacks=r.get("n_rollbacks", 0),
                n_rolled_tokens=r.get("n_rolled_tokens", 0),
                n_drain_steps=r.get("n_drain_steps", 0),
                finish="eos" if len(r["gen_raw"]) > len(gen) else ("stop" if until and has_stop(gen[:te], until) else "length"),
            )
            if a.save_gen:
                rec["gen"] = gen
            if tokzr is not None:
                rec["text"] = tokzr.decode(gen[:te], skip_special_tokens=False)
            recs.append(rec)

        OUT["modes"][key] = recs
        tok = sum(x["tokens"] for x in recs)
        tok_e = sum(x["tokens_eff"] for x in recs)
        sec = sum(x["sec"] for x in recs)
        steps = sum(x["steps"] for x in recs)
        tot_rb = sum(x["n_rollbacks"] for x in recs)
        tot_rbtok = sum(x["n_rolled_tokens"] for x in recs)
        if is_wake_like(mode):
            npass = parse_wake_mode(mode)["K"]
        elif mode.startswith("lk:"):
            npass = LK.parse_latek(mode)["K"]
        else:
            npass = cfg.T

        summ = dict(
            mode=mode, attn=at, rows=L.N, docs=len(recs), tokens=tok, tokens_eff=tok_e,
            sec=sec, steps=steps, passes_per_step=npass,
            n_rollbacks=tot_rb, n_rolled_tokens=tot_rbtok,
            tok_per_s=tok / max(1e-9, sec),
            tok_eff_per_s=tok_e / max(1e-9, sec),
            tok_per_step=tok_e / max(1, steps),
            decode_tok_per_s=sum(max(0, x["tokens_eff"] - 1) for x in recs) / max(1e-9, sec - sum(x["prefill_ms"] for x in recs) / 1000.0),
            prefill_ms_mean=statistics.mean(x["prefill_ms"] for x in recs),
        )
        ar = next((v for k, v in OUT["summary"].items() if k.startswith("ar@")), None)
        if ar and mode != "ar":
            summ["speedup_vs_ar"] = summ["tok_eff_per_s"] / ar["tok_eff_per_s"]
            summ["decode_speedup_vs_ar"] = summ["decode_tok_per_s"] / ar["decode_tok_per_s"]
        OUT["summary"][key] = summ
        log(
            f"MODE {key}: docs={len(recs)} tok_eff={tok_e} steps={steps} rb={tot_rb} (rolled_tok={tot_rbtok}) | "
            f"e2e={summ['tok_eff_per_s']:.1f} tok/s ({summ.get('speedup_vs_ar', 1.0):.2f}x) | "
            f"dec={summ['decode_tok_per_s']:.1f} tok/s ({summ.get('decode_speedup_vs_ar', 1.0):.2f}x)"
        )
        save()

    # Committed-token full-depth fidelity audit
    if a.audit:
        by_id = {d["doc_id"]: d for d in docs}
        for key, recs in OUT["modes"].items():
            nd = nt = 0
            for rec in recs:
                g = rec["gen"][: rec["tokens_eff"]]
                P0 = rec["prompt_tokens"]
                if not g:
                    continue
                ids = by_id[rec["doc_id"]]["ids"] + g
                fd, _ = LT.audit_full_depth(eng, ids, P0, len(g), NV)
                nd += sum(1 for i in range(len(g)) if fd[i] != g[i])
                nt += len(g)
            OUT["audit"][key] = dict(tokens=nt, diff=nd, share=nd / max(1, nt))
            log(f"AUDIT {key}: {nd}/{nt} committed tokens differ from T=4 full-depth ({100.0 * nd / max(1, nt):.3f}%)")
            save()

    OUT["meta"]["done"] = True
    save()
    log("ALL DONE")


if __name__ == "__main__":
    main()
