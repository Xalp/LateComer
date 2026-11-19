"""Offline multi-benchmark scorer and shard aggregator for Huginn formal evaluations.

Supports:
  - `gsm8k`   (n=1,319): pure Python `arch_explore.huginn.score_hg.score_one` (strict & flex EM)
  - `ifeval`  (n=541):   seeded `lm_eval.tasks.ifeval.utils.process_results` (`prompt_level_strict_acc`)
  - `math500` (n=500):   gold-verified `mv_lenient` (`math_verify` + `Answer:` fallback)
  - `mbpp`    (n=500):   gold-verified `pass_fixed` (`extract_code_blocks("python\\n" + text)` + 3 asserts)

Computes exact paired two-sided sign test (`p_sign`), Wilson 95% CI, `1 + E[c]` mean block length,
and paired same-GPU wall-clock speedup vs `blk:32:0` across all common documents.
"""

import argparse
import glob
import json
import math
import os
import random
import re
import statistics as st

from huginn.eval.score_gsm8k import score_one
from huginn.decode.hg_block_dec import load_merged_block_dir, parse_blk_cfg


DEFAULT_TASK_DIR = "data/tasks"


def mcnemar_sign(x: list, y: list):
    b = sum(1 for u, v in zip(x, y) if u and not v)
    c = sum(1 for u, v in zip(x, y) if v and not u)
    n = b + c
    if n == 0:
        return dict(wins=b, losses=c, p_sign=1.0)
    num = 2 * sum(math.comb(n, k) for k in range(0, min(b, c) + 1))
    p = min(1.0, num / (1 << n))
    return dict(wins=b, losses=c, p_sign=p)


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return (c - h, c + h)


def cut_until(text: str, until_list: list) -> str:
    for u in until_list or []:
        if u and u in text:
            text = text.split(u)[0]
    return text


_UNI = [
    ("−", "-"), ("–", "-"), ("×", "\\times "), ("·", "\\cdot "), ("⋅", "\\cdot "),
    ("÷", "\\div "), ("π", "\\pi "), ("∞", "\\infty "), ("≤", "\\le "), ("≥", "\\ge "),
    ("≠", "\\ne "), ("°", "^\\circ"), ("²", "^2"), ("³", "^3"), ("∪", "\\cup "),
    ("θ", "\\theta "), ("½", "\\frac{1}{2}"),
]


def _answer_line(t: str):
    ms = list(re.finditer(r"(?:final answer|answer)\s*(?:is)?\s*[:：]\s*", t, flags=re.I))
    if not ms:
        return None
    rest = t[ms[-1].end() :].strip("\n ")
    line = rest.split("\n")[0].strip() or (rest.split("\n")[1].strip() if "\n" in rest else "")
    s = line.strip().rstrip(".").strip("$ ").replace("**", "")
    for u, v in _UNI:
        s = s.replace(u, v)
    s = re.sub(r"√\s*\(([^()]*)\)", r"\\sqrt{\1}", s)
    s = re.sub(r"√\s*\{([^{}]*)\}", r"\\sqrt{\1}", s)
    s = re.sub(r"√\s*([0-9A-Za-z.]+)", r"\\sqrt{\1}", s)
    return s or None


def score_task_texts(task: str, task_meta: dict, texts: dict, seed: int = 0):
    """Return `(per_doc_dict, primary_metric_key)`."""
    if task == "gsm8k":
        out = {}
        for did, txt in texts.items():
            sc = score_one(txt, task_meta[did]["target"])
            out[did] = dict(flex=int(sc["flex"]), strict=int(sc["strict"]))
        return out, "flex"

    if task == "ifeval":
        from langdetect import DetectorFactory
        from lm_eval.tasks.ifeval.utils import process_results
        out = {}
        for did, txt in texts.items():
            random.seed(seed)
            DetectorFactory.seed = seed
            r = process_results(task_meta[did]["doc"], [txt])
            out[did] = dict(
                strict=int(r["prompt_level_strict_acc"]),
                loose=int(r["prompt_level_loose_acc"]),
                inst_strict=float(st.mean(int(x) for x in r["inst_level_strict_acc"])),
            )
        return out, "strict"

    if task == "math500":
        from lm_eval.tasks.minerva_math import utils as MU
        from math_verify import parse, verify
        out = {}
        for did, txt in texts.items():
            doc = task_meta[did]["doc"]
            r = MU.process_results(doc, [txt])
            al = _answer_line(txt)
            mv_al = 0
            if al is not None:
                gold = parse(doc["solution"])
                mv_al = int(bool(verify(gold, parse("\\boxed{" + al + "}"))))
                if not mv_al:
                    gs = re.sub(r"\\text\{([^}]*)\}", r"\1", doc["answer"]).strip().lower()
                    mv_al = int(bool(gs) and al.strip().lower().strip("\\ ") == gs)
            out[did] = dict(
                mv_lenient=int(bool(r["math_verify"]) or bool(mv_al)),
                math_verify=int(r["math_verify"]),
                exact_match=int(r["exact_match"]),
            )
        return out, "mv_lenient"

    if task == "mbpp":
        from lm_eval.tasks.mbpp import utils as BU
        dids = sorted(texts)
        refs = ["\n".join(task_meta[d]["doc"]["test_list"][:3]) for d in dids]
        codes_fixed = [BU.extract_code_blocks("python\n" + texts[d]) for d in dids]
        res = BU.pass_at_k.compute(references=refs, predictions=[[c] for c in codes_fixed], k=[1], num_workers=8)
        det = res[1]
        out = {}
        for idx, d in enumerate(dids):
            out[d] = dict(pass_fixed=int(det[idx][0][1]["passed"]))
        return out, "pass_fixed"

    if task in ("bbh", "mmlupro"):
        import sys
        lmo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "eval"))
        if lmo_dir not in sys.path:
            sys.path.insert(0, lmo_dir)
        import lmo_common as C
        from lm_eval import simple_evaluate

        _, _, ReplayLM = C.make_lm_classes()
        ouro_ref_model = "Xalphinions/ouro-2.6b-late-entry"
        by = {}
        for did in texts:
            by.setdefault(task_meta[did]["subtask"], []).append(did)
        samples, gens, ctxs, back = {}, {}, {}, {}
        for sub, gl in by.items():
            gl = sorted(gl, key=lambda g: task_meta[g]["sub_doc_id"])
            samples[sub] = [task_meta[g]["sub_doc_id"] for g in gl]
            for pos, g in enumerate(gl):
                gens[(sub, pos)] = texts[g]
                ctxs[(sub, pos)] = task_meta[g]["ouro_context"]
                back[(sub, task_meta[g]["sub_doc_id"])] = g
        lm = ReplayLM(ouro_ref_model, gens, ctxs)
        res = simple_evaluate(
            model=lm,
            tasks=sorted(by),
            num_fewshot=0,
            batch_size=1,
            apply_chat_template=True,
            log_samples=True,
            confirm_run_unsafe_code=True,
            samples=samples,
            bootstrap_iters=0,
        )
        assert not lm.mismatch, f"{len(lm.mismatch)} replayed contexts differ: {lm.mismatch[:3]}"
        assert not lm.missing, f"{len(lm.missing)} missing generations: {lm.missing[:3]}"
        prim = {"bbh": "flexible-extract", "mmlupro": "custom-extract"}[task]
        out = {}
        for sub in by:
            for s in res["samples"][sub]:
                g = back[(sub, s["doc_id"])]
                f = s["filter"]
                o = out.setdefault(g, {})
                fr = s["filtered_resps"][0] if isinstance(s["filtered_resps"], list) else s["filtered_resps"]
                if isinstance(fr, list):
                    fr = fr[0] if fr else ""
                o[f"acc_{f}"] = int(float(s["exact_match"]))
                o[f"invalid_{f}"] = int(str(fr).strip() == "[invalid]")
        for g, o in out.items():
            o["acc"] = o[f"acc_{prim}"]
            o["invalid"] = o[f"invalid_{prim}"]
        assert len(out) == len(texts), (len(out), len(texts))
        return out, "acc"

    raise ValueError(f"Unknown task: {task}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["gsm8k", "ifeval", "math500", "mbpp", "bbh", "mmlupro"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--task-file", default="")
    args = ap.parse_args()

    tfile = args.task_file or os.path.join(
        DEFAULT_TASK_DIR, "gsm8k_zs.jsonl" if args.task == "gsm8k" else f"{args.task}.jsonl"
    )
    with open(tfile) as f:
        task_meta = {int(r["doc_id"]): r for r in (json.loads(l) for l in f if l.strip())}

    raw_by_cfg = load_merged_block_dir(args.dir)
    if not raw_by_cfg:
        print(f"[score] no block jsonl files found in {args.dir}")
        return

    ref_map = {int(r["doc_id"]): r for r in raw_by_cfg["blk:32:0"]} if "blk:32:0" in raw_by_cfg else None

    per_doc_by_cfg = {}
    rows = []
    primary_key = None

    for cfg, recs in sorted(raw_by_cfg.items()):
        pc = parse_blk_cfg(cfg)
        texts = {
            int(r["doc_id"]): cut_until(r.get("text", ""), task_meta[int(r["doc_id"])].get("until", []))
            for r in recs
        }
        pdoc, pkey = score_task_texts(args.task, task_meta, texts)
        primary_key = pkey
        per_doc_by_cfg[cfg] = pdoc

        n = len(recs)
        hits = sum(int(pdoc[int(r["doc_id"])][pkey]) for r in recs)
        acc = hits / max(1, n)
        ci_lo, ci_hi = wilson_ci(hits, n)

        tot_toks = sum(int(r["tokens"]) for r in recs)
        tot_dec_toks = sum(int(r["dec_tokens"]) for r in recs)
        tot_steps = sum(int(r["steps"]) for r in recs)
        tot_dec_s = sum(float(r["decode_s"]) for r in recs)
        mean_blk = tot_dec_toks / max(1, tot_steps)
        tps = tot_dec_toks / max(1e-9, tot_dec_s)
        seq_layers = (4.0 + 4.0 * pc["K"]) / max(1e-9, mean_blk)

        ref_tps = None
        ex_match = None
        if ref_map:
            common = [int(r["doc_id"]) for r in recs if int(r["doc_id"]) in ref_map]
            if len(common) == n and n > 0:
                ref_tps = sum(int(ref_map[i]["dec_tokens"]) for i in common) / max(
                    1e-9, sum(float(ref_map[i]["decode_s"]) for i in common)
                )
                ex_match = sum(1 for r in recs if r["gen_ids"] == ref_map[int(r["doc_id"])]["gen_ids"]) / n
        if ref_tps is None:
            r32_vals = [float(r["same_gpu_r32_tok_per_s"]) for r in recs if r.get("same_gpu_r32_tok_per_s")]
            if r32_vals:
                ref_tps = sum(r32_vals) / len(r32_vals)

        secondary = {}
        for sk in pdoc[int(recs[0]["doc_id"])].keys():
            if sk != pkey:
                secondary[sk] = sum(float(pdoc[int(r["doc_id"])][sk]) for r in recs) / n

        rows.append(
            dict(
                cfg=cfg,
                kind=pc["kind"],
                K=pc["K"],
                B=pc["B"],
                bg=pc["bg"],
                n=n,
                primary_metric=pkey,
                acc=acc,
                hits=hits,
                ci95=[ci_lo, ci_hi],
                secondary=secondary,
                mean_tokens=tot_toks / n,
                mean_block_len=mean_blk,
                seq_layers_per_tok=seq_layers,
                decode_tok_per_s=tps,
                wall_speedup_vs_r32=(tps / ref_tps) if ref_tps else None,
                exact_seq_match=ex_match,
            )
        )

    rows.sort(key=lambda r: (r["kind"], r["B"], -r["K"], r["bg"]))
    for r in rows:
        cfg = r["cfg"]
        if "blk:32:0" in per_doc_by_cfg and cfg != "blk:32:0":
            cids = sorted(set(per_doc_by_cfg[cfg]) & set(per_doc_by_cfg["blk:32:0"]))
            if cids:
                ms = mcnemar_sign(
                    [per_doc_by_cfg[cfg][i][primary_key] for i in cids],
                    [per_doc_by_cfg["blk:32:0"][i][primary_key] for i in cids],
                )
                ms["delta"] = (ms["wins"] - ms["losses"]) / len(cids)
                r["vs_r32"] = ms
        if r["kind"] == "blk":
            ecfg = f"eblk:{r['K']}:{r['B']}"
            if ecfg in per_doc_by_cfg:
                cids = sorted(set(per_doc_by_cfg[cfg]) & set(per_doc_by_cfg[ecfg]))
                if cids:
                    ms = mcnemar_sign(
                        [per_doc_by_cfg[cfg][i][primary_key] for i in cids],
                        [per_doc_by_cfg[ecfg][i][primary_key] for i in cids],
                    )
                    ms["delta"] = (ms["wins"] - ms["losses"]) / len(cids)
                    r["vs_eblk"] = ms

    out_json = os.path.join(args.dir, f"scored_{args.task}.json")
    with open(out_json + ".tmp", "w") as f:
        json.dump(dict(task=args.task, tag=args.tag, primary=primary_key, rows=rows), f, indent=2)
    os.replace(out_json + ".tmp", out_json)

    lines = [
        f"# Huginn Formal Evaluation: `{args.task}` (`{args.tag or os.path.basename(args.dir)}`, primary=`{primary_key}`)",
        "",
        "| Config | Kind | B | K | BG | N | Primary Acc | Gap vs `r:32` (p) | vs `eblk` (p) | `1+E[c]` | Tok/s | Speedup |",
        "|---|---|---:|---:|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for r in rows:
        vr = r.get("vs_r32")
        ve = r.get("vs_eblk")
        vr_s = f"{vr['delta']*100:+.2f}% ({vr['p_sign']:.3g})" if vr else "-"
        ve_s = f"{ve['delta']*100:+.2f}% ({ve['p_sign']:.3g})" if ve else "-"
        sp_s = f"{r['wall_speedup_vs_r32']:.2f}x" if r["wall_speedup_vs_r32"] is not None else "-"
        lines.append(
            f"| `{r['cfg']}` | {r['kind']} | {r['B']} | {r['K']} | {r['bg']} | {r['n']} | "
            f"{r['acc']*100:.2f}% | {vr_s} | {ve_s} | {r['mean_block_len']:.2f} | "
            f"{r['decode_tok_per_s']:.2f} | {sp_s} |"
        )
    out_md = os.path.join(args.dir, f"scored_{args.task}.md")
    with open(out_md + ".tmp", "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(out_md + ".tmp", out_md)
    print("\n".join(lines))
    print(f"\n[score] wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
