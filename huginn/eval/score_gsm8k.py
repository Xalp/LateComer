"""Exact standalone GSM8K-CoT scorer and paired sign-test helper for Huginn-3.5B.

Matches lm-evaluation-harness `gsm8k_cot` (`gsm8k-cot.yaml`) bit-for-bit:
  - strict-match:     RegexFilter(r"The answer is (\-?[0-9\.\,]+).", group_select=0)
  - flexible-extract: RegexFilter(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", group_select=-1)
  - exact_match:      regexes_to_ignore=[',', '\\$', '(?s).*#### ', '\\.$'], ignore_case=True
"""

import json
import math
import re

RE_STRICT = re.compile(r"The answer is (\-?[0-9\.\,]+).")
RE_FLEX = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
IGNORE_RES = [re.compile(p) for p in [r",", r"\$", r"(?s).*#### ", r"\.$"]]
FALLBACK = "[invalid]"


def _extract(resp: str, regex, group_select: int = 0) -> str:
    if not isinstance(resp, str):
        resp = ""
    m = regex.findall(resp)
    if not m:
        return FALLBACK
    sel = m[group_select]
    if isinstance(sel, tuple):
        nz = [x for x in sel if x]
        sel = nz[0] if nz else FALLBACK
    return sel.strip()


def _norm(s: str) -> str:
    for rx in IGNORE_RES:
        s = rx.sub("", s)
    return s.lower()


def score_one(resp: str, target: str) -> dict:
    ref = _norm(str(target))
    pred_s = _norm(_extract(resp, RE_STRICT, group_select=0))
    pred_f = _norm(_extract(resp, RE_FLEX, group_select=-1))
    return dict(
        strict=int(pred_s == ref),
        flex=int(pred_f == ref),
        pred_strict=pred_s,
        pred_flex=pred_f,
        ref=ref,
    )


def load_gold(gold_path: str) -> dict:
    with open(gold_path) as f:
        data = json.load(f)
    return {int(d["doc_id"]): str(d["target"]) for d in data}


def sign_test_p(wins: int, losses: int) -> float:
    n = int(wins) + int(losses)
    if n == 0:
        return 1.0
    k = min(int(wins), int(losses))
    num = 2 * sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, num / (1 << n))


def score_records(records: list, gold_map: dict) -> dict:
    n = len(records)
    if n == 0:
        return dict(n=0)
    s_hits, f_hits, toks, crits, rsteps, ags = 0, 0, 0, 0, 0, []
    fin, per_doc = {}, {}
    for r in records:
        did = int(r["doc_id"])
        sc = score_one(r.get("text", ""), gold_map[did])
        s_hits += sc["strict"]
        f_hits += sc["flex"]
        toks += int(r.get("tokens", 0))
        crits += int(r.get("crit_steps", 0))
        rsteps += int(r.get("row_steps", 0))
        if r.get("agree") is not None:
            ags.append(float(r["agree"]))
        fn = r.get("finish", "unknown")
        fin[fn] = fin.get(fn, 0) + 1
        per_doc[did] = dict(
            strict=sc["strict"],
            flex=sc["flex"],
            tokens=int(r.get("tokens", 0)),
            crit_steps=int(r.get("crit_steps", 0)),
            agree=r.get("agree"),
        )
    cfg = records[0].get("cfg", "")
    parts = cfg.split(":") if cfg else []
    mode = parts[0] if parts else ""
    D = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    t = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else D
    mean_crit_tok = crits / max(toks, 1)
    crit_layers_tok = 4.0 + 4.0 * mean_crit_tok
    return dict(
        cfg=cfg,
        mode=mode,
        D=D,
        t=t,
        n=n,
        strict_acc=s_hits / n,
        flex_acc=f_hits / n,
        strict_hits=s_hits,
        flex_hits=f_hits,
        mean_tokens=toks / n,
        mean_crit_steps_per_tok=mean_crit_tok,
        mean_crit_layers_per_tok=crit_layers_tok,
        layer_speedup_vs_r32=132.0 / crit_layers_tok,
        mean_row_steps_per_tok=rsteps / max(toks, 1),
        mean_agree=(sum(ags) / len(ags)) if ags else None,
        finish_counts=fin,
        per_doc=per_doc,
    )


def paired_compare(doc_a: dict, doc_b: dict, key: str = "flex") -> dict:
    common = sorted(set(doc_a) & set(doc_b))
    if not common:
        return None
    w = sum(1 for d in common if doc_a[d][key] > doc_b[d][key])
    l = sum(1 for d in common if doc_a[d][key] < doc_b[d][key])
    tie = len(common) - w - l
    return dict(n=len(common), wins=w, losses=l, ties=tie, diff=(w - l) / len(common), p_sign=sign_test_p(w, l))
