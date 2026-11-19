"""score_x.py -- evalx scorer (CPU; lm-eval 0.4.12). Extends ../late_k/score_tasks.py to MATH500, MBPP,
BBH and MMLU-Pro. Inputs = run_latek.py result json(s) (modes -> per-doc records with doc_id = global id, text, gen, tokens_eff, finish).

  ENV="PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/evalx_pylib HF_DATASETS_CACHE=/tmp/hf_dscache_evalx HF_DATASETS_OFFLINE=1 HF_ALLOW_CODE_EVAL=1"
  $ENV python score_x.py --task math500 --glob 'pilot/pa_math500_*.json' [--cap 1024] --out pilot/score_math500.json
  $ENV ... score_x.py --task math500 --selftest      # gold-as-model-answer through the exact scorer (3 formats)
  $ENV ... score_x.py --task mbpp --selftest         # MBPP reference solutions through the extractor + executor (stock and prefix-restored)

Text handling (all tasks): the record's text (= tokens up to and including the first stop string; EOS excluded) is cut at the first `until`
string of the doc, as the harness's generation backend would (= ../late_k/score_tasks.py). --cap N re-derives the result a max_new=N run
would have produced from a longer run (greedy decoding is causal: the first N tokens are identical) by decoding gen[:N].
Scoring per task (the harness's own functions; per-doc diagnostics added):
  math500  lm_eval.tasks.minerva_math.utils.process_results(doc,[text]) -> exact_match (Minerva "Final Answer: The final answer is X. I hope
           it is correct." regex, then normalize + sympy equivalence) and math_verify (math_verify.verify(parse(solution), parse(text)));
           PRIMARY = mv_lenient = math_verify OR math_verify on the model's last 'Answer:' line (Unicode -> LaTeX), see PILOT.md; stock math_verify reported. Diagnostics: minerva sentence found, \\boxed found, math_verify parse(text) == [] (extraction failure).
           Needs antlr4-python3-runtime 4.11 (the env has 4.9.3; side dir /tmp/evalx_pylib, see PILOT.md). Main thread only (signal timeouts).
  mbpp     mbpp_instruct (gen_prefix "\\n```python\\n" is in the prompt). STOCK = utils.extract_code_blocks(text) -> code_eval pass@1 on the
           doc's 3 asserts (= the harness). The stock extractor prepends "```" and its pattern ```(?:\\w+)?\\n? then eats the first word of the
           code ("def", "import") as a language tag. FIXED (PRIMARY) = extract_code_blocks("python\\n" + text), i.e. the extractor sees the
           real assistant message "```python\\n" + text. Diagnostics: empty extraction, no closing fence (truncated / runaway block).
  bbh, mmlupro  replay through lm-eval itself (simple_evaluate with a ReplayLM, the per-subtask filter_list / metric kwargs / aggregation of
           the yaml; every replayed context asserted == the materialized context). PRIMARY filter: bbh flexible-extract, mmlupro custom-extract;
           group number = size-weighted mean over subtasks (= micro mean, as the group yamls say). Format loss = filtered answer "[invalid]".
Per mode: accuracy + Wilson 95% CI, paired exact McNemar vs `ar`, mean generation length (tokens_eff), cap-hit rate (tokens_eff >= cap),
speed from the run summary (tok_eff_per_s, decode_tok_per_s, speedup vs same-process ar).
"""
import argparse, glob, json, math, os, statistics as st, sys
ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=["math500", "mbpp", "bbh", "mmlupro"]); ap.add_argument("--glob", nargs="*", default=[])
ap.add_argument("--cap", type=int, default=0); ap.add_argument("--out", default=""); ap.add_argument("--selftest", action="store_true")
ap.add_argument("--model", required=True, help="path to the fine-tuned model directory")
ap.add_argument("--prompts", default="", help="prompt file name (default data/<task>_prompts.jsonl)")
a = ap.parse_args()
EX = os.path.dirname(os.path.abspath(__file__)); D = f"{EX}/data"
LMT = {"math500": "minerva_math500", "mbpp": "mbpp_instruct", "bbh": "bbh_cot_zeroshot", "mmlupro": "mmlu_pro"}[a.task]
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_dscache_evalx"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
RAW = {}
for l in open(f"{D}/raw/{LMT}.jsonl"):
    r = json.loads(l); RAW[r["gid"]] = r
PR = {}
for l in open(f"{D}/{a.prompts or a.task + '_prompts.jsonl'}"):
    r = json.loads(l); PR[r["doc_id"]] = r


def wilson(k, n, z=1.96):
    if n == 0: return (None, None)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar(x, y):
    b = sum(1 for u, v in zip(x, y) if u and not v); c = sum(1 for u, v in zip(x, y) if v and not u); n = b + c
    if n == 0: return b, c, 1.0
    return b, c, min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n)


def cut(text, until):
    for u in until or []:
        if u and u in text: text = text.split(u)[0]
    return text


TOK = None
def text_of(rec):
    global TOK
    t = rec["text"]
    if a.cap and rec["tokens_eff"] > a.cap:
        if TOK is None:
            from tokenizers import Tokenizer
            TOK = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
        t = TOK.decode(rec["gen"][: a.cap], skip_special_tokens=False)
    return cut(t, PR[rec["doc_id"]]["until"])


# ------------------------------------------------------------------------------------------------ per-task scorers: {gid: text} -> {gid: dict}
import re as _re
_UNI = [("−", "-"), ("–", "-"), ("×", "\\times "), ("·", "\\cdot "), ("⋅", "\\cdot "), ("÷", "\\div "), ("π", "\\pi "), ("∞", "\\infty "),
        ("≤", "\\le "), ("≥", "\\ge "), ("≠", "\\ne "), ("°", "^\\circ"), ("²", "^2"), ("³", "^3"), ("∪", "\\cup "), ("θ", "\\theta "), ("½", "\\frac{1}{2}")]


def answer_line(t):
    """the model's own final-answer statement (this model writes 'Answer: ...' in plain text / Unicode, rarely \\boxed): text after the LAST
    'Answer:' / 'Final Answer:' (same line, else the next non-empty line), Unicode math -> LaTeX, sqrt forms -> \\sqrt{..}"""
    ms = list(_re.finditer(r"(?:final answer|answer)\s*(?:is)?\s*[:：]\s*", t, flags=_re.I))
    if not ms: return None
    rest = t[ms[-1].end():].strip("\n "); line = rest.split("\n")[0].strip() or (rest.split("\n")[1].strip() if "\n" in rest else "")
    s = line.strip().rstrip(".").strip("$ ").replace("**", "")
    for u, v in _UNI: s = s.replace(u, v)
    s = _re.sub(r"√\s*\(([^()]*)\)", r"\\sqrt{\1}", s); s = _re.sub(r"√\s*\{([^{}]*)\}", r"\\sqrt{\1}", s); s = _re.sub(r"√\s*([0-9A-Za-z.]+)", r"\\sqrt{\1}", s)
    return s or None


def score_math(texts):
    from lm_eval.tasks.minerva_math import utils as MU
    from math_verify import parse, verify
    out = {}
    for g, t in texts.items():
        doc = RAW[g]["doc"]; r = MU.process_results(doc, [t])
        al = answer_line(t); mv_al = 0
        if al is not None:
            gold = parse(doc["solution"])
            mv_al = int(bool(verify(gold, parse("\\boxed{" + al + "}"))))
            if not mv_al:   # text answers (e.g. "Evelyn"): exact string match against the normalized gold
                gs = _re.sub(r"\\text\{([^}]*)\}", r"\1", doc["answer"]).strip().lower(); mv_al = int(bool(gs) and al.strip().lower().strip("\\ ") == gs)
        out[g] = dict(math_verify=int(r["math_verify"]), exact_match=int(r["exact_match"]), has_minerva=int(MU.get_unnormalized_answer(t) != "[invalidanswer]"),
                      has_boxed=int(MU.last_boxed_only_string(t) is not None), mv_extract_fail=int(len(parse(t)) == 0),
                      has_answer_line=int(al is not None), mv_answerline=mv_al, mv_lenient=int(bool(r["math_verify"]) or bool(mv_al)))
    return out, "mv_lenient"   # PRIMARY = stock math_verify OR answer-line math_verify (PILOT.md: stock misses Unicode/plain answers)


CE = None
def code_eval(refs, codes):
    """per-item pass (code_eval: 1 candidate per problem, 3 s timeout) -> list of 0/1"""
    global CE
    if CE is None:
        from lm_eval.tasks.mbpp import utils as BU
        CE = BU.pass_at_k
    res = CE.compute(references=refs, predictions=[[c] for c in codes], k=[1], num_workers=16)
    det = res[1]
    return [int(det[i][0][1]["passed"]) for i in range(len(refs))]


def mbpp_ref(doc): return "\n".join(doc["test_list"][:3])


def score_mbpp(texts):
    from lm_eval.tasks.mbpp import utils as BU
    gs = sorted(texts); refs = [mbpp_ref(RAW[g]["doc"]) for g in gs]
    stock = [BU.extract_code_blocks(texts[g]) for g in gs]; fixed = [BU.extract_code_blocks("python\n" + texts[g]) for g in gs]
    ps, pf = code_eval(refs, stock), code_eval(refs, fixed)
    out = {}
    for i, g in enumerate(gs):
        t = texts[g]
        out[g] = dict(pass_fixed=pf[i], pass_stock=ps[i], empty_fixed=int(fixed[i].strip() == ""), empty_stock=int(stock[i].strip() == ""),
                      no_close_fence=int("```" not in t), immediate_close=int(t.lstrip().startswith("```")))
    return out, "pass_fixed"


def score_replay(texts):
    """BBH / MMLU-Pro: replay through the harness (its filters + metrics). Returns per gid: acc (primary filter), acc_<filter>, invalid_<filter>"""
    sys.path.insert(0, os.environ.get("LE_LMEVAL_DIR", os.path.dirname(os.path.abspath(__file__))))  # lm-eval checkout with the scorer corrections (README)
    import lmo_common as C
    from lm_eval import simple_evaluate
    _, _, ReplayLM = C.make_lm_classes()
    by = {}
    for g in texts: by.setdefault(RAW[g]["task"], []).append(g)
    samples, gens, ctxs, back = {}, {}, {}, {}
    for sub, gl in by.items():
        gl = sorted(gl, key=lambda g: RAW[g]["doc_id"]); ids = [RAW[g]["doc_id"] for g in gl]; samples[sub] = ids
        for pos, g in enumerate(gl):
            gens[(sub, pos)] = texts[g]; ctxs[(sub, pos)] = RAW[g]["context"]; back[(sub, RAW[g]["doc_id"])] = g
    lm = ReplayLM(a.model, gens, ctxs)
    res = simple_evaluate(model=lm, tasks=sorted(by), num_fewshot=0, batch_size=1, apply_chat_template=True, log_samples=True,
                          confirm_run_unsafe_code=True, samples=samples, bootstrap_iters=0)
    assert not lm.mismatch, f"{len(lm.mismatch)} replayed contexts differ from the materialized ones, e.g. {lm.mismatch[:3]}"
    assert not lm.missing, f"{len(lm.missing)} docs without a generation, e.g. {lm.missing[:3]}"
    prim = {"bbh": "flexible-extract", "mmlupro": "custom-extract"}[a.task]
    out = {}
    for sub in by:
        n_eff = (res.get("n-samples", {}).get(sub) or {}).get("effective"); assert n_eff == len(by[sub]), (sub, n_eff, len(by[sub]))
        for s in res["samples"][sub]:
            g = back[(sub, s["doc_id"])]; f = s["filter"]; o = out.setdefault(g, dict(subtask=sub))
            fr = s["filtered_resps"][0] if isinstance(s["filtered_resps"], list) else s["filtered_resps"]
            if isinstance(fr, list): fr = fr[0] if fr else ""
            o[f"acc_{f}"] = int(float(s["exact_match"])); o[f"invalid_{f}"] = int(str(fr).strip() == "[invalid]"); o[f"pred_{f}"] = str(fr)[:80]   # "[invalid]" = the RegexFilter fallback ("invalid" is a real formal_fallacies answer)
            o["target"] = str(s.get("target"))[:80]
    import re
    for g, o in out.items():
        o["acc"] = o[f"acc_{prim}"]; o["invalid"] = o[f"invalid_{prim}"]
        if a.task == "mmlupro":   # DIAGNOSTIC only (not the harness number): a \boxed{X} letter where the stock regex found nothing
            bx = re.findall(r"\\boxed\{\s*\(?([A-J])\)?\s*\}", texts[g])
            o["acc_lenient_boxed"] = o["acc"] if not o["invalid"] else int(bool(bx) and bx[-1] == str(RAW[g]["doc"]["answer"]).strip())
    assert len(out) == len(texts), (len(out), len(texts))
    return out, "acc"


SCORE = {"math500": score_math, "mbpp": score_mbpp, "bbh": score_replay, "mmlupro": score_replay}[a.task]

# ------------------------------------------------------------------------------------------------ self-tests
if a.selftest:
    if a.task == "math500":
        from lm_eval.tasks.minerva_math import utils as MU
        gold = {g: MU.remove_boxed(MU.last_boxed_only_string(r["doc"]["solution"])) for g, r in RAW.items()}
        fmts = {"boxed_in_prose": "We work it out step by step and get $\\boxed{%s}$.",
                "minerva_sentence": "We work it out step by step.\nFinal Answer: The final answer is $%s$. I hope it is correct.",
                "bare": "%s", "answer_line": "We work it out step by step.\n\nAnswer: %s", "reference_solution": None,
                "NEG_shifted_gold_boxed": "shift", "NEG_shifted_gold_answer_line": "shiftal"}
        R = {}
        for name, f in fmts.items():
            gs_ = sorted(RAW); sh = {g: gold[gs_[(i + 1) % len(gs_)]] for i, g in enumerate(gs_)}   # NEG: the next doc's gold (false-positive rate)
            texts = {g: (RAW[g]["doc"]["solution"] if f is None else "We get $\\boxed{%s}$." % sh[g] if f == "shift" else "Answer: %s" % sh[g] if f == "shiftal" else f % gold[g]) for g in RAW}
            per, _ = score_math(texts); n = len(per)
            R[name] = {k: sum(p[k] for p in per.values()) / n for k in ("math_verify", "mv_lenient", "mv_answerline", "exact_match", "has_minerva", "has_boxed", "mv_extract_fail")}
            R[name]["mv_fail_gids"] = [g for g, p in per.items() if not p["math_verify"]][:40]; R[name]["lenient_fail_gids"] = [g for g, p in per.items() if not p["mv_lenient"]][:40]
            print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in R[name].items()}), flush=True)
        if a.out: json.dump(R, open(a.out if os.path.isabs(a.out) else f"{EX}/{a.out}", "w"), indent=1)
    elif a.task == "mbpp":
        from lm_eval.tasks.mbpp import utils as BU
        gs = sorted(RAW); refs = [mbpp_ref(RAW[g]["doc"]) for g in gs]
        R = {}
        for name, mk in {"stock_on_continuation": lambda c: BU.extract_code_blocks(c + "\n```"),
                         "fixed_on_continuation": lambda c: BU.extract_code_blocks("python\n" + c + "\n```"),
                         "raw_reference": lambda c: c}.items():
            p = code_eval(refs, [mk(RAW[g]["doc"]["code"]) for g in gs])
            R[name] = dict(pass_rate=sum(p) / len(p), n=len(p), fail_gids=[g for g, x in zip(gs, p) if not x])
            print(name, R[name]["pass_rate"], "fails:", R[name]["fail_gids"][:30], flush=True)
        if a.out: json.dump(R, open(a.out if os.path.isabs(a.out) else f"{EX}/{a.out}", "w"), indent=1)
    else:
        raise SystemExit("selftest only for math500 / mbpp")
    raise SystemExit

# ------------------------------------------------------------------------------------------------ score result files
files = []
for g in a.glob: files += sorted(glob.glob(g if os.path.isabs(g) else f"{EX}/{g}"))
MODES, SUMM, SRC = {}, {}, {}
for f in files:
    R = json.load(open(f))
    for key, recs in R["modes"].items():
        m = key.split("@")[0]
        for r in recs: MODES.setdefault(m, {})[r["doc_id"]] = r
        SUMM.setdefault(m, []).append(dict(R["summary"].get(key, {}), file=os.path.basename(f)))
        SRC.setdefault(m, set()).add(os.path.basename(f))
    for key, au in (R.get("audit") or {}).items():
        SUMM.setdefault(key.split("@")[0] + "#audit", []).append(dict(tokens=au["tokens"], diff=au["diff"]))
OUT = dict(task=a.task, lm_eval_task=LMT, cap=a.cap, files=[os.path.relpath(f, EX) for f in files], modes={}, per_doc={})
for m, recs in MODES.items():
    texts = {g: text_of(r) for g, r in recs.items()}
    per, prim = SCORE(texts)
    ids = sorted(per); n = len(ids); k = sum(per[g][prim] for g in ids)
    lens = [recs[g]["tokens_eff"] for g in ids]; capv = a.cap or None
    d = dict(n=n, primary=prim, acc=k / n, ci95=wilson(k, n), mean_len=st.mean(lens), p95_len=sorted(lens)[int(.95 * (n - 1))],
             finish={f_: sum(1 for g in ids if recs[g]["finish"] == f_) for f_ in ("eos", "stop", "length")}, sources=sorted(SRC[m]))
    for c in sorted({1024, 2048} | ({capv} if capv else set())):
        d[f"cap_hit_{c}"] = sum(1 for x in lens if x >= c) / n
    for key in per[ids[0]]:
        if key in ("subtask", "target") or key.startswith("pred_"): continue
        d[key] = sum(per[g][key] for g in ids) / n
    if a.task in ("bbh", "mmlupro"):
        subs = {}
        for g in ids: subs.setdefault(per[g]["subtask"], []).append(per[g])
        d["per_subtask"] = {s: dict(n=len(v), acc=sum(x["acc"] for x in v) / len(v), invalid=sum(x["invalid"] for x in v) / len(v)) for s, v in sorted(subs.items())}
    sm = SUMM.get(m, [])
    if sm:
        tok = sum(x.get("tokens_eff", 0) for x in sm); sec = sum(x.get("sec", 0) for x in sm)
        d["speed"] = dict(tok_eff_per_s=tok / max(1e-9, sec), per_file=[{k_: x.get(k_) for k_ in ("file", "docs", "tok_eff_per_s", "decode_tok_per_s", "speedup_vs_ar", "decode_speedup_vs_ar", "tok_per_step", "ms_step_median")} for x in sm])
    au = SUMM.get(m + "#audit")
    if au: d["audit_share"] = sum(x["diff"] for x in au) / max(1, sum(x["tokens"] for x in au))
    OUT["modes"][m] = d; OUT["per_doc"][m] = {str(g): dict(per[g], len=recs[g]["tokens_eff"], finish=recs[g]["finish"]) for g in ids}
if "ar" in OUT["per_doc"]:
    A = OUT["per_doc"]["ar"]; prim = OUT["modes"]["ar"]["primary"]
    for m, P_ in OUT["per_doc"].items():
        if m == "ar": continue
        ids = sorted(set(A) & set(P_), key=int); b, c, p = mcnemar([P_[i][prim] for i in ids], [A[i][prim] for i in ids])
        OUT["modes"][m]["vs_ar"] = dict(n=len(ids), mode_only=b, ar_only=c, p_mcnemar=p, delta=(b - c) / max(1, len(ids)))
for m in sorted(OUT["modes"], key=lambda k: (k != "ar", k)):
    d = OUT["modes"][m]
    print(f"{m:16s} n={d['n']:5d} {d['primary']}={d['acc']:.4f} [{d['ci95'][0]:.3f},{d['ci95'][1]:.3f}] len={d['mean_len']:.0f} cap1024={d['cap_hit_1024']:.3f} "
          + " ".join(f"{k_}={d[k_]:.3f}" for k_ in d if isinstance(d[k_], float) and k_ not in ("acc", "mean_len") and not k_.startswith("cap_hit"))
          + (f" | vs ar +{d['vs_ar']['mode_only']}/-{d['vs_ar']['ar_only']} p={d['vs_ar']['p_mcnemar']:.3f}" if "vs_ar" in d else "")
          + (f" | tok/s {d['speed']['tok_eff_per_s']:.1f}" if "speed" in d else ""))
if a.out: json.dump(OUT, open(a.out if os.path.isabs(a.out) else f"{EX}/{a.out}", "w"), indent=1, default=str)
