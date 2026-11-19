"""score_tasks.py (late2_full copy of ../gate_relaxed/score_tasks.py; changes: --seed N (python random + langdetect seed per IFEval doc,
default 0 = the gate_relaxed behaviour) and OUT["per_doc_all"] = every per-doc metric; earlier: gate_relaxed copy of ../score_tasks.py: data path -> ../data, mode sort by name, wider label) -- lm-eval accuracy for decode run.py outputs (CPU; python, lm_eval 0.4.12).

  python score_tasks.py --task gsm8k  --glob 'results/full_s*_full_*.json' 'results/p3_gsm_s*.json' --out results/score_gsm8k.json
  python score_tasks.py --task ifeval --glob 'results/p3_ife_s*.json' --out results/score_ifeval.json
  python score_tasks.py --task ifeval --selftest     # scorer check: the campaign's HF ar:4 IFEval texts must give its .704

GSM8K: text cut at the first stop string of the doc (harness convention, as lmeval_ouro/generate.py), then lm-eval's own
flexible-extract / strict-match filters + exact_match (identical to baselines/loopspec/summarize.py), gold = the `target` of
LoopSpec's gsm8k_chat0_v2 prompt file (same docs/contexts as the campaign's gsm8k_cot_zeroshot).
IFEval: lm_eval.tasks.ifeval.utils.process_results(doc, [text]) per doc (the task's own metric), docs = the harness docs logged in the
campaign's v2/ifeval/ar_4/score/samples.jsonl; langdetect seeded (the harness leaves it unseeded -> 0-4 docs of re-check noise).
Several files may hold the same mode (shards, or repeated measurements of the same docs): per mode, the LAST file (sorted glob order
within each --glob argument, arguments in the given order) wins per doc_id. Paired exact McNemar of every mode vs `ar` on common docs.
"""
import argparse, glob, json, math, os, statistics as st
ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=["gsm8k", "ifeval"]); ap.add_argument("--glob", nargs="*", default=[])
ap.add_argument("--out", default=""); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
ME = os.path.dirname(os.path.abspath(__file__))
CAMP = os.environ.get("LE_CAMPAIGN_DIR", "")  # directory with the task gold files (see README)


def mcnemar(x, y):
    """exact two-sided McNemar on paired 0/1 lists: (b = x1 y0, c = x0 y1, p)"""
    b = sum(1 for u, v in zip(x, y) if u and not v); c = sum(1 for u, v in zip(x, y) if v and not u); n = b + c
    if n == 0: return b, c, 1.0
    p = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return b, c, min(1.0, 2 * p)


if a.task == "gsm8k":
    from lm_eval.filters.extraction import RegexFilter
    from lm_eval.api.metrics import exact_match_hf_evaluate
    FLEX = RegexFilter(regex_pattern=r"(-?[$0-9.,]{2,})|(-?[0-9]+)", group_select=-1)
    STRICT = RegexFilter(regex_pattern=r"The answer is (\-?[0-9\.\,]+).", group_select=0)
    IGN = [",", "\\$", "(?s).*#### ", "\\.$"]
    GOLD = {}; UNTIL = {}
    for l in open(os.environ["LE_GSM8K_PROMPTS"]):  # prompts jsonl with gold answers (see README)
        d = json.loads(l); GOLD[d["doc_id"]] = d["target"]
    for l in open(f"{ME}/../data/gsm8k_v2_prompts.jsonl"):
        d = json.loads(l); UNTIL[d["doc_id"]] = d["until"]

    def em(p, g): return exact_match_hf_evaluate([p], [g], regexes_to_ignore=IGN, ignore_case=True, ignore_punctuation=False)["exact_match"]

    def score_doc(doc_id, text):
        for u in UNTIL[doc_id]:
            if u and u in text: text = text.split(u)[0]
        f = FLEX.apply([[text]], None)[0][0]; s = STRICT.apply([[text]], None)[0][0]
        return dict(flex=int(em(f, GOLD[doc_id])), strict=int(em(s, GOLD[doc_id])))
    PRIMARY = "flex"
else:
    import ast
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0
    from lm_eval.tasks.ifeval.utils import process_results
    DOCS = {}
    for l in open(f"{CAMP}/ifeval/ar_4/score/samples.jsonl"):
        r = json.loads(l); d = dict(r["doc"])
        for k in ("instruction_id_list", "kwargs"):
            if isinstance(d[k], str): d[k] = ast.literal_eval(d[k])
        DOCS[r["doc_id"]] = d

    import random

    def score_doc(doc_id, text):
        # verification 10-27: lm-eval's LetterFrequencyChecker draws a RANDOM letter (unseeded python random) when the doc's letter is not
        # a-z (docs 19 '#', 22 '!'), so identical texts could score differently (sp:13 vs gp:inf .739 vs .738). Seed python random AND
        # langdetect before every doc -> identical texts always get identical scores.
        random.seed(a.seed); DetectorFactory.seed = a.seed
        r = process_results(DOCS[doc_id], [text])
        return dict(strict=int(r["prompt_level_strict_acc"]), loose=int(r["prompt_level_loose_acc"]),
                    inst_strict=[int(x) for x in r["inst_level_strict_acc"]], inst_loose=[int(x) for x in r["inst_level_loose_acc"]])
    PRIMARY = "strict"


def agg(per):
    ids = sorted(per); out = dict(n=len(ids))
    for k in per[ids[0]]:
        v = [per[i][k] for i in ids]
        out[k] = st.mean(x for xs in v for x in xs) if isinstance(v[0], list) else st.mean(v)
    return out


if a.selftest:
    assert a.task == "ifeval"
    texts = {}
    for f in sorted(glob.glob(f"{CAMP}/ifeval/ar_4/shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r.get("finish") != "error": texts[r["doc_id"]] = r["text"]
    per = {i: score_doc(i, t) for i, t in texts.items()}
    s = agg(per); camp = json.load(open(f"{CAMP}/ifeval/ar_4/score/summary.json"))["metrics"]
    print(json.dumps(dict(mine=s, campaign={k: camp[k] for k in camp if "stderr" not in k}), indent=1)); raise SystemExit

files = []
for g in a.glob: files += sorted(glob.glob(g if os.path.isabs(g) else f"{ME}/{g}"))
MODES = {}; SRC = {}
for f in files:
    R = json.load(open(f))
    for key, recs in R["modes"].items():
        m = key.split("@")[0]
        for r in recs:
            MODES.setdefault(m, {})[r["doc_id"]] = r["text"]; SRC.setdefault(m, {})[r["doc_id"]] = os.path.basename(f)
OUT = dict(task=a.task, files=[os.path.relpath(f, ME) for f in files], primary=PRIMARY, modes={}, per_doc={}, per_doc_all={}, seed=a.seed)
for m, texts in MODES.items():
    per = {i: score_doc(i, t) for i, t in texts.items()}
    OUT["modes"][m] = agg(per); OUT["per_doc"][m] = {str(i): per[i][PRIMARY] for i in sorted(per)}
    OUT["per_doc_all"][m] = {str(i): per[i] for i in sorted(per)}
    OUT["modes"][m]["sources"] = sorted(set(SRC[m].values()))
if "ar" in OUT["per_doc"]:
    A = OUT["per_doc"]["ar"]
    for m, P_ in OUT["per_doc"].items():
        if m == "ar": continue
        ids = sorted(set(A) & set(P_), key=int); b, c, p = mcnemar([P_[i] for i in ids], [A[i] for i in ids])
        OUT["modes"][m]["vs_ar"] = dict(n=len(ids), mode_only=b, ar_only=c, p_mcnemar=p, delta=(b - c) / max(1, len(ids)))
for m in sorted(OUT["modes"], key=lambda k: (k != "ar", k)):
    d = OUT["modes"][m]
    print(f"{m:40s} n={d['n']:5d} " + " ".join(f"{k}={d[k]:.4f}" for k in d if isinstance(d[k], float)) + (f"  vs ar: +{d['vs_ar']['mode_only']}/-{d['vs_ar']['ar_only']} p={d['vs_ar']['p_mcnemar']:.3f}" if "vs_ar" in d else ""))
if a.out: json.dump(OUT, open(a.out if os.path.isabs(a.out) else f"{ME}/{a.out}", "w"), indent=1)
