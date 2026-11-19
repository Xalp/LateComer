"""materialize_x.py -- evaluation expansion (evalx): write, per lm-eval task, the exact prompts the harness sends under
--apply_chat_template (0-shot) with a model's tokenizer + the doc payload needed to score them offline.

Copy of MARS/integration/probes/lmeval_ouro/materialize.py (same DumpLM / chat template plumbing from lmo_common.py:
add_generation_prompt or continue_final_message for gen_prefix tasks, enable_thinking=False), extended for GROUP tasks
(bbh_cot_zeroshot, mmlu_pro: many subtasks, each with its own doc ids) and with the scoring doc saved per request:

  out: <out>/raw/<task>.jsonl  one line per request, global id `gid` 0..N-1 in (subtask, doc_id) order:
       {gid, task (subtask), doc_id, context, until, gen_kwargs, gen_prefix, context_sha, doc}
       <out>/raw/<task>.meta.json  (lm_eval version, chat-template sha, num_fewshot, n per subtask, ...)

Run (dev box CPU, lm-eval 0.4.12; antlr4 4.11 side-loaded for minerva_math):
  PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/evalx_pylib HF_DATASETS_CACHE=/tmp/hf_dscache_evalx \
    python materialize_x.py --model <dir> --tasks minerva_math500,bbh_cot_zeroshot,mmlu_pro,mbpp_instruct
Online on the first run (datasets are downloaded into HF_DATASETS_CACHE), offline afterwards.
"""
import argparse, hashlib, json, os, sys, time
sys.path.insert(0, os.environ.get("LE_LMEVAL_DIR", os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1"); os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false"); os.environ.setdefault("HAS_TT_DATA", "False")

import lmo_common as C   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=C.MODELS["v2"]); ap.add_argument("--tasks", required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
ap.add_argument("--num-fewshot", type=int, default=0); ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()
import lm_eval                                                  # noqa: E402
from lm_eval import simple_evaluate                             # noqa: E402


def jsonable(x):
    try:
        json.dumps(x); return x
    except TypeError:
        if isinstance(x, dict): return {k: jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
        return str(x)


_, DumpLM, _ = C.make_lm_classes()


class DumpLMX(DumpLM):
    def generate_until(self, requests, disable_tqdm=False):
        out = []
        for req in requests:
            context, gk = req.args[0], dict(req.args[1] or {}); until = gk.get("until", []) or []
            if isinstance(until, str): until = [until]
            self.records.append(dict(task=req.task_name, doc_id=req.doc_id, context=context, until=list(until),
                                     gen_kwargs={k: v for k, v in gk.items() if k != "until"}, doc=jsonable(req.doc)))
            out.append("")
        return out


odir = f"{a.out}/raw"; os.makedirs(odir, exist_ok=True)
lm = DumpLMX(a.model)
tmpl = lm.tokenizer.chat_template or ""
thash = hashlib.sha256(tmpl.encode()).hexdigest()[:12]
print(f"[mat] lm_eval {getattr(lm_eval, '__version__', '?')} model {a.model} chat-template sha {thash}", flush=True)
for task in a.tasks.split(","):
    lm.records = []; t0 = time.time()
    simple_evaluate(model=lm, tasks=[task], num_fewshot=a.num_fewshot, batch_size=1, apply_chat_template=True, limit=a.limit, log_samples=True,
                    predict_only=True, confirm_run_unsafe_code=True, bootstrap_iters=0)
    recs = sorted(lm.records, key=lambda r: (r["task"], r["doc_id"]))
    per = {}
    for r in recs: per.setdefault(r["task"], []).append(r["doc_id"])
    for t, ids in per.items(): assert ids == list(range(len(ids))), f"{t}: doc ids not contiguous {ids[:5]}.. {len(ids)}"
    path = f"{odir}/{task}.jsonl"
    with open(path + ".tmp", "w") as f:
        for g, r in enumerate(recs):
            r["gid"] = g; r["context_sha"] = hashlib.sha256(r["context"].encode()).hexdigest()[:16]
            f.write(json.dumps(dict(gid=g, **{k: v for k, v in r.items() if k != "gid"})) + "\n")
    os.replace(path + ".tmp", path)
    meta = dict(task=task, n=len(recs), n_per_subtask={t: len(v) for t, v in per.items()}, model=a.model, chat_template_sha=thash,
                lm_eval=getattr(lm_eval, "__version__", "?"), num_fewshot=a.num_fewshot, apply_chat_template=True,
                until=sorted({tuple(r["until"]) for r in recs}), gen_kwargs=sorted({json.dumps(r["gen_kwargs"], sort_keys=True) for r in recs}),
                limit=a.limit, written=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(meta, open(f"{odir}/{task}.meta.json", "w"), indent=1)
    print(f"[mat] {task}: {len(recs)} docs in {len(per)} subtasks, until={meta['until']} gen_kwargs={meta['gen_kwargs']} ({time.time() - t0:.0f}s) -> {path}", flush=True)
    print("[mat] first context:\n" + recs[0]["context"][-900:].replace("\n", "\\n") + "\n", flush=True)
print("[mat] DONE", flush=True)
