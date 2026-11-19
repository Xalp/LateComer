"""prep_x.py -- evalx: materialized lm-eval contexts (data/raw/<task>.jsonl from materialize_x.py) -> engine prompt files.

    python prep_x.py --task minerva_math500 --name math500 --S 2560 --max-new 1024 [--model <dir>]

Writes data/<name>_prompts.jsonl  {doc_id (= global id gid, 0..N-1), subtask, sub_doc_id, ids, until, context_sha}
       data/meta_<name>_prompts.json  {nv, eos, mask_id (what run_latek.py reads), task, variant, num_fewshot, apply_chat_template,
                                      model, tokenizer sha, chat_template_sha, S, max_new, prompt length stats, ...}
Tokenization = tok(context, add_special_tokens=False) exactly as lmeval_ouro/generate.py and ../prep_prompts_task.py.
Model-independence: the CONTEXT strings (raw/<task>.jsonl) are rendered by the model's chat template; to run another checkpoint
with the same template, change --model (one line in the wrappers: MODEL_DIR); with a different template, rerun materialize_x.py --model.
The script refuses a (S, max_new) pair that the engine would reject: S >= max prompt + max_new + 128 (lattice rows NB + B + 8 < 128).
"""
import argparse, hashlib, json, os, statistics as st
import transformers
ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--model", required=True, help="path to the fine-tuned model directory")
ap.add_argument("--S", type=int, required=True); ap.add_argument("--max-new", type=int, required=True)
ap.add_argument("--until-override", default=None, help="JSON list replacing the harness until strings (not used by default)")
a = ap.parse_args()
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
src = f"{D}/raw/{a.task}.jsonl"; rmeta = json.load(open(f"{D}/raw/{a.task}.meta.json"))
tok = transformers.AutoTokenizer.from_pretrained(a.model)
tsha = hashlib.sha256(open(os.path.join(a.model, "tokenizer.json"), "rb").read()).hexdigest()[:16]
lens = []; n = 0; subs = {}
out = f"{D}/{a.name}_prompts.jsonl"
with open(out + ".tmp", "w") as f:
    for l in open(src):
        d = json.loads(l); ids = tok(d["context"], add_special_tokens=False)["input_ids"]; lens.append(len(ids)); n += 1
        until = json.loads(a.until_override) if a.until_override else d["until"]
        subs[d["task"]] = subs.get(d["task"], 0) + 1
        f.write(json.dumps(dict(doc_id=d["gid"], subtask=d["task"], sub_doc_id=d["doc_id"], ids=ids, until=until, context_sha=d["context_sha"])) + "\n")
os.replace(out + ".tmp", out)
s = sorted(lens); q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
assert a.S >= s[-1] + a.max_new + 128, f"S={a.S} too small for max prompt {s[-1]} + max_new {a.max_new} + 128"
meta = dict(nv=len(tok), eos=sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}), mask_id=tok.convert_tokens_to_ids("<|mask|>"),
            task=a.task, variant=a.name, num_fewshot=rmeta["num_fewshot"], apply_chat_template=rmeta["apply_chat_template"],
            lm_eval=rmeta["lm_eval"], chat_template_sha=rmeta["chat_template_sha"], harness_gen_kwargs=rmeta["gen_kwargs"],
            until=sorted({tuple(json.loads(a.until_override)) if a.until_override else u for u in map(tuple, rmeta["until"])}),
            until_overridden=bool(a.until_override), source=src, model=a.model, tokenizer_sha=tsha, transformers=transformers.__version__,
            n=n, n_per_subtask=subs, S=a.S, max_new=a.max_new,
            prompt_len=dict(min=s[0], p50=q(.5), p95=q(.95), max=s[-1], mean=round(st.mean(s), 1)))
json.dump(meta, open(f"{D}/meta_{a.name}_prompts.json", "w"), indent=1)
print(json.dumps({k: meta[k] for k in ("variant", "n", "S", "max_new", "prompt_len", "until", "eos", "mask_id")}))
