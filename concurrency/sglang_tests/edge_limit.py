"""edge_limit.py -- (review 10-28) a request that runs to the context limit must finish by length without killing the server.

  python tests/edge_limit.py <url> <prompts.jsonl> <context_len>

GSM8K docs 0 and 1 with ignore_eos and max_new_tokens = context_len - 1 - P0 (the largest total SGLang admits). Before the max_new clamp
(spec_worker._len_cap) the LateStepExecutor assert (lc + D + TM < W) fired inside the scheduler, and at stride > 8 prepare_for_decode wrote
past the req_to_token row first. Pass = every request returns output_ids with finish_reason length, and a normal request afterwards
returns the same tokens as before the long one (server alive, state intact). Exit 1 on failure."""
import json, sys, urllib.request
url, prompts, ctx = sys.argv[1], sys.argv[2], int(sys.argv[3])
docs = [json.loads(l) for l in open(prompts)][:2]
def post(p, timeout=900):
    r = urllib.request.Request(url + "/generate", data=json.dumps(p).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f: return json.loads(f.read())
ok = True
norm = {"temperature": 0, "max_new_tokens": 128, "stop_token_ids": [0, 2], "skip_special_tokens": False}
before = post({"input_ids": docs[0]["ids"], "sampling_params": norm})["output_ids"]
for d in docs:
    P0 = len(d["ids"]); mx = ctx - 1 - P0
    try:
        r = post({"input_ids": d["ids"], "sampling_params": {"temperature": 0, "max_new_tokens": mx, "ignore_eos": True}})
        n = len(r["output_ids"]); fr = r["meta_info"]["finish_reason"]
        good = isinstance(fr, dict) and fr.get("type") == "length" and n >= 1
        print(f"[limit] doc {d['doc_id']}: P0 {P0} max_new {mx} -> {n} tokens (P0 + n = {P0 + n}), finish {fr}, ok {good}", flush=True)
    except Exception as e:
        good = False; print(f"[limit] doc {d['doc_id']}: P0 {P0} max_new {mx} -> request FAILED: {e!r}", flush=True)
    ok &= good
try:
    after = post({"input_ids": docs[0]["ids"], "sampling_params": norm})["output_ids"]
    same = after == before
except Exception as e:
    same = False; print(f"[limit] follow-up request FAILED: {e!r}", flush=True)
print(f"[limit] follow-up request identical to the one before: {same}", flush=True)
ok &= same
print("[limit] RESULT", "OK" if ok else "FAIL"); sys.exit(0 if ok else 1)
