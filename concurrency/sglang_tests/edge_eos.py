"""edge_eos.py -- requests whose stop-token set differs from late.py's EOS {0, 2}: ignore_eos (must return exactly max_new tokens, run past
EOS without a lattice/SGLang desync), no stop_token_ids (SGLang's own EOS set), and the normal payload; GSM8K docs 0-4. Exit 1 on failure."""
import json, sys, urllib.request
url, prompts = sys.argv[1], sys.argv[2]
docs = [json.loads(l) for l in open(prompts)][:5]
def post(p):
    r = urllib.request.Request(url + "/generate", data=json.dumps(p).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as f: return json.loads(f.read())
ok = True
for d in docs:
    a = post({"input_ids": d["ids"], "sampling_params": {"temperature": 0, "max_new_tokens": 256, "ignore_eos": True}})
    b = post({"input_ids": d["ids"], "sampling_params": {"temperature": 0, "max_new_tokens": 512}})
    c = post({"input_ids": d["ids"], "sampling_params": {"temperature": 0, "max_new_tokens": 512, "stop_token_ids": [0, 2], "skip_special_tokens": False}})
    na, nb, nc = len(a["output_ids"]), len(b["output_ids"]), len(c["output_ids"])
    fa, fb, fc = (x["meta_info"]["finish_reason"] for x in (a, b, c))
    pre = c["output_ids"] == b["output_ids"][:nc] or b["output_ids"] == c["output_ids"][:nb]
    good = na == 256 and a["output_ids"][:nc] == c["output_ids"][:min(nc, 256)]
    print(f"[edge] doc {d['doc_id']}: ignore_eos {na} tok {fa} | default-stop {nb} tok {fb} | stop[0,2] {nc} tok {fc} | ignore_eos prefix == normal: {good} | default vs [0,2] prefix-consistent {pre}", flush=True)
    ok &= good and pre
print("[edge] RESULT", "OK" if ok else "FAIL"); sys.exit(0 if ok else 1)
