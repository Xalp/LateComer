"""compare.py -- token identity of two decoders' per-doc records (run_late.py / client_late.py / equiv format), up to each doc's stop point.

  python tests/compare.py --a <ours.json> --akey <mode key> --b <late2 results json> --bkey 'l2:lkv:am:latt:4@paged' [--docs 0:100]

Per doc: identical iff gen[:tokens_eff] is equal; else the first differing index (and both tokens). Reports identical share,
tokens per step of both, and the decode tok/s of both files' summaries when present.
"""
import argparse, json, statistics
ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True); ap.add_argument("--akey", required=True); ap.add_argument("--b", required=True); ap.add_argument("--bkey", required=True)
ap.add_argument("--docs", default="0:100000"); ap.add_argument("--out", default="")
x = ap.parse_args()
d0, d1 = map(int, x.docs.split(":"))
A = {r["doc_id"]: r for r in json.load(open(x.a))["modes"][x.akey] if d0 <= r["doc_id"] < d1}
JB = json.load(open(x.b)); B = {r["doc_id"]: r for r in JB["modes"][x.bkey] if d0 <= r["doc_id"] < d1}
ids = sorted(set(A) & set(B)); same = 0; div = []
for i in ids:
    ga, gb = A[i]["gen"][: A[i]["tokens_eff"]], B[i]["gen"][: B[i]["tokens_eff"]]
    if ga == gb: same += 1; continue
    k = next((j for j in range(min(len(ga), len(gb))) if ga[j] != gb[j]), min(len(ga), len(gb)))
    div.append(dict(doc=i, first=k, a=ga[k] if k < len(ga) else None, b=gb[k] if k < len(gb) else None, len=[len(ga), len(gb)]))
tps = lambda R: sum(r["tokens"] for r in R.values()) / max(1, sum(r.get("steps", 0) for r in R.values()))
res = dict(n=len(ids), identical=same, share=same / max(1, len(ids)), divergences=div, tok_per_step=[tps(A), tps(B)])
print(f"{x.akey} vs {x.bkey}: {same}/{len(ids)} docs identical up to the stop point ({res['share']:.3f}); tok/step {res['tok_per_step'][0]:.3f} vs {res['tok_per_step'][1]:.3f}")
for dv in div[:30]: print("  diverge", dv)
if x.out: json.dump(res, open(x.out, "w"), indent=1)
