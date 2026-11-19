"""make_results_la.py -- aggregate the LoopAhead-repro measurement into one table, verifying n and done flags.

  python3 make_results_la.py --tag fair120 [--hfar-tag fair120] [--n 120] > results/RESULTS.md
"""
import argparse, glob, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--hfar-tag", default=None)
ap.add_argument("--n", type=int, default=120)
ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
a = ap.parse_args()
TASKS = ["gsm8k", "ifeval", "math500", "mbpp", "bbh", "mmlupro"]

rows, problems = [], []
for task in TASKS:
    f = f"{a.dir}/{a.tag}_{task}.json"
    if not os.path.exists(f):
        problems.append(f"MISSING {f}"); continue
    d = json.load(open(f))
    if not d["meta"].get("done"): problems.append(f"{task}: meta.done false")
    n = d["meta"].get("n_docs")
    if n != a.n: problems.append(f"{task}: n_docs {n} != {a.n}")
    for mode, recs in d["modes"].items():
        if len(recs) != a.n: problems.append(f"{task}/{mode}: {len(recs)} recs != {a.n}")
    hf = None
    if a.hfar_tag:
        fh = f"{a.dir}/{a.hfar_tag}_hfar_{task}.json"
        if os.path.exists(fh):
            h = json.load(open(fh))
            if not h["meta"].get("done"): problems.append(f"hfar {task}: done false")
            if len(h["recs"]) != a.n: problems.append(f"hfar {task}: {len(h['recs'])} != {a.n}")
            hf = h.get("summary")
    s = d["summary"]
    ar = s.get("ar"); la = s.get("la"); nl = s.get("la:nolad")
    au = d.get("audit", {})
    row = dict(task=task, ar=ar, la=la, nolad=nl, hf=hf,
               audit_la=au.get("la", {}).get("share"), audit_ar=au.get("ar", {}).get("share"),
               checks=d.get("checks", {}))
    rows.append(row)

print(f"# LoopAhead -- OUR reimplementation -- matched {a.n}-document protocol (released Ouro-2.6B, batch 1, greedy, bf16, one GPU)\n")
print("All numbers from one process per task: engine-AR first, then the repro, same GPU (tag "
      f"`{a.tag}`). HF-AR is a separate job (tag `{a.hfar_tag}`). 'repro/engineAR' uses OUR engine's AR as the "
      "denominator; 'repro/hfAR' shows the same repro over a plain HuggingFace AR denominator. Speeds are "
      "tokens_eff / wall seconds including prefill (protocol standard); decode-only in parentheses.\n")
print("| task | engine AR tok/s | LA-repro tok/s | repro/engineAR | no-LAD/engineAR | HF AR tok/s | repro/hfAR | tok/step | emits n/l/b | audit diff |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    ar, la, nl, hf = r["ar"], r["la"], r["nolad"], r["hf"]
    def f1(x): return f"{x:.1f}" if x is not None else "-"
    def f2(x): return f"{x:.2f}x" if x is not None else "-"
    sp = la["tok_eff_per_s"] / ar["tok_eff_per_s"] if (la and ar) else None
    spd = la["decode_tok_per_s"] / ar["decode_tok_per_s"] if (la and ar) else None
    spn = nl["tok_eff_per_s"] / ar["tok_eff_per_s"] if (nl and ar) else None
    sph = la["tok_eff_per_s"] / hf["tok_eff_per_s"] if (la and hf) else None
    em = f"{la['emit_native']}/{la['emit_lad']}/{la['emit_bonus']}" if la else "-"
    print(f"| {r['task']} | {f1(ar and ar['tok_eff_per_s'])} ({f1(ar and ar['decode_tok_per_s'])}) "
          f"| {f1(la and la['tok_eff_per_s'])} ({f1(la and la['decode_tok_per_s'])}) "
          f"| {f2(sp)} ({f2(spd)}) | {f2(spn)} | {f1(hf and hf['tok_eff_per_s'])} | {f2(sph)} "
          f"| {la['tok_per_step']:.2f} | {em} | {r['audit_la'] if r['audit_la'] is not None else '-'} |" if la or ar else f"| {r['task']} | no data |")
print()
if problems:
    print("## PROBLEMS (fix before trusting the table)\n")
    for p in problems: print("-", p)
    sys.exit(1)
print("All document counts and done flags verified.")
