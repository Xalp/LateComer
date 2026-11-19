"""make_headtohead_full.py -- aggregate the FULL-test-set head-to-head (Table 6 upgrade).

All six benchmarks, full sets (21,403 documents). Per task:
  ours      : full-depth AR + K=4/3/2 (B=8), pooled over the eval-campaign shards
  LoopSpec  : baseline (their full-depth AR) + config 1:2, pooled over lsfull_* shards
  LoopAhead : engine AR pooled over every lafull_/lavfull_ job; best branch =
              max(faithful la, favourable la, no-drafter) by pooled decode rate
  denom     : max(ours_ar, loopspec_ar, loopahead_ar)  [decode tokens/s]
Decode-rate pooling is exact: every file stores its own decode rate and decode-token
count, so pooled rate = sum(tokens_d) / sum(tokens_d / rate).

  python3 make_headtohead_full.py [--partial]        # --partial: report holes, no assert
"""
import argparse, glob, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--partial", action="store_true")
ap.add_argument("--out", default="results/HEADTOHEAD_FULL")
a = ap.parse_args()

HD = os.environ.get("LE_DATA_ROOT", ".")
N_EXPECT = dict(gsm8k=1319, ifeval=541, math500=500, mbpp=500, bbh=6511, mmlupro=12032)
TASKS = list(N_EXPECT)
problems = []


def note(msg):
    problems.append(msg)
    print("[!]", msg)


# ---------------- ours: run_latek-style jsons ----------------
def pool_latek(files, want_modes):
    """returns {short_mode: dict(n, tok_d, sec_d, tokens, docs)} pooled."""
    agg = {}
    for f in files:
        d = json.load(open(f))
        for key, s in d.get("summary", {}).items():
            if not isinstance(s, dict) or "decode_tok_per_s" not in s:
                continue
            short = key.split("@")[0]
            if short not in want_modes:
                continue
            docs = s["docs"]; tok_d = s["tokens_eff"] - docs
            sec_d = tok_d / s["decode_tok_per_s"]
            g = agg.setdefault(short, dict(n=0, tok_d=0, sec_d=0.0, tokens=0))
            g["n"] += docs; g["tok_d"] += tok_d; g["sec_d"] += sec_d; g["tokens"] += s["tokens_eff"]
    return agg


OUR_MODES = {"ar": "ar", "lk:4:latt:8": "k4", "lk:3:latt:8": "k3", "lk:2:latt:8": "k2"}
ours = {}
for t in TASKS:
    if t in ("gsm8k", "ifeval"):
        pat = "gsm" if t == "gsm8k" else "ife"
        files = glob.glob(f"{HD}/decode/late_k/results/lk_s*_{pat}_*.json") \
              + glob.glob(f"{HD}/decode/late_k/results/k4full_{pat}_*.json")
    else:
        files = glob.glob(f"{HD}/decode/evalx/results/full_{t}_s*_{t}_*.json")
    agg = pool_latek(files, OUR_MODES)
    ours[t] = {OUR_MODES[k]: v for k, v in agg.items()}
    for m, v in ours[t].items():
        if v["n"] % N_EXPECT[t] != 0:  # grid campaigns may hold one pass per mode
            note(f"ours {t}/{m}: pooled docs {v['n']} not a multiple of {N_EXPECT[t]}")

# ---------------- LoopSpec ----------------
ls = {}
for t in TASKS:
    for cfg, label in (("baseline", "ar"), ("1_2", "best")):
        tot_tok = tot_sec = tot_n = 0
        for f in sorted(glob.glob(f"{HD}/baselines/loopspec/results/lsfull_*/v2/{cfg}/metrics_{t}.json")):
            d = json.load(open(f))
            tot_tok += d["decode_tokens"]; tot_sec += d["decode_seconds"]; tot_n += d["n"]
        if tot_n != N_EXPECT[t]:
            note(f"loopspec {t}/{cfg}: n {tot_n} != {N_EXPECT[t]}")
        ls.setdefault(t, {})[label] = dict(n=tot_n, tok_d=tot_tok, sec_d=tot_sec)

# ---------------- LoopAhead ----------------
la = {}
for t in TASKS:
    arms = {}
    for f in sorted(glob.glob(f"{HD}/baselines/loopahead_repro/results/la*full_{t}*_{t}.json")
                    + glob.glob(f"{HD}/baselines/loopahead_repro/results/la*full_{t}.json")
                    + glob.glob(f"{HD}/baselines/loopahead_repro/results/lare*_{t}_s*_{t}.json")):
        base = os.path.basename(f)
        fav = base.startswith("lavfull")
        d = json.load(open(f))
        if not d.get("meta", {}).get("done", True):
            # per-mode summaries are saved incrementally; keep what finished, the n
            # checks below catch any mode that is genuinely short.
            note(f"loopahead {base}: meta.done false (keeping completed modes: "
                 f"{sorted(d.get('summary', {}).keys())})")
        for mode, s in d.get("summary", {}).items():
            if not isinstance(s, dict):
                continue
            label = {"ar": "ar", "la": ("la_fav" if fav else "la"), "la:nolad": "nolad"}.get(mode)
            if label is None:
                continue
            docs = s["docs"]; tok_d = s["tokens"] - docs
            sec_d = tok_d / s["decode_tok_per_s"]
            g = arms.setdefault(label, dict(n=0, tok_d=0, sec_d=0.0))
            g["n"] += docs; g["tok_d"] += tok_d; g["sec_d"] += sec_d
    for lab, v in arms.items():
        want = N_EXPECT[t] * (2 if lab == "ar" else 1)   # ar rides in both arms' jobs
        if v["n"] != want:
            note(f"loopahead {t}/{lab}: n {v['n']} != {want}")
    la[t] = arms

# ---------------- assemble ----------------
rate = lambda v: v["tok_d"] / v["sec_d"] if v and v["sec_d"] else None
out = {}
for t in TASKS:
    o = ours.get(t, {}); l = ls.get(t, {}); h = la.get(t, {})
    ours_ar = rate(o.get("ar")); ls_ar = rate(l.get("ar")); la_ar = rate(h.get("ar"))
    cand = {"ours": ours_ar, "loopspec": ls_ar, "loopahead": la_ar}
    have = {k: v for k, v in cand.items() if v}
    if len(have) < 3 and not a.partial:
        note(f"{t}: missing AR arms {sorted(set(cand) - set(have))}")
    denom_is, denom = (max(have.items(), key=lambda kv: kv[1]) if have else (None, None))
    branches = {k: rate(h.get(k)) for k in ("la", "la_fav", "nolad")}
    la_best_name, la_best = (None, None)
    bb = {k: v for k, v in branches.items() if v}
    if bb:
        la_best_name, la_best = max(bb.items(), key=lambda kv: kv[1])
    row = dict(
        n=N_EXPECT[t],
        ours_ar=ours_ar, ls_ar=ls_ar, la_ar=la_ar,
        ours_k4=rate(o.get("k4")), ours_k3=rate(o.get("k3")), ours_k2=rate(o.get("k2")),
        ls_best=rate(l.get("best")), la_branches=branches, la_best=la_best, la_best_arm=la_best_name,
        mean_gen_len=(o.get("ar", {}).get("tokens", 0) / max(1, o.get("ar", {}).get("n", 1))) if o.get("ar") else None,
        denom=denom, denom_is=denom_is,
    )
    if denom:
        for k in ("ours_k4", "ours_k3", "ours_k2", "ls_best", "la_best"):
            row["r_" + k.replace("ours_", "")] = (row[k] / denom) if row[k] else None
    out[t] = row

summary = dict(tasks=out, problems=problems, n_total=sum(N_EXPECT.values()),
               protocol="decode tokens/s; denom = max(full-depth AR of ours, LoopSpec, LoopAhead) per task; "
                        "shards pooled as sum(decode tokens)/sum(decode seconds); batch 1, greedy, bf16, one GPU")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(summary, open(a.out + ".json", "w"), indent=1)

md = ["# Head-to-head, FULL test sets (21,403 documents)\n",
      "| task | n | len | ours AR | LS AR | LA AR | denom(src) | LS 1:2 | LA best(arm) | K=4 | K=3 | K=2 |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
fmt = lambda x, p=1: ("" if x is None else f"{x:.{p}f}")
for t, r in out.items():
    md.append(f"| {t} | {r['n']} | {fmt(r['mean_gen_len'],0)} | {fmt(r['ours_ar'])} | {fmt(r['ls_ar'])} | {fmt(r['la_ar'])} "
              f"| {fmt(r['denom'])} ({r['denom_is']}) | {fmt(r['ls_best'])} ({fmt(r.get('r_ls_best'),2)}x) "
              f"| {fmt(r['la_best'])} ({fmt(r.get('r_la_best'),2)}x, {r['la_best_arm']}) "
              f"| {fmt(r.get('r_k4'),2)}x | {fmt(r.get('r_k3'),2)}x | {fmt(r.get('r_k2'),2)}x |")
md.append("\nproblems: " + ("none" if not problems else "; ".join(problems)))
open(a.out + ".md", "w").write("\n".join(md) + "\n")
print("\n".join(md[1:]))
print("wrote", a.out + ".json/.md", "| problems:", len(problems))
