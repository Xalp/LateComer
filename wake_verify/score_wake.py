"""Score Ouro-2.6B and Huginn-3.5B Wake verification outputs across GSM8K, MATH500, IFEval, and MBPP.

Example:
  PYTHONPATH=. python wake_verify/score_wake.py \
    --results-dir results/wake_runs \
    --ouro-prompts-dir prompts \
    --huginn-tasks-dir data/tasks \
    --out results/WAKE_SCORED.json
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def sign_test_2sided(wins: int, losses: int) -> float:
    from math import comb
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * p)


def main():
    ap = argparse.ArgumentParser(description="Score Wake verification JSON runs.")
    ap.add_argument("--results-dir", required=True, help="Directory containing Ouro/Huginn Wake JSON outputs")
    ap.add_argument("--huginn-tasks-dir", default="data/tasks", help="Directory with Huginn <task>.jsonl files")
    ap.add_argument("--out", default="wake_verify/results/WAKE_SCORED.json", help="Output summary JSON path")
    args = ap.parse_args()

    from huginn.eval.score_hg_formal import load_task_meta, score_texts

    summary = {"huginn": {}, "ouro_files": sorted(glob.glob(os.path.join(args.results_dir, "ouro_*.json")))}
    for task in ("gsm8k", "ifeval", "mbpp"):
        tpath = os.path.join(args.huginn_tasks_dir, f"{task}.jsonl")
        if not os.path.exists(tpath):
            continue
        meta = load_task_meta(task, tpath)
        summary["huginn"][task] = {}
        for jpath in sorted(glob.glob(os.path.join(args.results_dir, f"hg_*_{task}*.json"))):
            data = json.load(open(jpath))
            cfgs = data.get("configs", {})
            for cfg, rec in cfgs.items():
                texts = {int(r["doc_id"]): r["text"] for r in rec.get("records", [])}
                if not texts:
                    continue
                scores, prim = score_texts(task, texts, meta)
                acc = sum(scores[d][prim] for d in scores) / len(scores)
                summary["huginn"][task][cfg] = {
                    "n": len(scores),
                    "metric": prim,
                    "acc": round(acc, 4),
                    "decode_tok_s": rec.get("summary", {}).get("decode_tok_s"),
                    "wake_rollbacks_per_100_tok": rec.get("summary", {}).get("wake_rollbacks_per_100_tok", 0.0),
                }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote scored Wake summary to {args.out}")


if __name__ == "__main__":
    main()
