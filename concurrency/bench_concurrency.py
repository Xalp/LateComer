"""bench_concurrency.py -- Steady-state continuous-batching concurrency benchmark on Ouro-2.6B (T = 4).

Measures total decode throughput (tok/s), per-request decode speed (tok/s/req & TPOT ms), end-to-end throughput
(including online slot-refill prefills), and step acceptance length across concurrencies C in {1, 2, 4, 8, 16, 32}
for a given decoding mode:
  - 'ar'            : Lossless full-depth (T = 4) continuous-batching sequential decoding (B = 0)
  - 'lk:4:<shape>'  : Lossless full-depth (T = 4) block decoding (e.g. latt:8, latt:4)
  - 'lk:3:<shape>'  : Lossy late-entry (K = 3, passes 2..4) block / pipelined decoding (latt:8, latt:4, latt:2, b0)
  - 'lk:2:<shape>'  : Lossy late-entry (K = 2, passes 3..4) block / pipelined decoding (latt:8, latt:4, latt:2, b0)
"""
import argparse
import json
import os
import socket
import sys
import time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)

from sglang_late.core_ops import OuroCore, FlatKV, load_weights
from sglang_late.shapes import LateShape
from sglang_late.late_step import ARStepExecutor
from batch_latek import LateKBatchExecutor, LateKReqState, SlotPool, parse_conc_mode, prefill_batch


def burn_gpu(seconds: float, dev: str, log):
    if seconds <= 0 or not dev.startswith("cuda"):
        return
    log(f"GPU thermal warmup burn for {seconds:.1f}s ...")
    a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(20):
            torch.matmul(a, b)
        torch.cuda.synchronize()
    del a, b
    torch.cuda.empty_cache()
    log("GPU burn done")


def build_stop_checker(model_dir: str, nv: int, use_until: bool):
    if not use_until:
        return set(), lambda gen, until: len(gen)
    tok_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        return set(), lambda gen, until: len(gen)
    from tokenizers import Tokenizer
    tokzr = Tokenizer.from_file(tok_path)
    # Precompute token IDs whose piece contains any character of 'Q:', '</s>', '<|im_end|>'
    # so 98%+ of decode steps avoid calling Rust tokenizer decode on host CPU.
    trig_chars = ("Q", ":", "<", "/", "|")
    trig_ids = set()
    for tid in range(min(nv, tokzr.get_vocab_size())):
        s = tokzr.decode([tid], skip_special_tokens=False)
        if any(ch in s for ch in trig_chars):
            trig_ids.add(tid)

    def trunc_on_stop(gen, until):
        if not until:
            return len(gen)
        tail = gen[-48:]
        txt = tokzr.decode(tail, skip_special_tokens=False)
        if not any(u in txt for u in until):
            return len(gen)
        # Find exact earliest token prefix where stop string appears
        lo = max(1, len(gen) - 48)
        for k in range(lo, len(gen) + 1):
            sub = tokzr.decode(gen[max(0, k - 48) : k], skip_special_tokens=False)
            if any(u in sub for u in until):
                return k
        return len(gen)

    return trig_ids, trunc_on_stop


@torch.no_grad()
def run_eager_steps(core, pm, kv, pool, W, nv, mask_id, docs, attn):
    """Run 12 eager steps at R=2 (before any graph capture/replay) to reference-check graph output."""
    R = 2
    rtt = torch.zeros(R + 1, W + 16, dtype=torch.int32, device=core.dev)
    allocs = [pool.alloc(256) for _ in range(R)]
    for b in range(R):
        rtt[b + 1, :256] = torch.tensor(allocs[b], dtype=torch.int32, device=core.dev)
    cur_known = []
    for b in range(R):
        pids = docs[b]["ids"]
        first = prefill_batch(core, kv, [pids], rtt, [b + 1], nv, W, pad=0, attn=attn)[0]
        cur_known.append(list(pids) + [first])
    if pm["kind"] == "ar":
        ex = ARStepExecutor(core, kv, rtt, W, nv, mask_id, R=R, attn=attn, graphs=False)
        for _ in range(12):
            tops = ex.step([len(cur_known[b]) - 1 for b in range(R)], [1, 2], [cur_known[b][-1] for b in range(R)])
            for b in range(R):
                cur_known[b].append(tops[b])
    else:
        sh = LateShape(pm["B"], pm["sets"])
        ex = LateKBatchExecutor(core, pm["K"], sh, kv, rtt, W, nv, mask_id, R=R, attn=attn, graphs=False)
        sts = [LateKReqState(pm["K"], sh.D, len(docs[b]["ids"])) for b in range(R)]
        for _ in range(12):
            hdrs = [tuple(sts[b].hist) for b in range(R)]
            ids_rows = [
                sts[b].build_ids(lambda p, kn=cur_known[b]: kn[p], cur_known[b][-1], sh, mask_id)
                for b in range(R)
            ]
            tops = ex.step(hdrs, [1, 2], ids_rows)
            for b in range(R):
                cm, _ = sts[b].advance(tops[b], sh)
                cur_known[b].extend(cm)
    for al in allocs:
        pool.free(al)
    return [cur_known[b][len(docs[b]["ids"]) :] for b in range(R)]


@torch.no_grad()
def verify_captured_r2(core, pm, ex_r2, rtt_r2, kv, pool, W, nv, mask_id, docs, attn, eager_ref, log):
    """Replay pre-captured R=2 executor for 12 steps and compare against eager_ref."""
    R = 2
    rtt_r2.zero_()
    allocs = [pool.alloc(256) for _ in range(R)]
    for b in range(R):
        rtt_r2[b + 1, :256] = torch.tensor(allocs[b], dtype=torch.int32, device=core.dev)
    cur_known = []
    for b in range(R):
        pids = docs[b]["ids"]
        if pm["kind"] == "lk":
            ex_r2.HB[(b + 1) * ex_r2.RING : (b + 2) * ex_r2.RING].zero_()
        first = prefill_batch(core, kv, [pids], rtt_r2, [b + 1], nv, W, pad=0, attn=attn)[0]
        cur_known.append(list(pids) + [first])
    if pm["kind"] == "ar":
        for _ in range(12):
            tops = ex_r2.step([len(cur_known[b]) - 1 for b in range(R)], [1, 2], [cur_known[b][-1] for b in range(R)])
            for b in range(R):
                cur_known[b].append(tops[b])
    else:
        sh = ex_r2.sh
        sts = [LateKReqState(pm["K"], sh.D, len(docs[b]["ids"])) for b in range(R)]
        for _ in range(12):
            hdrs = [tuple(sts[b].hist) for b in range(R)]
            ids_rows = [
                sts[b].build_ids(lambda p, kn=cur_known[b]: kn[p], cur_known[b][-1], sh, mask_id)
                for b in range(R)
            ]
            tops = ex_r2.step(hdrs, [1, 2], ids_rows)
            for b in range(R):
                cm, _ = sts[b].advance(tops[b], sh)
                cur_known[b].extend(cm)
    for al in allocs:
        pool.free(al)
    rtt_r2.zero_()
    got = [cur_known[b][len(docs[b]["ids"]) :] for b in range(R)]
    exact = got == eager_ref
    diffs = sum(sum(x != y for x, y in zip(got[b], eager_ref[b])) for b in range(R))
    log(f"[check] R=2 12-step graph vs eager exact={exact} token_diffs={diffs}")
    return dict(exact=exact, token_diffs=diffs)


@torch.no_grad()
def bench_one_concurrency(
    core, pm, ex, rtt, rows_per_req, t_capture, kv, pool, W, nv, eos_set, mask_id, docs, C: int,
    max_new: int, trig_ids, trunc_on_stop, attn: str,
    warmup_steps: int, min_steps: int, min_tokens_per_slot: int, max_meas_s: float, log
):
    dev = core.dev
    cuda = str(dev).startswith("cuda")
    sh = ex.sh if pm["kind"] == "lk" else None
    rtt.zero_()

    n_docs = len(docs)
    q_ptr = 0
    slots = [None] * C
    extra_margin = (sh.NB + sh.D + 24) if sh else 24

    def alloc_more(s, b, min_needed):
        while s["cap"] < min_needed:
            chunk = min(128, W + 16 - s["cap"])
            assert chunk > 0, f"Exceeded page-table width W={W} on slot {b}: cur={min_needed}"
            more = pool.alloc(chunk)
            rtt[b + 1, s["cap"] : s["cap"] + chunk] = torch.tensor(more, dtype=torch.int32, device=dev)
            s["alloc"].extend(more)
            s["cap"] += chunk

    def mount_slot(b, doc_idx, stagger_cap=None):
        doc = docs[doc_idx % n_docs]
        pids = list(doc["ids"])
        P0 = len(pids)
        init_cap = min(W, P0 + 192 + extra_margin)
        alloc = pool.alloc(init_cap)
        rtt[b + 1, :init_cap] = torch.tensor(alloc, dtype=torch.int32, device=dev)
        if pm["kind"] == "lk":
            ex.HB[(b + 1) * ex.RING : (b + 2) * ex.RING].zero_()
        first = prefill_batch(core, kv, [pids], rtt, [b + 1], nv, W, pad=0, attn=attn)[0]
        gen = [] if first in eos_set else [first]
        known = pids + gen
        mnew = stagger_cap if stagger_cap is not None else max_new
        done = (first in eos_set) or (len(gen) >= mnew)
        st = LateKReqState(pm["K"], sh.D, P0) if sh else None
        slots[b] = dict(
            doc_id=doc["doc_id"],
            until=doc.get("until", []),
            P0=P0,
            mnew=mnew,
            known=known,
            eff_gen_len=len(gen),
            st=st,
            alloc=alloc,
            cap=init_cap,
            done=done,
            stagger=stagger_cap is not None,
        )

    # Mount initial C slots; for C > 1 stagger initial warmup caps so slots naturally desynchronize
    if cuda:
        torch.cuda.synchronize()
    for b in range(C):
        stag = (12 + (b * 97) % 140) if C > 1 else None
        mount_slot(b, q_ptr, stagger_cap=stag)
        q_ptr += 1
    if cuda:
        torch.cuda.synchronize()

    rpis = [b + 1 for b in range(C)]
    target_tokens = max(2500, min_tokens_per_slot * C)

    # Phase 0: warmup steps (desynchronizes slots & warms caches)
    # Phase 1: steady-state measurement window
    in_meas = False
    step_idx = 0
    meas_steps = 0
    meas_tokens = 0
    meas_raw_commits = 0
    meas_refills = 0
    t_decode = 0.0
    t_refill = 0.0
    t_meas_wall0 = 0.0
    step_lat_ms = []

    while True:
        if not in_meas and step_idx >= warmup_steps:
            # Ensure no staggered-cap artificial request remains before starting measurement
            for b in range(C):
                slots[b]["mnew"] = max_new
                slots[b]["stagger"] = False
            if cuda:
                torch.cuda.synchronize()
            in_meas = True
            t_meas_wall0 = time.perf_counter()

        # Refill any finished slots before next decode step
        refill_idxs = [b for b in range(C) if slots[b]["done"]]
        if refill_idxs:
            if cuda:
                torch.cuda.synchronize()
            tr0 = time.perf_counter()
            # Free finished slots first
            batch_docs = []
            for b in refill_idxs:
                pool.free(slots[b]["alloc"])
                rtt[b + 1, : slots[b]["cap"]].zero_()
                batch_docs.append(docs[q_ptr % n_docs])
                q_ptr += 1
            # Batch-prefill all refilling slots together in ONE T-pass call!
            new_rpis = [b + 1 for b in refill_idxs]
            new_prompts = [list(d["ids"]) for d in batch_docs]
            new_allocs = []
            for b, pids in zip(refill_idxs, new_prompts):
                ic = min(W, len(pids) + 192 + extra_margin)
                al = pool.alloc(ic)
                rtt[b + 1, :ic] = torch.tensor(al, dtype=torch.int32, device=dev)
                if pm["kind"] == "lk":
                    ex.HB[(b + 1) * ex.RING : (b + 2) * ex.RING].zero_()
                new_allocs.append((ic, al))
            firsts = prefill_batch(core, kv, new_prompts, rtt, new_rpis, nv, W, pad=0, attn=attn)
            for b, d, pids, (ic, al), first in zip(refill_idxs, batch_docs, new_prompts, new_allocs, firsts):
                gen = [] if first in eos_set else [first]
                st = LateKReqState(pm["K"], sh.D, len(pids)) if sh else None
                slots[b] = dict(
                    doc_id=d["doc_id"],
                    until=d.get("until", []),
                    P0=len(pids),
                    mnew=max_new,
                    known=pids + gen,
                    eff_gen_len=len(gen),
                    st=st,
                    alloc=al,
                    cap=ic,
                    done=(first in eos_set) or (len(gen) >= max_new),
                    stagger=False,
                )
            if cuda:
                torch.cuda.synchronize()
            if in_meas:
                t_refill += time.perf_counter() - tr0
                meas_refills += len(refill_idxs)

        # Ensure KV slot capacity for the upcoming decode step
        for b in range(C):
            s = slots[b]
            cur_lc = s["st"].lc if s["st"] is not None else (len(s["known"]) - 1)
            if cur_lc + extra_margin > s["cap"]:
                alloc_more(s, b, cur_lc + extra_margin)

        # Timed decode step
        td0 = time.perf_counter()
        if pm["kind"] == "ar":
            lcs = [len(slots[b]["known"]) - 1 for b in range(C)]
            lasts = [slots[b]["known"][-1] for b in range(C)]
            tops = ex.step(lcs, rpis, lasts)
            step_eff_delta = 0
            step_raw_delta = 0
            for b in range(C):
                s = slots[b]
                tk = tops[b]
                step_raw_delta += 1
                if tk in eos_set:
                    s["done"] = True
                else:
                    s["known"].append(tk)
                    gen_len = len(s["known"]) - s["P0"]
                    if tk in trig_ids and s["until"]:
                        gen_slice = s["known"][s["P0"] :]
                        eff = trunc_on_stop(gen_slice, s["until"])
                        if eff < len(gen_slice):
                            s["known"] = s["known"][: s["P0"] + eff]
                            s["done"] = True
                            gen_len = eff
                        else:
                            # Also check if stop string ended right at the end of gen_slice
                            tail_check = trunc_on_stop(gen_slice + [mask_id], s["until"])
                            if tail_check <= len(gen_slice):
                                s["known"] = s["known"][: s["P0"] + tail_check]
                                s["done"] = True
                                gen_len = tail_check
                    if gen_len >= s["mnew"]:
                        s["known"] = s["known"][: s["P0"] + s["mnew"]]
                        gen_len = s["mnew"]
                        s["done"] = True
                    delta = max(0, gen_len - s["eff_gen_len"])
                    s["eff_gen_len"] = gen_len
                    step_eff_delta += delta
        else:
            hdrs = [tuple(slots[b]["st"].hist) for b in range(C)]
            ids_rows = [
                slots[b]["st"].build_ids(lambda p, kn=slots[b]["known"]: kn[p], slots[b]["known"][-1], sh, mask_id)
                for b in range(C)
            ]
            tops = ex.step(hdrs, rpis, ids_rows)
            step_eff_delta = 0
            step_raw_delta = 0
            for b in range(C):
                s = slots[b]
                commit, _ = s["st"].advance(tops[b], sh)
                step_raw_delta += len(commit)
                hit_eos = False
                kept = []
                saw_trig = False
                for tk in commit:
                    if tk in eos_set:
                        hit_eos = True
                        break
                    kept.append(tk)
                    if tk in trig_ids:
                        saw_trig = True
                s["known"].extend(kept)
                gen_len = len(s["known"]) - s["P0"]
                if hit_eos:
                    s["done"] = True
                if saw_trig and s["until"] and gen_len > 0:
                    gen_slice = s["known"][s["P0"] :]
                    eff = trunc_on_stop(gen_slice + [mask_id], s["until"])
                    if eff <= len(gen_slice):
                        s["known"] = s["known"][: s["P0"] + eff]
                        gen_len = eff
                        s["done"] = True
                if gen_len >= s["mnew"]:
                    s["known"] = s["known"][: s["P0"] + s["mnew"]]
                    gen_len = s["mnew"]
                    s["done"] = True
                delta = max(0, gen_len - s["eff_gen_len"])
                s["eff_gen_len"] = gen_len
                step_eff_delta += delta

        dt = time.perf_counter() - td0
        step_idx += 1

        if in_meas:
            t_decode += dt
            meas_steps += 1
            meas_tokens += step_eff_delta
            meas_raw_commits += step_raw_delta
            step_lat_ms.append(dt * 1000.0)
            elapsed_meas = time.perf_counter() - t_meas_wall0
            if (meas_steps >= min_steps and meas_tokens >= target_tokens) or (meas_steps >= 80 and elapsed_meas >= max_meas_s):
                break

    # Clean up active slot allocations
    peak_used_slots = pool.size - 1 - pool.available
    for b in range(C):
        pool.free(slots[b]["alloc"])
    assert pool.available == pool.size - 1, f"Slot leak after C={C}: {pool.available} != {pool.size - 1}"

    step_lat_sorted = sorted(step_lat_ms)
    med_step_ms = step_lat_sorted[len(step_lat_sorted) // 2]
    p90_step_ms = step_lat_sorted[int(0.9 * (len(step_lat_sorted) - 1))]
    decode_tok_s = meas_tokens / max(t_decode, 1e-9)
    per_req_tok_s = decode_tok_s / C
    tpot_ms = 1000.0 / max(per_req_tok_s, 1e-9)
    e2e_tok_s = meas_tokens / max(t_decode + t_refill, 1e-9)
    mean_accept = meas_tokens / max(meas_steps * C, 1)

    res = dict(
        C=C,
        rows_per_req_per_call=rows_per_req,
        total_rows_per_call=[C * r for r in rows_per_req],
        meas_steps=meas_steps,
        meas_tokens=meas_tokens,
        meas_raw_commits=meas_raw_commits,
        meas_refills=meas_refills,
        t_decode_s=round(t_decode, 4),
        t_refill_s=round(t_refill, 4),
        t_capture_s=round(t_capture, 3),
        decode_tok_s=round(decode_tok_s, 2),
        per_req_tok_s=round(per_req_tok_s, 2),
        tpot_ms=round(tpot_ms, 3),
        e2e_tok_s=round(e2e_tok_s, 2),
        mean_eff_tokens_per_req_step=round(mean_accept, 3),
        med_step_ms=round(med_step_ms, 3),
        p90_step_ms=round(p90_step_ms, 3),
        peak_kv_slots=peak_used_slots,
        gpu_mem_gb=round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2) if cuda else 0.0,
    )
    log(
        f"  C={C:2d} | rows/call={res['total_rows_per_call']} | "
        f"total={decode_tok_s:7.1f} tok/s | per-req={per_req_tok_s:6.1f} tok/s (TPOT {tpot_ms:5.2f} ms) | "
        f"e2e={e2e_tok_s:7.1f} tok/s | acc/step={mean_accept:.2f} | step_med={med_step_ms:.2f}ms | mem={res['gpu_mem_gb']:.1f}GB"
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--meta", default="")
    ap.add_argument("--mode", required=True, help="ar or lk:<K>:<shape>")
    ap.add_argument("--concurrencies", default="1,2,4,8,16,32")
    ap.add_argument("--W", type=int, default=1536)
    ap.add_argument("--pool-slots", type=int, default=26624, help="Physical base slots in FlatKV (~39 GiB at 26624)")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--until", type=int, default=1)
    ap.add_argument("--attn", default="paged")
    ap.add_argument("--graphs", type=int, default=1)
    ap.add_argument("--fused", type=int, default=1)
    ap.add_argument("--burn-s", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--min-steps", type=int, default=200)
    ap.add_argument("--min-tokens-per-slot", type=int, default=320)
    ap.add_argument("--max-meas-s", type=float, default=35.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.time()
    def log(*x):
        print(f"[conc {time.time() - t0:7.1f}s] [{a.mode}]", *x, flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    pm = parse_conc_mode(a.mode)
    concs = [int(x) for x in a.concurrencies.split(",") if x.strip()]
    max_C = max(concs)

    meta_path = a.meta or os.path.join(
        os.path.dirname(a.prompts), "meta_" + os.path.basename(a.prompts).replace(".jsonl", ".json")
    )
    pmeta = json.load(open(meta_path))
    nv, eos_list, mask_id = pmeta["nv"], pmeta["eos"], pmeta["mask_id"]
    eos_set = set(eos_list)
    docs = [json.loads(line) for line in open(a.prompts)]
    log(f"Loaded {len(docs)} prompts; mode={a.mode} concurrencies={concs} dev={dev} W={a.W} pool_slots={a.pool_slots}")

    burn_gpu(a.burn_s, dev, log)

    W_dict, cfg = load_weights(a.model, dev, dt)
    core = OuroCore(cfg, dev, dt, W=W_dict, fused=bool(a.fused), log=log)
    del W_dict
    if dev == "cuda":
        torch.cuda.empty_cache()

    sh = LateShape(pm["B"], pm["sets"]) if pm["kind"] == "lk" else None
    scr_len = max(max_C * (sh.PH if sh else 0), 16)
    kv = FlatKV.alloc(cfg, a.pool_slots, scr_len, dev, dt)
    core.ops.ensure_rope(a.W + 64)
    core.ops.frozen = True
    pool = SlotPool(a.pool_slots)
    trig_ids, trunc_on_stop = build_stop_checker(a.model, nv, bool(a.until))
    log(f"FlatKV allocated: pool_slots={a.pool_slots}, scr_len={scr_len}, trig_ids={len(trig_ids)}")

    out = dict(
        meta=dict(
            host=socket.gethostname(),
            gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            torch=torch.__version__,
            cuda=torch.version.cuda,
            mode=a.mode,
            parsed_mode=pm,
            W=a.W,
            pool_slots=a.pool_slots,
            max_new=a.max_new,
            ops=core.ops.chosen,
        ),
        check={},
        results=[],
    )

    def save():
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, a.out)

    # 1. Eager reference run BEFORE any CUDA graph capture/replay
    eager_ref = run_eager_steps(core, pm, kv, pool, a.W, nv, mask_id, docs, a.attn) if dev == "cuda" else None

    # 2. Capture ALL executors for every C up front BEFORE any graph replay (avoids CUDA-13 capture-after-replay bug
    #    and keeps private graph pools alive until process exit).
    capture_set = sorted(set(concs) | ({2} if dev == "cuda" else set()))
    captured = {}
    for C in capture_set:
        rtt_c = torch.zeros(C + 1, a.W + 16, dtype=torch.int32, device=dev)
        # Populate valid dummy base slots during capture warmup
        tmp_slots = pool.alloc(min(pool.available, C * 64))
        for b in range(C):
            seg = tmp_slots[b * 64 : (b + 1) * 64]
            if seg:
                rtt_c[b + 1, : len(seg)] = torch.tensor(seg, dtype=torch.int32, device=dev)
        tc0 = time.perf_counter()
        if pm["kind"] == "ar":
            ex_c = ARStepExecutor(core, kv, rtt_c, a.W, nv, mask_id, R=C, attn=a.attn, graphs=bool(a.graphs))
            ex_c.rpi.copy_(torch.arange(1, C + 1, dtype=torch.int64, device=dev))
            rpr = [1, 1, 1, 1]
        else:
            sh_c = LateShape(pm["B"], pm["sets"])
            ex_c = LateKBatchExecutor(core, pm["K"], sh_c, kv, rtt_c, a.W, nv, mask_id, R=C, attn=a.attn, graphs=bool(a.graphs))
            ex_c.rpi.copy_(torch.arange(1, C + 1, dtype=torch.int64, device=dev))
            rpr = [cl.N for cl in ex_c.calls]
        ex_c.capture()
        tc = time.perf_counter() - tc0
        pool.free(tmp_slots)
        rtt_c.zero_()
        captured[C] = (ex_c, rtt_c, rpr, tc)
        log(f"Captured C={C:2d} in {tc:.2f}s (rows/req/call={rpr})")

    if dev == "cuda" and eager_ref is not None:
        ex2, rtt2, _, _ = captured[2]
        out["check"] = verify_captured_r2(core, pm, ex2, rtt2, kv, pool, a.W, nv, mask_id, docs, a.attn, eager_ref, log)
    else:
        out["check"] = dict(skipped=True)
    save()

    for C in concs:
        ex_c, rtt_c, rpr, tc = captured[C]
        w_steps = a.warmup_steps * 2 if pm["B"] == 0 else a.warmup_steps
        m_steps = a.min_steps * 2 if pm["B"] == 0 else a.min_steps
        res = bench_one_concurrency(
            core, pm, ex_c, rtt_c, rpr, tc, kv, pool, a.W, nv, eos_set, mask_id, docs, C=C,
            max_new=a.max_new, trig_ids=trig_ids, trunc_on_stop=trunc_on_stop,
            attn=a.attn,
            warmup_steps=w_steps, min_steps=m_steps,
            min_tokens_per_slot=a.min_tokens_per_slot, max_meas_s=a.max_meas_s, log=log
        )
        out["results"].append(res)
        save()

    log("DONE mode", a.mode)


if __name__ == "__main__":
    main()
