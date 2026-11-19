"""train_lad.py -- LoopAhead repro: train the LAD drafter on the frozen released Ouro-2.6B teacher data (their A.5 recipe).

One GPU job, two phases:
  A) pass-1 state extraction: for every teacher sequence (prompt + greedy continuation), a teacher-forced single-pass
     forward through the released model gives h^(1) for every position. Runs through the SAME engine code the decoders
     use (decode/engine.py loop_pass t=0, torch attention) -- exact numerics by construction. Cached to a /tmp
     memmap with a .ready marker (idempotent).
  B) drafter training, their recipe: 3 epochs x 32 anchors per request, microbatch 8 x accum 8 (eff 64), AdamW
     (0.9,0.95) wd 0.01 clip 1.0, LR linear-warmup 5% -> 2e-4 then cosine -> 2e-5, seed 42, dev eval every 500 updates.
     Anchor = the teacher's actual token (ground-truth anchor; recorded assumption), window = the 64 pass-1 states
     ending at the anchor's predecessor, labels = the next K_train=7 teacher tokens (masked past the continuation).

  python train_lad.py --model /tmp/ouro26 --data-dir <repro>/data --out-dir <repro>/drafter/ckpt

Outputs: ckpt/lad_ouro26.pt (final), ckpt/lad_ouro26_best.pt (best dev CE), ckpt/trainlog.jsonl,
         ckpt/dev_acceptance.json (teacher-forced q_cum per draft position, full + restricted vocab), meta done=true.
"""
import argparse, json, math, os, random, sys, time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "decode"))
from lad import LAD
import engine as E

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data-dir", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--h1-dir", default="/tmp/la_h1")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--anchors-per-req", type=int, default=32)
ap.add_argument("--micro", type=int, default=8)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--lr-min", type=float, default=2e-5)
ap.add_argument("--warmup-frac", type=float, default=0.05)
ap.add_argument("--k-train", type=int, default=7)
ap.add_argument("--window", type=int, default=64)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--dev-every", type=int, default=500)
ap.add_argument("--dev-anchors", type=int, default=2048)
ap.add_argument("--max-updates", type=int, default=0, help="debug cap")
a = ap.parse_args()
T0 = time.time()
def log(*x): print(f"[lad {time.time()-T0:7.1f}s]", *x, flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
os.makedirs(a.out_dir, exist_ok=True); os.makedirs(a.h1_dir, exist_ok=True)

# ---------------------------------------------------------------- load teacher data
def load_split(name):
    mix = {json.loads(l)["sid"]: json.loads(l) for l in open(f"{a.data_dir}/mix_{name}.jsonl")}
    seqs = []
    for l in open(f"{a.data_dir}/teacher_{name}.jsonl"):
        t = json.loads(l)
        m = mix[t["sid"]]
        if len(t["gen"]) < 2: continue                       # need at least one (anchor, label) pair
        seqs.append(dict(sid=t["sid"], source=t["source"], ids=m["ids"] + t["gen"], P=len(m["ids"])))
    return seqs

train = load_split("train"); devset = load_split("dev")
log(f"train {len(train)} seqs, dev {len(devset)} seqs")

# ---------------------------------------------------------------- phase A: h^(1) extraction via the engine
def extract(split, seqs):
    idxf, binf = f"{a.h1_dir}/{split}.idx.json", f"{a.h1_dir}/{split}.h1.fp16"
    if os.path.exists(f"{a.h1_dir}/{split}.ready"):
        log(f"h1 {split}: cached"); return json.load(open(idxf)), binf
    W, cfg = E.load_weights(a.model, dev, torch.bfloat16)
    S = max(len(s["ids"]) for s in seqs) + 8
    eng = E.OuroEngine(W, cfg, dev, torch.bfloat16, S=S, fused=False, log=log); del W
    idx = {}; off = 0
    f = open(binf + ".tmp", "wb")
    with torch.no_grad():
        for n, s in enumerate(seqs):
            L = len(s["ids"])
            m = E.Meta()
            m.pos = torch.arange(L, device=dev); m.slot = m.pos
            m.page_table = torch.arange(L, device=dev, dtype=torch.int32)[None]
            m.seqlens = torch.tensor([L], device=dev, dtype=torch.int32)
            m.cu_q = torch.tensor([0, L], device=dev, dtype=torch.int32); m.max_q = L
            X = F.embedding(torch.tensor(s["ids"], device=dev), eng.embed)
            h1 = eng.loop_pass(X, 0, m, "torch")                     # [L, 2048] pass-1 normed states
            f.write(h1.to(torch.float16).cpu().numpy().tobytes())
            idx[s["sid"]] = [off, s["P"], L]; off += L
            if n % 500 == 0: log(f"h1 {split}: {n}/{len(seqs)}")
    f.close(); os.replace(binf + ".tmp", binf)
    json.dump(idx, open(idxf, "w")); open(f"{a.h1_dir}/{split}.ready", "w").write("1")
    log(f"h1 {split}: {off} positions, {off*2048*2/1e9:.1f} GB")
    return idx, binf

idx_tr, bin_tr = extract("train", train)
idx_dv, bin_dv = extract("dev", devset)
H1_TR = np.memmap(bin_tr, dtype=np.float16, mode="r").reshape(-1, 2048)
H1_DV = np.memmap(bin_dv, dtype=np.float16, mode="r").reshape(-1, 2048)

# frozen embed + head only (no full model in phase B)
W, cfg = E.load_weights(a.model, dev, torch.bfloat16)
EMB = W["model.embed_tokens.weight"].to(dev); HEAD = W["lm_head.weight"].to(dev); del W
NV = EMB.shape[0]
log(f"embed/head loaded, vocab {NV}")

WSZ, K = a.window, a.k_train

def gather(seqs, idx, H1, picks):
    """picks: list of (seq_i, anchor_pos a). returns H [B,64,2048], hmask, E_a, labels [B,K] (-100 masked)"""
    B = len(picks)
    H = torch.zeros(B, WSZ, 2048, dtype=torch.float16)
    hmask = torch.zeros(B, WSZ, dtype=torch.bool)
    anc = torch.zeros(B, dtype=torch.long); lab = torch.full((B, K), -100, dtype=torch.long)
    for i, (si, apos) in enumerate(picks):
        s = seqs[si]; off, P, L = idx[s["sid"]]
        w0 = max(0, apos - WSZ)
        chunk = H1[off + w0: off + apos]                                  # states of positions w0..apos-1
        H[i, WSZ - len(chunk):] = torch.from_numpy(np.ascontiguousarray(chunk))
        hmask[i, WSZ - len(chunk):] = True
        anc[i] = s["ids"][apos]
        for j in range(K):
            if apos + 1 + j < L: lab[i, j] = s["ids"][apos + 1 + j]
    return H.to(dev, torch.float32), hmask.to(dev), F.embedding(anc.to(dev), EMB).float(), lab.to(dev)

def logits_of(model, H, hmask, E_a, Kq):
    delta = model(H, hmask, E_a, K=Kq)                                     # [B,K,2048] fp32 (LAD runs fp32)
    h_tilde = H[:, -1][:, None] + delta
    return F.linear(h_tilde.to(torch.bfloat16), HEAD).float()              # frozen bf16 head, fp32 logits

# ---------------------------------------------------------------- dev eval
dv_rng = random.Random(a.seed + 1)
dev_picks = []
for si, s in enumerate(devset):
    L, P = len(s["ids"]), s["P"]
    for apos in dv_rng.sample(range(P, L - 1), min(4, L - 1 - P)):
        dev_picks.append((si, apos))
dv_rng.shuffle(dev_picks); dev_picks = dev_picks[:a.dev_anchors]
log(f"dev anchors: {len(dev_picks)}")
dvocab = None
dv_path = f"{a.data_dir}/draft_vocab.json"
if os.path.exists(dv_path):
    dvocab = set(json.load(open(dv_path))["vocab"])
    log(f"draft vocab loaded: {len(dvocab)}")

@torch.no_grad()
def dev_eval(model):
    model.eval()
    tot = n = 0.0; cum_full = np.zeros(K); cum_rest = np.zeros(K); cnt = np.zeros(K)
    by_src = {}   # source -> [cum_full, cum_rest, cnt] (their B.2 anchor is per-task: Ouro/GSM8K q1~.643 q3~.361)
    for i in range(0, len(dev_picks), 64):
        picks = dev_picks[i:i+64]
        H, hm, Ea, lab = gather(devset, idx_dv, H1_DV, picks)
        lg = logits_of(model, H, hm, Ea, K)
        v = (lab != -100)
        tot += F.cross_entropy(lg.view(-1, NV), lab.view(-1), ignore_index=-100, reduction="sum").item()
        n += v.sum().item()
        top = lg.argmax(-1)
        okf = (top == lab) | ~v
        for b, (si, apos) in enumerate(picks):
            s = devset[si]
            sb = by_src.setdefault(s["source"], [np.zeros(K), np.zeros(K), np.zeros(K)])
            allow = dvocab | set(s["ids"][:apos + 1]) | {0, 2} if dvocab else None
            if allow is not None:
                al = torch.tensor(sorted(allow), device=dev)
                topr = al[lg[b][:, al].argmax(-1)]
            good_f = good_r = True
            for j in range(K):
                if not v[b, j]: break
                good_f &= bool(top[b, j] == lab[b, j])
                good_r &= bool((topr[j] if allow is not None else top[b, j]) == lab[b, j])
                cum_full[j] += good_f; cum_rest[j] += good_r; cnt[j] += 1
                sb[0][j] += good_f; sb[1][j] += good_r; sb[2][j] += 1
    model.train()
    return dict(ce=tot / max(1, n), q_full=[round(x, 4) for x in (cum_full / np.maximum(cnt, 1)).tolist()],
                q_restricted=[round(x, 4) for x in (cum_rest / np.maximum(cnt, 1)).tolist()],
                n_per_pos=cnt.astype(int).tolist(),
                by_source={src: dict(q_full=[round(x, 4) for x in (cf / np.maximum(c, 1)).tolist()],
                                     q_restricted=[round(x, 4) for x in (cr / np.maximum(c, 1)).tolist()],
                                     n=int(c[0]))
                           for src, (cf, cr, c) in sorted(by_src.items())})

# ---------------------------------------------------------------- phase B: training
model = LAD(h_tgt=2048, d=512, heads=8, ffn=1536, layers=2, w=WSZ, k_train=K).to(dev)
log(f"LAD params: {model.n_params()}")
assert model.n_params() == 11051008, model.n_params()
decay, nodecay = [], []
for nm, p in model.named_parameters():
    (nodecay if ("norm" in nm or "pos" in nm) else decay).append(p)
opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.01), dict(params=nodecay, weight_decay=0.0)],
                        lr=a.lr, betas=(0.9, 0.95))
per_epoch = (len(train) * a.anchors_per_req) // (a.micro * a.accum)
total_updates = a.epochs * per_epoch
if a.max_updates: total_updates = min(total_updates, a.max_updates)
warm = max(1, int(a.warmup_frac * total_updates))
def lr_at(u):
    if u < warm: return a.lr * (u + 1) / warm
    t = (u - warm) / max(1, total_updates - warm)
    return a.lr_min + 0.5 * (a.lr - a.lr_min) * (1 + math.cos(math.pi * t))

rng = random.Random(a.seed)
tlog = open(f"{a.out_dir}/trainlog.jsonl", "a")
upd = 0; best = 1e9; t_tr = time.time(); losses = []
log(f"training: {total_updates} updates ({per_epoch}/epoch), warmup {warm}")
for ep in range(a.epochs):
    picks = []
    for si, s in enumerate(train):
        L, P = len(s["ids"]), s["P"]
        cand = range(P, L - 1)
        picks.extend((si, apos) for apos in (rng.sample(cand, a.anchors_per_req) if len(cand) >= a.anchors_per_req
                                             else [rng.choice(cand) for _ in range(a.anchors_per_req)]))
    rng.shuffle(picks)
    i = 0
    while i + a.micro * a.accum <= len(picks):
        opt.zero_grad(set_to_none=True)
        for g in opt.param_groups: g["lr"] = lr_at(upd)
        acc = 0.0
        for _ in range(a.accum):
            mb = picks[i:i + a.micro]; i += a.micro
            H, hm, Ea, lab = gather(train, idx_tr, H1_TR, mb)
            lg = logits_of(model, H, hm, Ea, K)
            loss = F.cross_entropy(lg.view(-1, NV), lab.view(-1), ignore_index=-100)
            (loss / a.accum).backward()
            acc += loss.item() / a.accum
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); upd += 1; losses.append(acc)
        if upd % 100 == 0:
            log(f"ep{ep} upd {upd}/{total_updates} loss {sum(losses[-100:])/len(losses[-100:]):.4f} "
                f"lr {lr_at(upd-1):.2e} gn {gn:.2f} ({(time.time()-t_tr)/upd*1000:.0f} ms/upd)")
        if upd % a.dev_every == 0 or upd == total_updates:
            dv = dev_eval(model)
            log(f"DEV upd {upd}: ce {dv['ce']:.4f} q_full {dv['q_full'][:3]} q_restr {dv['q_restricted'][:3]}")
            tlog.write(json.dumps(dict(upd=upd, loss=sum(losses[-100:])/len(losses[-100:]), **dv)) + "\n"); tlog.flush()
            if dv["ce"] < best:
                best = dv["ce"]
                torch.save(dict(sd=model.state_dict(), upd=upd, dev=dv, args=vars(a)), f"{a.out_dir}/lad_ouro26_best.pt")
        if a.max_updates and upd >= a.max_updates: break
    if a.max_updates and upd >= a.max_updates: break

dv = dev_eval(model)
torch.save(dict(sd=model.state_dict(), upd=upd, dev=dv, args=vars(a)), f"{a.out_dir}/lad_ouro26.pt")
json.dump(dict(done=True, updates=upd, final_dev=dv, best_dev_ce=best,
               train_seqs=len(train), dev_seqs=len(devset), torch=torch.__version__,
               gpu=torch.cuda.get_device_name() if dev == "cuda" else "cpu"),
          open(f"{a.out_dir}/meta.json", "w"), indent=1)
log(f"DONE: final dev {dv}")
