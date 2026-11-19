"""lattice_trainer.py -- objective and HF Trainer of the K-pass late-lattice fine-tune (DESIGN.md sections 4-5).

loss per micro-batch = w1 * L1 + w_tok * L_tok + w_ph * L_ph   (each = mean over sequences of the per-sequence mean; MARS 'sequence')
  L1    stream-1 CE (full depth, every supervised next token)
  L_tok token-slot rows (chain rows + extra token slots): 'kl' (default) = KL(stream-1 full-depth distribution at the same position,
        DETACHED || slot readout); 'ce' = CE to x_{p+1}. Both are always logged (lat/tok_ce, lat/tok_kl) plus argmax agreement with the
        full-depth argmax (lat/tok_agree = the decoder's audit, per row) and with the true token (lat/tok_acc).
  L_ph  placeholder CE to x_{p+1}
K (block passes) and B (proposals) are drawn per micro-batch from (seed, global_step, micro-step) -> identical on every rank (equal work);
anchors / block choice are drawn per rank from (seed, step, micro, rank).
Loss scaling = the v2 / train path (HF 4.57 + DeepSpeed: logged `loss` = GA x real per-micro-batch loss; real values = lat/*).
"""
import random, time

import torch
import torch.nn.functional as F
import transformers

from lattice_model import build_lattice


def _per_seq_mean(vals, b_idx, B):
    """sum over sequences of the per-sequence mean of vals (b_idx = sequence of each value) and the number of sequences with values"""
    cnt = torch.zeros(B, device=vals.device, dtype=vals.dtype).index_add_(0, b_idx, torch.ones_like(vals))
    sm = torch.zeros(B, device=vals.device, dtype=vals.dtype).index_add_(0, b_idx, vals)
    has = cnt > 0
    return (sm[has] / cnt[has]).sum(), int(has.sum())


def lattice_loss(model, input_ids, labels, K, B, cfg, rng=None, return_parts=False, lat=None):
    """cfg: dict(T, stop_ids, mask_id, alpha, extra_max, extra_frac, tok_loss in {kl, ce}, w1, w_tok, w_ph)"""
    Bs, n = input_ids.shape; dev = input_ids.device
    if lat is None:
        lat = build_lattice(input_ids, labels, K, B, cfg["T"], cfg["stop_ids"], cfg["mask_id"], alpha=cfg.get("alpha", 0.8),
                            extra_max=cfg.get("extra_max", 1024), extra_frac=cfg.get("extra_frac", 1.0), rng=rng)
    lat.to(dev)
    # an empty selection would leave a readout without a graph node on this rank: select row (0,0) with weight 0 instead
    r1, rt, rp = lat.rows1.clone(), lat.tok_sel.clone(), lat.ph_sel.clone()
    e1, et, ep = not bool(r1.any()), not bool(rt.any()), not bool(rp.any())
    if e1: r1[0, 0] = True
    if et: rt[0, 0] = True
    if ep: rp[0, 0] = True
    sel_lat = lat if not (e1 or et or ep) else _with_rows(lat, r1, rt, rp)
    out = model(input_ids=input_ids, lat=sel_lat)
    b1 = torch.arange(Bs, device=dev)[:, None].expand(Bs, n)[r1]
    Q = rt.shape[1]; bq = torch.arange(Bs, device=dev)[:, None].expand(Bs, Q)
    bt, bp = bq[rt], bq[rp]
    lg1 = out.logits1.float(); lgt = out.logits_tok.float(); lgp = out.logits_ph.float()
    ce1 = F.cross_entropy(lg1, lat.tgt1[r1].clamp(min=0), reduction="none")
    # stream-1 teacher at the token rows' positions (all are rows1 positions: p in [a, vend) has a label at p+1)
    idx = torch.full((Bs, n), -1, dtype=torch.long, device=dev); idx[r1] = torch.arange(int(r1.sum()), device=dev)
    ti = idx[bt, lat.tok_pos[rt]]
    assert et or bool((ti >= 0).all()), "token row without a stream-1 teacher row"
    ti = ti.clamp(min=0)
    lp1 = F.log_softmax(lg1.detach()[ti], -1); lpt = F.log_softmax(lgt, -1)
    kl = (lp1.exp() * (lp1 - lpt)).sum(-1)
    cet = F.cross_entropy(lgt, lat.tok_tgt[rt].clamp(min=0), reduction="none")
    cep = F.cross_entropy(lgp, lat.ph_tgt[rp].clamp(min=0), reduction="none")
    tokl = kl if cfg.get("tok_loss", "kl") == "kl" else cet
    z = lambda x: (x * 0.0).sum()
    s1, k1 = (z(ce1), 0) if e1 else _per_seq_mean(ce1, b1, Bs)
    st, kt = (z(tokl), 0) if et else _per_seq_mean(tokl, bt, Bs)
    sp, kp = (z(cep), 0) if ep else _per_seq_mean(cep, bp, Bs)
    loss = (cfg.get("w1", 1.0) * s1 + cfg.get("w_tok", 1.0) * st + cfg.get("w_ph", 1.0) * sp) / Bs
    if not return_parts: return loss
    with torch.no_grad():
        ag = (lgt.argmax(-1) == lg1[ti].argmax(-1)).float(); ac = (lgt.argmax(-1) == lat.tok_tgt[rt].clamp(min=0)).float()
        pa = (lgp.argmax(-1) == lat.ph_tgt[rp].clamp(min=0)).float()
        def ps(v, b, empty):
            if empty: return (0.0, 0)
            sm, kk = _per_seq_mean(v, b, Bs); return (float(sm), kk)
        parts = dict(K=int(K), B=int(B), s1=float(s1), k1=k1, s_tok=float(st), k_tok=kt, s_ph=float(sp), k_ph=kp,
                     tok_ce=ps(cet, bt, et), tok_kl=ps(kl, bt, et), tok_agree=ps(ag, bt, et), tok_acc=ps(ac, bt, et), ph_acc=ps(pa, bp, ep),
                     n1=int(lat.rows1.sum()), n_tok=int(lat.tok_sel.sum()), n_ph=int(lat.ph_sel.sum()), E=int(lat.E), out=out, lat=lat)
    return loss, parts


def _with_rows(lat, r1, rt, rp):
    import copy
    l2 = copy.copy(lat); l2.rows1, l2.tok_sel, l2.ph_sel = r1, rt, rp
    return l2


def parse_spec(choices, weights=""):
    xs = [int(x) for x in str(choices).replace(" ", "").split(",") if x]
    ws = [float(x) for x in str(weights).replace(" ", "").split(",") if x] if weights else [1.0] * len(xs)
    assert xs and len(xs) == len(ws) and all(w >= 0 for w in ws) and sum(ws) > 0, (choices, weights)
    return xs, ws


class StopAtStep(transformers.TrainerCallback):
    def __init__(self, stop):
        self.stop = int(stop)

    def on_step_end(self, args, state, control, **kw):
        if self.stop and state.global_step >= self.stop:
            control.should_training_stop = True; control.should_save = True
        return control


class LatticeTrainer(transformers.Trainer):
    ACC = ("s1", "k1", "s_tok", "k_tok", "s_ph", "k_ph", "n1", "n_tok", "n_ph", "E", "micro")
    PAIR = ("tok_ce", "tok_kl", "tok_agree", "tok_acc", "ph_acc")

    def __init__(self, *args, k_choices=(1, 2, 3), k_weights=None, b_choices=(4, 8), b_weights=None, lat_cfg=None, lat_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_choices, self.k_weights = list(k_choices), list(k_weights or [1.0] * len(k_choices))
        self.b_choices, self.b_weights = list(b_choices), list(b_weights or [1.0] * len(b_choices))
        self.lat_cfg = dict(lat_cfg or {}); self.lat_seed = int(lat_seed)
        self._gs = None; self._micro = 0; self._acc = self._zero(); self._t_log = None; self._gs_log = None

    def _zero(self):
        a = {k: 0.0 for k in self.ACC}
        for p in self.PAIR: a[p] = 0.0; a[p + "_k"] = 0.0
        for K in self.k_choices:
            for p in ("tok_ce", "tok_agree", "tok_kl"): a[f"{p}_K{K}"] = 0.0; a[f"{p}_K{K}_k"] = 0.0
            a[f"m_K{K}"] = 0.0
        for B in self.b_choices: a[f"ph_acc_B{B}"] = 0.0; a[f"ph_acc_B{B}_k"] = 0.0
        return a

    def _get_train_sampler(self, train_dataset=None):
        from mars.trainers.mars_trainer import _arrow_lengths
        ds = train_dataset if train_dataset is not None else self.train_dataset
        lengths = _arrow_lengths(ds, "input_ids") if (self.args.group_by_length and ds is not None) else None
        if lengths is None:
            return super()._get_train_sampler(train_dataset)
        from transformers.trainer_pt_utils import LengthGroupedSampler
        mi = self.processing_class.model_input_names[0] if self.processing_class is not None else None
        print(f"[lat] group_by_length: {len(lengths)} lengths from Arrow offsets (fast path)", flush=True)
        return LengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps, dataset=ds, lengths=lengths, model_input_name=mi)

    def draw(self):
        gs = int(self.state.global_step)
        if gs != self._gs: self._gs, self._micro = gs, 0
        r = random.Random(f"lat-KB/{self.lat_seed}/{gs}/{self._micro}")
        K = r.choices(self.k_choices, weights=self.k_weights, k=1)[0]; B = r.choices(self.b_choices, weights=self.b_weights, k=1)[0]
        rng = random.Random(f"lat-anchor/{self.lat_seed}/{gs}/{self._micro}/{self.args.process_index}")
        self._micro += 1
        return K, B, rng

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ids, labels = inputs["input_ids"], inputs["labels"]
        n = min(ids.shape[1], labels.shape[1]); ids, labels = ids[:, :n], labels[:, :n]
        if self._t_log is None: self._t_log, self._gs_log = time.time(), int(self.state.global_step)
        K, B, rng = self.draw()
        loss, parts = lattice_loss(model, ids, labels, K, B, self.lat_cfg, rng=rng, return_parts=True)
        a = self._acc
        for k in ("s1", "k1", "s_tok", "k_tok", "s_ph", "k_ph", "n1", "n_tok", "n_ph", "E"): a[k] += parts[k]
        a["micro"] += 1; a[f"m_K{K}"] = a.get(f"m_K{K}", 0.0) + 1
        for p in self.PAIR: a[p] += parts[p][0]; a[p + "_k"] += parts[p][1]
        for p in ("tok_ce", "tok_agree", "tok_kl"):
            if f"{p}_K{K}" in a: a[f"{p}_K{K}"] += parts[p][0]; a[f"{p}_K{K}_k"] += parts[p][1]
        if f"ph_acc_B{B}" in a: a[f"ph_acc_B{B}"] += parts["ph_acc"][0]; a[f"ph_acc_B{B}_k"] += parts["ph_acc"][1]
        return (loss, parts["out"]) if return_outputs else loss

    def log(self, logs, start_time=None, **kw):
        if "loss" in logs:
            keys = sorted(self._acc); vec = torch.tensor([self._acc[k] for k in keys], dtype=torch.float64,
                                                         device=self.args.device if torch.cuda.is_available() else "cpu")
            peak = torch.tensor([torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0], dtype=torch.float64, device=vec.device)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(vec); torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
            s = dict(zip(keys, vec.tolist()))
            now = time.time(); gs = int(self.state.global_step); dt = (now - (self._t_log or now)) / max(1, gs - (self._gs_log or 0))
            f = lambda num, den: s[num] / max(1.0, s[den])
            logs["lat/ce1"] = f("s1", "k1"); logs["lat/tok"] = f("s_tok", "k_tok"); logs["lat/ph_ce"] = f("s_ph", "k_ph")
            for p in self.PAIR: logs[f"lat/{p}"] = f(p, p + "_k")
            for K in self.k_choices:
                for p in ("tok_ce", "tok_agree", "tok_kl"):
                    if s.get(f"{p}_K{K}_k", 0) > 0: logs[f"lat/{p}_K{K}"] = s[f"{p}_K{K}"] / s[f"{p}_K{K}_k"]
                logs[f"lat/frac_K{K}"] = f(f"m_K{K}", "micro")
            for B in self.b_choices:
                if s.get(f"ph_acc_B{B}_k", 0) > 0: logs[f"lat/ph_acc_B{B}"] = s[f"ph_acc_B{B}"] / s[f"ph_acc_B{B}_k"]
            logs["lat/rows_tok_per_rows1"] = f("n_tok", "n1"); logs["lat/rows_ph_per_rows1"] = f("n_ph", "n1"); logs["lat/E_mean"] = f("E", "micro")
            logs["lat/sec_per_step"] = dt; logs["lat/peak_gb"] = float(peak.item())
            if self.is_world_process_zero():
                print(f"[lat] step {gs} ce1 {logs['lat/ce1']:.4f} tok({self.lat_cfg.get('tok_loss', 'kl')}) {logs['lat/tok']:.4f} ph_ce {logs['lat/ph_ce']:.4f} | "
                      f"tok_ce {logs['lat/tok_ce']:.4f} tok_kl {logs['lat/tok_kl']:.4f} agree {logs['lat/tok_agree']:.4f} ph_acc {logs['lat/ph_acc']:.4f} | "
                      + " ".join(f"K{K}:ce {logs[f'lat/tok_ce_K{K}']:.3f}/ag {logs[f'lat/tok_agree_K{K}']:.3f}" for K in self.k_choices if f"lat/tok_ce_K{K}" in logs)
                      + f" | E {logs['lat/E_mean']:.0f} | {dt:.2f} s/step | peak {logs['lat/peak_gb']:.1f} GB | lr {logs.get('learning_rate', float('nan')):.3g}"
                      + f" | logged loss {logs['loss']:.4f} (= GA x real)", flush=True)
            self._acc = self._zero(); self._t_log = now; self._gs_log = gs
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        try: return super().log(logs, start_time, **kw)
        except TypeError: return super().log(logs)
