"""
MARS "depth forcing" trainer (2026-10-06).

Same objective as MARSTrainer (noisy-stream CE + AR CE on the clean stream), plus: for a random subset of the
sequences in every batch, the noisy rows are BOUND to their own true token at a sampled layer l: a zero-initialised
linear binder maps (E[x_i] - E[mask]) into the residual stream at the input of layer l (noisy half only). Layers >= l then
run with the token identity available, exactly like the partial-forward "tail" of the plan-and-refine decoder, so the
model learns one skill for every binding depth: fully parallel (never bound), late bound (small k = L - l), early bound.
Sequences that are not bound keep the plain MARS objective, so nothing about the parallel draft is unlearned.

The binders live in `model.binders[str(l)]` (nn.Linear(H, H, bias=False)) and are saved with the checkpoint; the clean
stream and everything else are untouched. Loss bookkeeping reports the noisy CE separately for bound / free sequences.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mars_trainer import MARSTrainer


def add_binders(model, layers, zero_init=True):
    H = model.config.hidden_size
    model.binders = nn.ModuleDict({str(l): nn.Linear(H, H, bias=False) for l in layers})
    if zero_init:
        for m in model.binders.values():
            nn.init.zeros_(m.weight)
    return model


class MARSDepthForceTrainer(MARSTrainer):
    def __init__(self, *args, df_layers=(), df_prob=0.5, binder_lr_mult=100.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.df_layers = [int(l) for l in df_layers]
        self.df_prob = float(df_prob)
        self.binder_lr_mult = float(binder_lr_mult)   # the binders start at zero; at the model's 5e-6 they would not move
        self._df = None          # per-step context read by the hooks
        self._hooks = []
        self._df_stats = {}

    # ---- optimizer: the zero-initialised binders get their own (much larger) learning rate --------------------------
    def create_optimizer(self):
        if self.optimizer is None:
            m = self.model
            decay = set(self.get_decay_parameter_names(m))
            is_b = lambda n: n.startswith("binders.") or ".binders." in n
            named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
            groups = [
                {"params": [p for n, p in named if n in decay and not is_b(n)], "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in named if n not in decay and not is_b(n)], "weight_decay": 0.0},
                {"params": [p for n, p in named if is_b(n)], "weight_decay": 0.0, "lr": self.args.learning_rate * self.binder_lr_mult},
            ]
            cls, kw = self.get_optimizer_cls_and_kwargs(self.args, m)
            self.optimizer = cls(groups, **kw)
            nb = sum(p.numel() for p in groups[2]["params"])
            print(f"[depthforce] optimizer groups: decay {len(groups[0]['params'])}, no-decay {len(groups[1]['params'])}, binders {len(groups[2]['params'])} ({nb / 1e6:.0f}M params) at lr x{self.binder_lr_mult}", flush=True)
        return self.optimizer

    # ---- hooks -------------------------------------------------------------------------------------------------
    def _ensure_hooks(self, model):
        if self._hooks:
            return
        base = self.accelerator.unwrap_model(model)
        for l in self.df_layers:
            self._hooks.append(base.model.layers[l].register_forward_pre_hook(self._make_hook(l), with_kwargs=True))

    def _make_hook(self, l):
        def hook(module, args, kwargs):
            # ZeRO-3: EVERY rank must run EVERY binder EVERY step (parameter all-gather / gradient reduce-scatter are
            # collectives), so the binder is always called and sequences that are not bound at this layer are multiplied by 0.
            ctx = self._df
            if ctx is None:
                return None
            h = args[0] if args else kwargs["hidden_states"]
            n, rows = ctx["n"], ctx["rows"][l]                       # rows: bool [b] = sequences bound at this layer
            delta = ctx["binders"][str(l)](ctx["emb_delta"])          # [b, n, H] (zero-init binder -> starts at plain MARS)
            delta = delta * (rows[:, None, None] & ctx["masked"][..., None]).to(delta.dtype)
            h2 = torch.cat([h[:, :n] + delta.to(h.dtype), h[:, n:]], dim=1)
            if args:
                return (h2,) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = h2
            return args, kwargs
        return hook

    # ---- loss ---------------------------------------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self._ensure_hooks(model)
        base = self.accelerator.unwrap_model(model)
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels = inputs["input_ids"], inputs["labels"]
        min_len = min(input_ids.shape[1], labels.shape[1])
        input_ids, labels = input_ids[:, :min_len], labels[:, :min_len]
        b, n = input_ids.shape
        masked = labels != -100
        fixed = torch.where(masked, labels, input_ids)                 # the true token at every supervised position
        # which sequences are bound, and at which layer
        bound = torch.rand(b, device=input_ids.device) < self.df_prob
        choice = torch.randint(len(self.df_layers), (b,), device=input_ids.device)
        rows = {l: bound & (choice == j) for j, l in enumerate(self.df_layers)}       # every layer present, possibly all-False
        emb = base.model.embed_tokens
        mask_id = self.processing_class.mask_token_id
        emb_delta = (emb(fixed) - emb(torch.full_like(fixed, mask_id))).detach()       # computed on every rank, every step
        self._df = {"n": n, "rows": rows, "masked": masked, "emb_delta": emb_delta, "binders": base.binders}
        # NOTE: the context must stay alive through backward: with gradient checkpointing the layers are recomputed
        # in backward and the hooks must add the same delta again. It is replaced at the next compute_loss.
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        # per-regime noisy CE for logging (outputs.logits = noisy logits after the parent's postprocess)
        with torch.no_grad():
            lg = outputs.logits[:, :n]
            ce = F.cross_entropy(lg[masked].float(), labels[masked], reduction="none")
            seq = torch.arange(b, device=lg.device)[:, None].expand(b, n)[masked]
            isb = bound[seq]
            st = self._df_stats
            for name, sel in (("ce_bound", isb), ("ce_free", ~isb)):
                if bool(sel.any()):
                    st[name] = st.get(name, 0.0) + float(ce[sel].sum()); st[name + "_n"] = st.get(name + "_n", 0) + int(sel.sum())
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        st = self._df_stats
        for name in ("ce_bound", "ce_free"):
            if st.get(name + "_n", 0):
                logs[name] = round(st[name] / st[name + "_n"], 4)
        self._df_stats = {}
        return super().log(logs, *args, **kwargs)
