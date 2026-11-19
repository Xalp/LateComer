"""OuroLateForCausalLM -- Ouro-2.6B (v2 checkpoint) as an SGLang model whose forward is the late2 engine's full-depth T-pass forward.

* Declares end_layer = T*NL (192) so ModelRunner sizes the KV pool for every loop (kv_pool.LoopStridedMHAPool gives it the flat
  loop-strided layout); no RadixAttention modules: attention is sgl_kernel FA3 called directly on the flat buffers (core_ops).
* forward() serves EXTEND (prefill, chunked or with a prefix), DECODE (plain AR without the LATE worker), TARGET_VERIFY / IDLE dummy runs:
  rows = input_ids at `positions`, writes at t*STRIDE + out_cache_loc, keys of request b at t*STRIDE + req_to_token[rpi_b, :seq_len],
  4 passes, readout = Ouro's gate mixture over the 4 passes of each request's last row (engine._prefill_body), fp32, logits beyond nv
  set to -inf so a greedy argmax equals late.py's.
* LATE_PREFILL_PAD (default 64): a single-request extend without prefix is padded to a multiple of 64 rows after the prompt (pad rows
  write / read the dummy slot) -- the GEMM shapes of late.py's graph prefill, so the prompt K/V and first token are bit-identical.
* Weights: own load_weights into fused [q;k;v] and [gate;up] parameters; fails on any unmatched or missing checkpoint tensor.
"""
import logging, os
import torch
from torch import nn

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from ..core_ops import Cfg, OuroCore, extend_metas

logger = logging.getLogger(__name__)


class _Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        H, I = c.H, c.INTER
        self.qkv = nn.Parameter(torch.empty(3 * H, H), requires_grad=False)
        self.o = nn.Parameter(torch.empty(H, H), requires_grad=False)
        self.gu = nn.Parameter(torch.empty(2 * I, H), requires_grad=False)
        self.down = nn.Parameter(torch.empty(H, I), requires_grad=False)
        for n in ("n1", "n2", "n3", "n4"): setattr(self, n, nn.Parameter(torch.empty(H), requires_grad=False))


class OuroLateForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = "", **kwargs):
        super().__init__()
        assert quant_config is None, "no quantization"
        self.config = config
        c = Cfg(config.to_dict()); self.c = c
        self.layers = nn.ModuleList([_Layer(c) for _ in range(c.NL)])
        self.embed = nn.Parameter(torch.empty(c.V, c.H), requires_grad=False)
        self.norm = nn.Parameter(torch.empty(c.H), requires_grad=False)
        self.gw = nn.Parameter(torch.empty(1, c.H), requires_grad=False)
        self.gb = nn.Parameter(torch.empty(1), requires_grad=False)
        self.lm_head_w = nn.Parameter(torch.empty(c.V, c.H), requires_grad=False)
        self.start_layer = 0
        self.end_layer = c.T * c.NL                    # one logical KV layer per (loop, layer): the pool holds every loop
        self.nv = int(os.environ.get("LATE_NV", str(c.V)))
        self.prefill_pad = int(os.environ.get("LATE_PREFILL_PAD", "64"))
        self.pt_width = int(os.environ.get("LATE_PT_WIDTH", "0"))
        self.core = None; self.late_kv = None; self.late_rtt = None

    # ---- weights ----
    def _targets(self):
        c = self.c; H, I = c.H, c.INTER; t = {}
        for i, L in enumerate(self.layers):
            p = f"model.layers.{i}."
            t[p + "self_attn.q_proj.weight"] = (L.qkv, slice(0, H)); t[p + "self_attn.k_proj.weight"] = (L.qkv, slice(H, 2 * H))
            t[p + "self_attn.v_proj.weight"] = (L.qkv, slice(2 * H, 3 * H)); t[p + "self_attn.o_proj.weight"] = (L.o, None)
            t[p + "mlp.gate_proj.weight"] = (L.gu, slice(0, I)); t[p + "mlp.up_proj.weight"] = (L.gu, slice(I, 2 * I))
            t[p + "mlp.down_proj.weight"] = (L.down, None)
            t[p + "input_layernorm.weight"] = (L.n1, None); t[p + "input_layernorm_2.weight"] = (L.n2, None)
            t[p + "post_attention_layernorm.weight"] = (L.n3, None); t[p + "post_attention_layernorm_2.weight"] = (L.n4, None)
        t["model.embed_tokens.weight"] = (self.embed, None); t["model.norm.weight"] = (self.norm, None)
        t["model.early_exit_gate.weight"] = (self.gw, None); t["model.early_exit_gate.bias"] = (self.gb, None)
        t["lm_head.weight"] = (self.lm_head_w, None)
        return t

    @torch.no_grad()
    def load_weights(self, weights):
        tg = self._targets(); seen = set(); unmatched = []
        for name, w in weights:
            if name not in tg: unmatched.append(name); continue
            p, sl = tg[name]; dst = p.data if sl is None else p.data[sl]
            assert tuple(dst.shape) == tuple(w.shape), (name, tuple(dst.shape), tuple(w.shape))
            dst.copy_(w.to(dst.dtype)); seen.add(name)
        missing = sorted(set(tg) - seen)
        logger.info(f"[late] load_weights: {len(seen)} tensors loaded, {len(unmatched)} unmatched, {len(missing)} missing")
        if unmatched or missing:
            raise ValueError(f"[late] weight mismatch: unmatched {unmatched[:8]} missing {missing[:8]}")
        self.build_core()

    def build_core(self):
        dev = self.embed.device; dt = self.embed.dtype
        params = dict(layers=[dict(qkv=L.qkv.data, o=L.o.data, gu=L.gu.data, down=L.down.data, n1=L.n1.data, n2=L.n2.data, n3=L.n3.data, n4=L.n4.data)
                              for L in self.layers],
                      embed=self.embed.data, norm=self.norm.data, gw=self.gw.data, gb=self.gb.data, lm_head=self.lm_head_w.data)
        self.core = OuroCore(self.c, dev, dt, params=params, fused=str(dev).startswith("cuda"), log=lambda *x: logger.info(" ".join(map(str, x))))
        logger.info(f"[late] core built on {dev} {dt}: ops {self.core.ops.chosen}, nv {self.nv}, prefill pad {self.prefill_pad}")

    # ---- forward ----
    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch: ForwardBatch, input_embeds=None, **kwargs):
        fb = forward_batch; mode = fb.forward_mode; core = self.core; dev = input_ids.device
        V = self.c.V
        bs = fb.batch_size
        if bs == 0 or input_ids.numel() == 0 or mode.is_idle():
            return LogitsProcessorOutput(next_token_logits=torch.zeros(0, V, device=dev, dtype=torch.float32))
        kv, rtt = self.late_kv, self.late_rtt              # set by LateWorker.alloc_memory_pool (ForwardBatch has no pool refs)
        assert kv is not None and rtt is not None, "OuroLateForCausalLM needs the LATE worker (pool references not bound)"
        ntok = input_ids.shape[0]
        if mode.is_extend() and not mode.is_target_verify():
            ext = list(fb.extend_seq_lens_cpu); seq = [int(x) for x in fb.seq_lens_cpu.tolist()]
            pre = list(fb.extend_prefix_lens_cpu) if fb.extend_prefix_lens_cpu is not None else [s - e for s, e in zip(seq, ext)]
            ids, pos, wbase = input_ids, positions, fb.out_cache_loc.long()
            if bs == 1 and pre[0] == 0 and self.prefill_pad > 0:
                P = ext[0]; Pb = -(-P // self.prefill_pad) * self.prefill_pad
                pad = Pb - P
                ids = torch.cat([ids, torch.zeros(pad, dtype=ids.dtype, device=dev)])
                pos = torch.cat([pos, torch.arange(P, Pb, dtype=pos.dtype, device=dev)])
                wbase = torch.cat([wbase, torch.zeros(pad, dtype=torch.int64, device=dev)])
                base_rows = torch.zeros(1, Pb, dtype=torch.int64, device=dev); base_rows[0, :P] = rtt[fb.req_pool_indices[0], :P].long()
                seql = [Pb]; qls = [Pb]; Wd = Pb
            else:
                Wd = max(seq); seql = seq; qls = ext
                base_rows = rtt[fb.req_pool_indices, :Wd].long()
            cu_l = [0]
            for x in qls: cu_l.append(cu_l[-1] + x)
            sel = torch.tensor([cu_l[i] + ext[i] - 1 for i in range(bs)], device=dev)
            max_q = max(qls)
        else:                                                    # DECODE (AR) / TARGET_VERIFY / DRAFT dummy runs: q rows per request
            q = ntok // bs; assert q * bs == ntok, (ntok, bs)
            ids, pos, wbase = input_ids, positions, fb.out_cache_loc.long()
            Wd = self.pt_width or rtt.shape[1]
            base_rows = rtt[fb.req_pool_indices, :Wd].long()
            seql = fb.seq_lens + (q if mode.is_target_verify() else 0)
            cu_l = None; max_q = q
            sel = torch.arange(ntok, device=dev) if q > 1 else torch.arange(bs, device=dev)
        if cu_l is not None:
            cu = torch.tensor(cu_l, dtype=torch.int32, device=dev); sl = torch.tensor(seql, dtype=torch.int32, device=dev)
        else:
            q = max_q; cu = (torch.arange(bs + 1, device=dev) * q).to(torch.int32); sl = seql.to(torch.int32)
        core.ops.ensure_rope(int(Wd) + 64)
        metas = extend_metas(self.c.T, kv, pos, wbase, base_rows, sl, cu, max_q, Wd)
        exp = core.full_depth(ids, metas, kv, sel, self.nv)
        logits = torch.full((exp.shape[0], V), float("-inf"), device=dev, dtype=torch.float32)
        logits[:, : self.nv] = exp
        return LogitsProcessorOutput(next_token_logits=logits)


EntryClass = [OuroLateForCausalLM]
