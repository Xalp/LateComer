"""spec_worker.py -- the LATE speculative algorithm: a spec-V2 worker plugin (non-overlap) on SGLang's standard scheduler.

Modelled on sglang/srt/speculative/ngram_worker.py (NGRAMWorker: no draft model, V2 result contract). Per forward:
  EXTEND  -> the target worker's normal forward of OuroLateForCausalLM (full-depth prefill writing all 4 loop regions), sampling skipped
             (is_verify=True); the first token = top-1 of the 4-pass gate mixture (late.py's prefill readout, topk(2)[0]).
  DECODE  -> one lattice step of late_step.LateStepExecutor (own CUDA graphs, own FA3 calls on the loop-strided pool) for the batch, host
             verification per request (LateReqState.advance); or, with --late-mode ar, one plain full-depth AR token (ARStepExecutor).
Result contract (as NGRAM, consumed by Scheduler.run_batch's non-overlap V2 branch and batch_result_processor._resolve_spec_v2_tokens):
  next_token_ids [bs*D] (request b's commit at [b*D, b*D + c_b + 1)), accept_lens [bs] = c_b + 1, new_seq_lens = new lc (device),
  next_draft_input = NgramVerifyInput(draft_token_num=D, accept_tokens, accept_lens, new_seq_lens) (filter/merge + prepare_for_decode:
  reserve 2D slots ahead, pre-claim the bonus slot -> kv_committed_len = lc + 1 during the step, resolve adds c -> = new lc).
The per-request lattice state lives on the Req (req._late) and is created at its first decode step (lc = p1 = p2 = seq_len).
"""
import json, logging, os, time
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.ngram_info import NgramVerifyInput

from .late_step import LateStepExecutor, ARStepExecutor, LateReqState
from .shapes import LateShape, parse_mode

logger = logging.getLogger(__name__)
from types import SimpleNamespace
_NO_DRAFT = SimpleNamespace(draft_runner=SimpleNamespace(token_to_kv_pool=None, model_config=None, device_timer=None))


def late_settings():
    mode = os.environ.get("LATE_MODE", "l2:lkv:am:latt:4")
    p = parse_mode(mode)
    D = 1 if p["kind"] == "ar" else p["B"] + 1
    sh = LateShape(p["B"], p["sets"]) if p["kind"] == "l2" else None
    return dict(mode=mode, p=p, D=D, shape=sh, PH=(sh.PH if sh else 0),
                nv=int(os.environ.get("LATE_NV", "49153")), mask_id=int(os.environ.get("LATE_MASK_ID", "49152")),
                W=int(os.environ.get("LATE_PT_WIDTH", "0")), Rmax=int(os.environ.get("LATE_RMAX", "1")),
                graphs=os.environ.get("LATE_GRAPHS", "1") != "0", stats=os.environ.get("LATE_STATS", ""),
                margins=os.environ.get("LATE_MARGINS", "0") == "1", K=max(1, int(os.environ.get("LATE_STEPS_PER_CALL", "1"))))


def _stop_ids(req):
    """the token ids on which SGLang's Req._check_token_based_finish (0.5.14) finishes this request: none with ignore_eos, else
    sampling_params.stop_token_ids | req.eos_token_ids | tokenizer.eos_token_id | tokenizer.additional_stop_token_ids (cached on the req)"""
    s = getattr(req, "_late_stop", None)
    if s is not None: return s
    sp = req.sampling_params; s = set()
    if not getattr(sp, "ignore_eos", False):
        s |= set(getattr(sp, "stop_token_ids", None) or ())
        s |= set(getattr(req, "eos_token_ids", None) or ())
        tk = getattr(req, "tokenizer", None)
        if tk is not None:
            if getattr(tk, "eos_token_id", None) is not None: s.add(tk.eos_token_id)
            s |= set(getattr(tk, "additional_stop_token_ids", None) or ())
    req._late_stop = frozenset(s)
    return req._late_stop


class LateWorker(BaseSpecWorker):
    def __init__(self, server_args, gpu_id, tp_rank, dp_rank, moe_ep_rank, attn_cp_rank, moe_dp_rank, nccl_port, target_worker):
        self.server_args = server_args
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.device = self.model_runner.device
        self.cfg = late_settings()
        self.D = self.cfg["D"]; self.K = self.cfg["K"]; self.stride = self.K * self.D          # lattice tokens per step / steps per call
        self.speculative_num_draft_tokens = self.stride; self.draft_token_num = self.stride
        self.eos = set(int(x) for x in os.environ.get("LATE_EOS", "0,2").split(","))
        assert server_args.speculative_num_draft_tokens == self.stride, (server_args.speculative_num_draft_tokens, self.stride)
        assert server_args.disable_overlap_schedule and server_args.page_size == 1
        self.ex = None
        self._stats_f = open(self.cfg["stats"], "a", buffering=1) if self.cfg["stats"] else None
        self._t_last_exit = None
        self.n_steps = 0
        logger.info(f"[late] LateWorker mode {self.cfg['mode']} D={self.D} W={self.cfg['W']} Rmax={self.cfg['Rmax']} graphs={self.cfg['graphs']}")

    # ---- BaseSpecWorker interface ----
    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        # No draft model. Scheduler.init_disaggregation -> kv_cache_builder.get_draft_kv_pool dereferences
        # draft_worker.draft_worker.draft_runner for every non-NGRAM algorithm; this stub makes it return (None, None)
        # (= no draft KV pool, target model_config), which is what NGRAM gets from its own early return.
        return _NO_DRAFT

    def clear_cache_pool(self):
        pass

    def update_weights_from_tensor(self, recv_req):
        return self.target_worker.update_weights_from_tensor(recv_req)

    def alloc_memory_pool(self, **kwargs):
        self.req_to_token_pool, self.token_to_kv_pool_allocator = self._target_worker.get_memory_pool()
        # ForwardBatch carries no pool references in 0.5.14: hand the model its KV store and req_to_token table directly
        mr = self.model_runner
        mr.model.late_kv = mr.token_to_kv_pool.flat; mr.model.late_rtt = self.req_to_token_pool.req_to_token
        assert mr.model.late_kv is not None, "LoopStridedMHAPool not active"

    def init_cuda_graphs(self):
        """build the executor and capture its graphs now (server init, before any replay of anything)"""
        mr = self.model_runner; model = mr.model; cfg = self.cfg
        kv = mr.token_to_kv_pool.flat; assert kv is not None, "LoopStridedMHAPool not active"
        rtt = self.req_to_token_pool.req_to_token
        W = cfg["W"] or rtt.shape[1]
        assert W <= rtt.shape[1], (W, rtt.shape)
        t0 = time.time()
        # (review 10-28) the graphs capture the RoPE table by address: size it for every position this server can see, then freeze it
        model.core.ops.ensure_rope(max(W, rtt.shape[1]) + 64)
        if cfg["p"]["kind"] == "ar":
            self.ex = ARStepExecutor(model.core, kv, rtt, W, cfg["nv"], cfg["mask_id"], R=cfg["Rmax"], graphs=cfg["graphs"], log=logger.info)
        else:
            self.ex = LateStepExecutor(model.core, cfg["shape"], kv, rtt, W, cfg["nv"], cfg["mask_id"], R=cfg["Rmax"], read=cfg["p"]["read"],
                                       graphs=cfg["graphs"], log=logger.info, margins=cfg["margins"])
        self.ex.capture()
        if self.ex.graphs is not None: model.core.ops.frozen = True
        torch.cuda.synchronize()
        logger.info(f"[late] executor ready ({type(self.ex).__name__}, W={W}, rtt {tuple(rtt.shape)}, graphs {self.ex.graphs is not None}) "
                    f"in {time.time() - t0:.1f}s; mem allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    # ---- forward ----
    def forward_batch_generation(self, batch, on_publish=None):
        if batch.forward_mode.is_decode():
            return self._decode(batch, on_publish)
        return self._extend(batch, on_publish)

    def _result(self, logits_output, next_ids_dev, accept_lens_dev, new_seq_lens, can_graph):
        nd = NgramVerifyInput(draft_token_num=self.stride, new_seq_lens=new_seq_lens, accept_tokens=next_ids_dev, accept_lens=accept_lens_dev)
        return GenerationBatchResult(logits_output=logits_output, next_token_ids=next_ids_dev, can_run_cuda_graph=can_graph,
                                     accept_lens=accept_lens_dev, new_seq_lens=new_seq_lens, next_draft_input=nd,
                                     speculative_num_draft_tokens=self.stride)

    def _extend(self, batch, on_publish):
        t0 = time.perf_counter()
        res = self.target_worker.forward_batch_generation(batch, is_verify=True)        # full-depth prefill; sampling skipped
        lo = res.logits_output
        bs = len(batch.reqs)
        if lo is None or lo.next_token_logits is None or lo.next_token_logits.shape[0] != bs:
            # (review 10-28) was a silent fallback to token 0 = an EOS: fail loudly instead
            raise RuntimeError(f"[late] prefill readout missing: bs {bs}, logits "
                               f"{None if lo is None or lo.next_token_logits is None else tuple(lo.next_token_logits.shape)}")
        predict = lo.next_token_logits[:, : self.cfg["nv"]].topk(2, -1).indices[:, 0]
        acc = torch.zeros(bs, self.stride, dtype=torch.int64, device=self.device); acc[:, 0] = predict
        new_seq_lens = batch.seq_lens.clone()
        for req in batch.reqs:
            req._late = None                                                                 # (re)prefill -> fresh lattice state
            self._clamp_len(req)
            sp = req.sampling_params
            if (getattr(sp, "temperature", 0) or 0) > 1e-6 and getattr(sp, "top_k", 1) != 1 and not getattr(req, "_late_warned", False):
                req._late_warned = True
                logger.warning(f"[late] request {req.rid}: non-greedy sampling requested; the late lattice decodes greedily (argmax verification)")
        if on_publish is not None: on_publish(new_seq_lens)
        out = self._result(lo, predict, torch.ones(bs, dtype=torch.int32, device=self.device), new_seq_lens, res.can_run_cuda_graph)
        out.next_draft_input.accept_tokens = acc.flatten()
        if self._stats_f is not None:
            torch.cuda.synchronize(); t1 = time.perf_counter()
            for req in batch.reqs:
                self._stats_f.write(json.dumps(dict(k="x", rid=req.rid, t0=t0, t1=t1, ext=len(getattr(req, "fill_ids", None) or []))) + "\n")
            self._t_last_exit = t1
        return out

    def _len_cap(self, P0):
        """(review 10-28) largest max_new_tokens the lattice can serve for a P0-token prompt. SGLang only guarantees
        P0 + max_new <= context_len - 2, which is not enough here:
          (a) the executor needs lc + D + TM < W (page-table columns of the block tails; ARStepExecutor: lc + 1 < W) -- the assert
              fires inside the scheduler and kills the server;
          (b) prepare_for_decode (eagle_info_v2) reserves req_to_token[rpi, :lc + 2*stride] in a row that is only
              context_len + 4 + stride wide -> for stride > ~8 (B=8, or K > 1) it writes past the row with an unchecked kernel.
        Every step starts with len(output_ids) <= max_new - 1, i.e. lc = P0 + len(out) - 1 <= P0 + max_new - 2 (a K>1 call only
        starts a further step while its total stays below max_new), so (a) <=> max_new <= W - D - TM - P0 + 1 and
        (b) <=> max_new <= row - 2*stride - P0 + 2."""
        sh = self.cfg["shape"]; D, TM = (sh.D, sh.TM) if sh is not None else (1, 0)
        W = self.ex.W; row = self.req_to_token_pool.req_to_token.shape[1]
        return min(W - D - TM - P0 + 1, row - 2 * self.stride - P0 + 2)

    def _clamp_len(self, req):
        sp = req.sampling_params; mx = getattr(sp, "max_new_tokens", None)
        cap = max(1, self._len_cap(len(req.origin_input_ids)))
        if mx is None or mx > cap:
            sp.max_new_tokens = cap
            logger.warning(f"[late] request {req.rid}: max_new_tokens {mx} -> {cap} (prompt {len(req.origin_input_ids)}; lattice "
                           f"page-table width {self.ex.W} / req_to_token row {self.req_to_token_pool.req_to_token.shape[1]})")

    def _decode(self, batch, on_publish):
        """K = LATE_STEPS_PER_CALL lattice steps per scheduler iteration (default 1). With K > 1 the request's later steps run inside
        this call (their ids read the commits of the earlier ones); the call stops early after a commit containing EOS or reaching
        max_new_tokens. The stride of next_token_ids is K*D (= --speculative-num-draft-tokens)."""
        reqs = batch.reqs; bs = len(reqs); D = self.D; K = self.K; stride = self.stride
        seq = batch.seq_lens_cpu.tolist()
        rpis = [r.req_pool_idx for r in reqs]
        mask = self.cfg["mask_id"]; sh = self.cfg["shape"]
        stops = [_stop_ids(r) for r in reqs]
        states, news, budget = [], [], []
        for b, req in enumerate(reqs):
            st = getattr(req, "_late", None)
            if st is None:
                st = LateReqState(seq[b]); req._late = st
            P0 = len(req.origin_input_ids); out = req.output_ids
            if st.lc != seq[b] or P0 + len(out) != st.lc + 1:
                raise RuntimeError(f"[late] state desync rid {req.rid}: lc {st.lc} seq_len {seq[b]} prompt {P0} outputs {len(out)} "
                                   f"kv_committed {req.kv_committed_len} kv_allocated {req.kv_allocated_len}")
            states.append(st); news.append([])
            mx = getattr(req.sampling_params, "max_new_tokens", None)
            budget.append((mx - len(out)) if mx is not None else 1 << 30)
        live = list(range(bs)); recs = []
        gap = None
        for k in range(K):
            t_in = time.perf_counter()
            if k == 0 and self._t_last_exit is not None: gap = t_in - self._t_last_exit
            if sh is None:                                                                  # plain AR
                last = [(news[b][-1] if news[b] else reqs[b].output_ids[-1]) for b in range(bs)]
                top = self.ex.step([st.lc for st in states], rpis, last)
                t_g = time.perf_counter()
                for b in live:
                    st = states[b]; st.p2 = st.p1 = st.lc = st.lc + 1; st.steps += 1; news[b].append(top[b])
                    recs.append((b, k, 1, 0, t_in, t_g, None))
            else:
                hdrs, idrows = [], []
                for b in range(bs):
                    st = states[b]; P0 = len(reqs[b].origin_input_ids); out = reqs[b].output_ids; nw = news[b]; no = len(out)
                    tok = (lambda p, out=out, nw=nw, P0=P0, no=no: out[p - P0] if p - P0 < no else nw[p - P0 - no])
                    idrows.append(st.build_ids(tok, nw[-1] if nw else out[-1], sh, mask)); hdrs.append((st.lc, st.p1, st.p2))
                tops = self.ex.step(hdrs, rpis, idrows)
                t_g = time.perf_counter()
                lg_ = self.ex.last_gap
                for b in live:
                    cm, c = states[b].advance(tops[b], sh); news[b].extend(cm)
                    recs.append((b, k, len(cm), c, t_in, t_g, None if lg_ is None else [round(x, 4) for x in lg_[b * sh.NB : b * sh.NB + len(cm)]]))
            # stop running further steps for a request whose run hit EOS or its token budget (bs > 1 with K > 1 is M2: all stop together)
            live = [b for b in live if not any(t in stops[b] for t in news[b]) and len(news[b]) < budget[b] and len(news[b]) + D <= stride]
            if not live: break
            if bs > 1 and len(live) < bs: break
        t_g = time.perf_counter()
        flat = [0] * (bs * stride); al = []; nl = []
        for b in range(bs):
            cm = news[b]; lc = states[b].lc
            e = next((i for i, t in enumerate(cm) if t in stops[b]), None)
            if e is not None and e + 1 < len(cm):
                # late.py ends the request at the first EOS of a commit. SGLang's update_finish_state checks max_new_tokens before
                # stop tokens on a multi-token run, so a run crossing the budget would otherwise finish by "length" with the EOS and
                # the tokens after it in the output: hand SGLang the run cut after its first stop token. Only tokens SGLang itself
                # stops on are cut (stops[b] mirrors _check_token_based_finish), so the request always finishes at the cut and the
                # lattice state (already advanced past it) is never used again.
                lc -= len(cm) - (e + 1); cm = cm[: e + 1]
            flat[b * stride : b * stride + len(cm)] = cm; al.append(len(cm)); nl.append(lc)
        n = bs * stride + 2 * bs
        if getattr(self, "_h_pack", None) is None or self._h_pack.shape[0] < n:
            self._h_pack = torch.zeros(max(n, 4096), dtype=torch.int64).pin_memory()
        self._h_pack[:n].copy_(torch.as_tensor(flat + al + nl, dtype=torch.int64))
        packed = self._h_pack[:n].to(self.device, non_blocking=True)
        next_ids = packed[: bs * stride]; accept_lens = packed[bs * stride : bs * stride + bs].to(torch.int32); new_seq_lens = packed[bs * stride + bs :]
        if on_publish is not None: on_publish(new_seq_lens)
        out = self._result(LogitsProcessorOutput(next_token_logits=None), next_ids, accept_lens, new_seq_lens, self.ex.graphs is not None)
        self.n_steps += 1
        if self._stats_f is not None:
            t_out = time.perf_counter()
            for (b, k, nn, c, ti, tg, g) in recs:
                rec = dict(k="d", rid=reqs[b].rid, n=nn, c=c, t_in=ti, t_gpu=tg, t_out=(t_out if k == recs[-1][1] else tg),
                           gap=(gap if k == 0 else None), bs=bs, call_k=k)
                if g is not None: rec["g"] = g
                self._stats_f.write(json.dumps(rec) + "\n")
            self._t_last_exit = t_out
        return out


def register():
    """register LATE with SGLang (idempotent; must run in the HTTP parent before ServerArgs validation, and in every child process)"""
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.speculative.spec_registry import get_spec
    if get_spec("LATE") is not None: return

    def _validate(sa):
        assert sa.disable_overlap_schedule, "LATE needs --disable-overlap-schedule (non-overlap V2 path)"
        assert sa.page_size == 1, "LATE needs --page-size 1"
        st = late_settings(); D = st["D"] * st["K"]
        assert sa.speculative_num_draft_tokens == D, ("--speculative-num-draft-tokens must be (B+1)*steps_per_call", sa.speculative_num_draft_tokens, D)

    from sglang.srt.speculative.spec_registry import CustomSpecAlgo

    class LateSpecAlgo(CustomSpecAlgo):
        """CustomSpecAlgo of 0.5.14 lacks three enum methods the scheduler calls on any spec algorithm (create_future_map in
        Scheduler.init_overlap, carries_draft_hidden_states in disaggregation / output paths, need_topk); same behaviour as NGRAM's"""
        def create_future_map(self, device, req_to_token_pool, needs_cpu_seq_lens=True):
            from sglang.srt.managers.overlap_utils import FutureMap
            return FutureMap(device, self, req_to_token_pool, needs_cpu_seq_lens)

        def carries_draft_hidden_states(self):
            return False

        def need_topk(self):
            return False

    SpeculativeAlgorithm.register("LATE", supports_overlap=False, validate_server_args=_validate, spec_class=LateSpecAlgo)(lambda server_args: LateWorker)
