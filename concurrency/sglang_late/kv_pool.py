"""kv_pool.py -- LoopStridedMHAPool: SGLang's MHATokenToKVPool with the late lattice's loop-strided physical layout.

The model declares T*NL = 192 logical layers (OuroLateForCausalLM.end_layer), so ModelRunner profiles and sizes the pool for 192 layers
at one base slot per position. This subclass allocates, per PHYSICAL layer l, one flat buffer [T*STRIDE, NH, HD] (K and V), STRIDE =
size + page_size + SCR; loop t of base slot s lives at t*STRIDE + s; [size + page_size, STRIDE) of each region is placeholder scratch.
SGLang's per-logical-layer buffers k_buffer[t*NL + l] are contiguous VIEWS of region t, so get/set_kv_buffer keep their meaning.
`self.flat` (core_ops.FlatKV) is what the late executors and the model's own forward use.
install_pool_patch() replaces the name MHATokenToKVPool inside model_runner_kv_cache_mixin (imported by name there), in the scheduler
process, before ModelRunner builds its pools.
"""
import logging
import torch

logger = logging.getLogger(__name__)
CONFIG = dict(T=None, NL=None, SCR=0)


def install_pool_patch(T, NL, scr):
    from sglang.srt.model_executor import model_runner_kv_cache_mixin as mix
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    CONFIG.update(T=T, NL=NL, SCR=int(scr))
    if getattr(mix.MHATokenToKVPool, "_late_patched", False): return
    base = mix.MHATokenToKVPool
    assert base is MHATokenToKVPool, base

    class LoopStridedMHAPool(base):
        _late_patched = True

        def _create_buffers(self):
            T, NL, SCR = CONFIG["T"], CONFIG["NL"], CONFIG["SCR"]
            if T is None or self.layer_num != T * NL or getattr(self, "kv_cache_layout", "nhd") != "nhd":
                logger.info(f"[late] LoopStridedMHAPool: layer_num {self.layer_num} != T*NL -> stock buffers")
                self.flat = None
                return super()._create_buffers()
            from contextlib import nullcontext
            from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
            m = self.size + self.page_size; stride = m + SCR
            with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                with (torch.cuda.use_mem_pool(self.custom_mem_pool) if self.enable_custom_mem_pool else nullcontext()):
                    self.store_k = [torch.zeros((T * stride, self.head_num, self.head_dim), dtype=self.store_dtype, device=self.device) for _ in range(NL)]
                    self.store_v = [torch.zeros((T * stride, self.head_num, self.v_head_dim), dtype=self.store_dtype, device=self.device) for _ in range(NL)]
            self.k_buffer = [self.store_k[l][t * stride : t * stride + m] for t in range(T) for l in range(NL)]
            self.v_buffer = [self.store_v[l][t * stride : t * stride + m] for t in range(T) for l in range(NL)]
            self.k_data_ptrs = torch.tensor([x.data_ptr() for x in self.k_buffer], dtype=torch.uint64, device=self.device)
            self.v_data_ptrs = torch.tensor([x.data_ptr() for x in self.v_buffer], dtype=torch.uint64, device=self.device)
            self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
            import numpy as np
            self.data_strides = torch.tensor([int(np.prod(x.shape[1:])) * x.dtype.itemsize for x in self.k_buffer + self.v_buffer], device=self.device)
            from .core_ops import FlatKV
            self.flat = FlatKV([x.view(T * stride, 1, self.head_num, self.head_dim) for x in self.store_k],
                               [x.view(T * stride, 1, self.head_num, self.v_head_dim) for x in self.store_v], T, stride, m, SCR)
            logger.info(f"[late] LoopStridedMHAPool: {NL} physical layers x T={T} regions, size {self.size}, stride {stride} (scratch {SCR}), "
                        f"{2 * NL * T * stride * self.head_num * self.head_dim * self.store_dtype.itemsize / 2**30:.2f} GiB")

        def _clear_buffers(self):
            for n in ("k_buffer", "v_buffer", "store_k", "store_v"):
                if hasattr(self, n): delattr(self, n)

    mix.MHATokenToKVPool = LoopStridedMHAPool
    logger.info(f"[late] installed LoopStridedMHAPool (T={T}, NL={NL}, scratch {scr})")
    return LoopStridedMHAPool
