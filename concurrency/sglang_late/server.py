"""python -m sglang_late.server -- SGLang HTTP server decoding Ouro v2 with the two-pass late lattice (or plain AR) at batch 1.

  python -m sglang_late.server --model-path <staged v2 dir with the stock Ouro config> --late-mode l2:lkv:am:latt:4 --port 30000
  python -m sglang_late.server --model-path <dir> --late-mode ar                      # plain full-depth AR baseline (same executor family)

Launcher pattern copied from LoopSpec (sglang_recurrent/server/cli.py): parse ServerArgs, force the settings this path needs, register
the plugin in the parent, and hand SGLang our scheduler / detokenizer process entry points, which register again in each child.
Forced: --speculative-algorithm LATE, --speculative-num-draft-tokens B+1 (1 for ar), steps 1, topk 1, --disable-overlap-schedule,
--page-size 1, --disable-radix-cache, --disable-cuda-graph (no target graphs; the executor captures its own), --model-config-parser
ouro_late. Defaults: --attention-backend fa3 (metadata only), --max-running-requests 1, --mem-fraction-static 0.6, --dtype bfloat16.
Late options (exported as LATE_* env, inherited by the children):
  --late-mode MODE        l2:<lkv|old>:am:<b0|latt:B|sp:B:sets> or ar
  --late-pt-width W       page-table width of the decode attention (late.py used S = 2560; 0 = the req_to_token width)
  --late-prefill-pad N    pad a single-request prefill to a multiple of N rows (64 = late.py's graph-prefill shapes; 0 = off)
  --late-graphs 0|1       executor CUDA graphs
  --late-stats PATH       per-step timing log (jsonl) written by the worker
  --late-nv / --late-mask-id   argmax range (49153) and <|mask|> id (49152) of v2
  --late-steps-per-call K      K lattice steps per scheduler iteration (default 1; K > 1 amortizes SGLang's per-iteration host time,
                               stops early on EOS / max_new_tokens; a stop STRING is only seen by SGLang after the call)
  --late-margins 0|1      log the top-2 gap of every committed slot in the stats (diagnostic)
"""
import argparse, os, sys


def run_late_scheduler_process(server_args, *args, **kwargs):
    from .integrations import child_setup
    child_setup(server_args)
    from sglang.srt.managers import scheduler as scheduler_module
    return scheduler_module.run_scheduler_process(server_args, *args, **kwargs)


def run_late_detokenizer_process(server_args, port_args):
    from .integrations import install_config_parser
    from .spec_worker import register
    register(); install_config_parser()
    from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
    return run_detokenizer_process(server_args, port_args)


def main(argv=None):
    from sglang.srt.server_args import ServerArgs
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.set_defaults(attention_backend="fa3", dtype="bfloat16", host="127.0.0.1", port=30000, max_running_requests=1,
                        mem_fraction_static=0.6, trust_remote_code=True)
    parser.add_argument("--late-mode", default="l2:lkv:am:latt:4")
    parser.add_argument("--late-pt-width", type=int, default=2560)
    parser.add_argument("--late-prefill-pad", type=int, default=64)
    parser.add_argument("--late-graphs", type=int, default=1)
    parser.add_argument("--late-stats", default="")
    parser.add_argument("--late-nv", type=int, default=49153)
    parser.add_argument("--late-mask-id", type=int, default=49152)
    parser.add_argument("--late-steps-per-call", type=int, default=1)
    parser.add_argument("--late-margins", type=int, default=0)
    args = parser.parse_args(argv)
    if args.max_running_requests != 1:
        parser.error("M1 serves batch 1 only (--max-running-requests 1); continuous batching is M2")
    os.environ.update(LATE_MODE=args.late_mode, LATE_PT_WIDTH=str(args.late_pt_width), LATE_PREFILL_PAD=str(args.late_prefill_pad),
                      LATE_GRAPHS=str(args.late_graphs), LATE_STATS=args.late_stats, LATE_NV=str(args.late_nv),
                      LATE_MASK_ID=str(args.late_mask_id), LATE_RMAX=str(args.max_running_requests),
                      LATE_STEPS_PER_CALL=str(args.late_steps_per_call), LATE_MARGINS=str(args.late_margins))
    from .spec_worker import register, late_settings
    from .integrations import install_config_parser
    register()
    s = late_settings()
    args.speculative_algorithm = "LATE"; args.speculative_num_draft_tokens = s["D"] * s["K"]
    args.speculative_num_steps = 1; args.speculative_eagle_topk = 1
    args.disable_overlap_schedule = True; args.page_size = 1; args.disable_radix_cache = True; args.disable_cuda_graph = True
    args.model_config_parser = install_config_parser()
    server_args = ServerArgs.from_cli_args(args)
    from sglang.srt.entrypoints.http_server import launch_server
    launch_server(server_args, run_scheduler_process_func=run_late_scheduler_process,
                  run_detokenizer_process_func=run_late_detokenizer_process)


if __name__ == "__main__":
    main()
