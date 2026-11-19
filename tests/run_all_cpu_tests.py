"""Run all deterministic CPU mathematical exactness and Roofline verification gates (zero GPU required)."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

SUITES = [
    ("Wake Exactness & EOS Drain (Ouro T=4 & Huginn T=8, 258/258 fp64 gates)", ["wake_verify/test_wake_cpu.py"]),
    ("Huginn Paged-KV Fast Engine (128/128 pipe/p24/p16/sync/none/eeblk gates)", ["huginn/tests/test_hg_fast_cpu.py"]),
    ("Huginn Single & Block Reference Decoder Gates", ["huginn/tests/test_hg_eval_cpu.py"]),
    ("Ouro K-Pass Late-Entry Lattice & KV Bookkeeping Gates", ["tests/test_cpu_latek.py", "--quick", "1"]),
    ("Ouro Early-Exit Truncated Block Decoder Gates", ["tests/test_cpu_eeblk.py"]),
    ("Ouro Restart-Repair (Deep Readout on Full Rejection) Gates", ["tests/test_cpu_lkdb.py", "--n", "20"]),
    ("Ouro Multi-Request Continuous-Batching Executor (546 fp64/fp32 gates)", ["concurrency/test_cpu_batch_latek.py"]),
    ("SGLang Spec-V2 Executor & Shuffled Physical KV Equivalence Gates", ["concurrency/sglang_tests/test_cpu_core.py", "--quick", "1"]),
    ("Zero-Shot L_p Roofline Law Calibration & Concurrency B*(C) Predictor", ["wake_verify/analyze_roofline_v8.py"]),
]


def main():
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "2")
    env["NT"] = env.get("NT", "2")
    pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{pypath}" if pypath else REPO_ROOT

    print("=" * 84)
    print(f"Running Late Entry Full CPU Verification Suite ({len(SUITES)} test modules)")
    print("=" * 84)
    t_start = time.time()
    for idx, (title, cmd_args) in enumerate(SUITES, 1):
        t0 = time.time()
        print(f"\n[{idx}/{len(SUITES)}] {title}")
        cmd = [sys.executable] + [os.path.join(REPO_ROOT, cmd_args[0])] + cmd_args[1:]
        res = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        dt = time.time() - t0
        if res.returncode != 0:
            print(f"FAILED ({dt:.1f}s): {' '.join(cmd_args)}")
            sys.exit(res.returncode)
        print(f"PASSED ({dt:.1f}s)")

    print("\n" + "=" * 84)
    print(f"ALL {len(SUITES)} CPU VERIFICATION SUITES PASSED ({time.time() - t_start:.1f}s)")
    print("=" * 84)


if __name__ == "__main__":
    main()
