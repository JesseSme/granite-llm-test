#!/usr/bin/env python3
"""Build and run the cocotb regression for RMSNorm unit."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

try:
    from cocotb.runner import get_runner  # cocotb 1.9.x
except ImportError:
    from cocotb_tools.runner import get_runner  # cocotb 2.x

ROOT = Path(__file__).resolve().parent
FP_RTL = Path("systemverilog_fp_unit/rtl")

RTL_SRCS = [FP_RTL / f for f in (
    "fp_pkg.sv", "fp_add.sv", "fp_mul.sv", "fp_div.sv",
    "fp_sqrt.sv", "fp_fma.sv", "fp_minmax.sv", "fp_cmp.sv",
    "fp_totalorder.sv", "fp_roundint.sv", "fp_unit.sv",
    "fp_add_pipe.sv", "fp_mul_pipe.sv", "fp_mul_core.sv",
    "fp_fma_pipe.sv", "fp_minmax_pipe.sv", "fp_cmp_pipe.sv",
    "fp_totalorder_pipe.sv", "fp_roundint_pipe.sv", "fp_divsqrt_iter.sv", "fp32_to_bf16_round.sv",
)] + [ROOT / "rmsnorm_unit.sv"]


def _sim_env() -> dict:
    env = dict(os.environ)
    # Point PYGPI to the venv Python (not oss-cad-suite's Python 3.11)
    import sysconfig
    venv_python = sys.executable
    env["PYGPI_PYTHON_BIN"] = venv_python
    env["PYTHONPATH"] = (
        str(ROOT)
        + os.pathsep
        + sysconfig.get_paths()["purelib"]
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    libdir = sysconfig.get_config_var("LIBDIR")
    if libdir:
        env["LD_LIBRARY_PATH"] = str(libdir) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def _run_attempt(bin_path: Path, env: dict, timeout: int, test_module: str,
                 startup_grace: int = 10) -> tuple[int, str, str]:
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        log_path = f.name
    run_env = dict(env)
    run_env["COCOTB_TEST_MODULES"] = test_module
    t0 = time.time()
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        [str(bin_path)], cwd=str(bin_path.parent), env=run_env,
        stdout=log_fh, stderr=subprocess.STDOUT,
    )
    start = time.time()
    saw_output = False
    rc = None
    while proc.poll() is None:
        if os.path.getsize(log_path) > 0:
            saw_output = True
        elapsed = time.time() - start
        if elapsed > timeout or (not saw_output and elapsed > startup_grace):
            proc.kill()
            rc = 124
            break
        time.sleep(0.1)
    if rc is None:
        rc = proc.returncode
    log_fh.close()
    elapsed = time.time() - t0
    log = Path(log_path).read_text(errors="replace")
    return rc, log, f"({elapsed:.0f}s)"


def main() -> int:
    test_module = sys.argv[1] if len(sys.argv) > 1 else "tb_rmsnorm_unit"
    timeout = 600 if "inloop" in test_module else 120
    params = {"WIDTH": 768, "W_DATA": 16}
    build_dir = ROOT / "sim_build"
    env = _sim_env()

    print("=== Building RMSNorm unit ===", flush=True)
    runner = get_runner("verilator")
    runner.build(
        sources=RTL_SRCS,
        hdl_toplevel="rmsnorm_unit",
        parameters=params,
        build_args=["--timing"],
        timescale=("1ns", "1ps"),
        build_dir=build_dir,
    )
    bin_path = build_dir / "rmsnorm_unit"
    if not bin_path.exists():
        print(f"ERROR: no simulator binary at {bin_path}", flush=True)
        return 1

    retries = 3
    for attempt in range(1, retries + 1):
        print(f"  attempt {attempt}:", end=" ", flush=True)
        rc, log, timing = _run_attempt(bin_path, env, timeout, test_module)
        if rc == 0 and "TESTS=" in log:
            m = re.search(r"\*\* TESTS=\d+ PASS=(\d+) FAIL=(\d+)", log)
            if m:
                passed, failed = int(m.group(1)), int(m.group(2))
                print(f"TESTS PASS={passed} FAIL={failed} {timing}", flush=True)
                if failed == 0:
                    print("RMSNorm unit PASSED", flush=True)
                    return 0
                # Print error lines
                for line in log.splitlines():
                    if "ERROR" in line or "FAIL" in line:
                        print("    " + line.strip(), flush=True)
                return 1
        if rc == 124:
            print(f"TIMED OUT {timing} - retrying", flush=True)
        else:
            print(f"rc={rc} {timing}", flush=True)
            # Print last 30 lines of log for debugging
            for line in log.splitlines()[-30:]:
                print("    " + line, flush=True)
            if attempt < retries:
                print("  retrying...", flush=True)

    print("RMSNorm unit FAILED after retries", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
