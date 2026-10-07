#!/usr/bin/env python3
"""Build and run the cocotb regression for the matrix unit.

Usage:
  run_test.py            # unit test (small synthetic parameters)
  run_test.py inloop     # in-loop test for real Granite Linear layers
"""

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
FP_RTL = ROOT.parent.parent / "systemverilog_fp_unit" / "rtl"

RTL_SRCS = [FP_RTL / f for f in (
    "fp_pkg.sv", "fp_add.sv", "fp_add_pipe2.sv", "fp_mul.sv", "fp_mul_pipe2.sv", "fp_div.sv",
    "fp_sqrt.sv", "fp_fma.sv", "fp_minmax.sv", "fp_cmp.sv",
    "fp_totalorder.sv", "fp_roundint.sv", "fp32_to_bf16_round.sv",
    "fp_unit.sv",
    "fp_add_pipe.sv", "fp_mul_pipe.sv", "fp_mul_core.sv",
    "fp_fma_pipe.sv", "fp_minmax_pipe.sv", "fp_cmp_pipe.sv",
    "fp_totalorder_pipe.sv", "fp_roundint_pipe.sv", "fp_divsqrt_iter.sv",
)] + [ROOT.parent / "matrix_unit" / "matrix_unit.sv",
     ROOT / "output_projection_unit.sv"]

# (IN_FEATURES, OUT_FEATURES, module path, build subdir) for in-loop runs
INLOOP_CONFIGS = [
    (768, 512, None, "sim_build_inloop"),
]

UNIT_CONFIG = (32, 64, "tb_output_projection_unit", "sim_build")


def _sim_env() -> dict:
    env = dict(os.environ)
    env["PYGPI_PYTHON_BIN"] = sys.executable
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
                 startup_grace: int = 20) -> tuple[int, str, str]:
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


def run_config(IN: int, OUT: int, test_module: str, build_subdir: str,
               env: dict, timeout: int, module_path: str | None = None,
               show: bool = True) -> bool:
    build_dir = ROOT / build_subdir
    print(f"=== Building output_projection_unit HIDDEN={IN} VOCAB={OUT} -> {build_subdir} ===", flush=True)
    runner = get_runner("verilator")
    runner.build(
        sources=RTL_SRCS,
        hdl_toplevel="output_projection_unit",
        parameters={"HIDDEN": IN, "VOCAB": OUT},
        build_args=["--timing"],
        timescale=("1ns", "1ps"),
        build_dir=build_dir,
    )
    bin_path = build_dir / "output_projection_unit"
    if not bin_path.exists():
        print(f"ERROR: no simulator binary at {bin_path}", flush=True)
        return False

    run_env = dict(env)
    run_env["OUTPROJ_HIDDEN"] = str(IN)
    run_env["OUTPROJ_VOCAB"] = str(OUT)
    if module_path is not None:
        run_env["OUTPROJ_MODULE"] = module_path

    for attempt in range(1, 4):
        print(f"  attempt {attempt}:", end=" ", flush=True)
        rc, log, timing = _run_attempt(bin_path, run_env, timeout, test_module)
        if rc == 0 and "TESTS=" in log:
            m = re.search(r"\*\* TESTS=\d+ PASS=(\d+) FAIL=(\d+)", log)
            if m:
                passed, failed = int(m.group(1)), int(m.group(2))
                print(f"TESTS PASS={passed} FAIL={failed} {timing}", flush=True)
                if show:
                    for line in log.splitlines():
                        if ("Results" in line or "In-loop results" in line
                                or "Bit-exact" in line or "Max abs" in line
                                or "Max rel" in line or "passed" in line):
                            print("    " + line.strip(), flush=True)
                if failed == 0:
                    return True
                for line in log.splitlines():
                    if "ERROR" in line or "FAIL" in line:
                        print("    " + line.strip(), flush=True)
                return False
        if rc == 124:
            print(f"TIMED OUT {timing} - retrying", flush=True)
        else:
            print(f"rc={rc} {timing}", flush=True)
            for line in log.splitlines()[-30:]:
                print("    " + line.strip(), flush=True)
    return False


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "unit"
    env = _sim_env()

    if mode == "inloop":
        ok = True
        for IN, OUT, module_path, subdir in INLOOP_CONFIGS:
            ok &= run_config(IN, OUT, "tb_output_projection_unit_inloop", subdir, env,
                             timeout=3600, module_path=module_path)
        print("output_projection unit in-loop PASSED" if ok else "output_projection unit in-loop FAILED",
              flush=True)
        return 0 if ok else 1

    ok = run_config(UNIT_CONFIG[0], UNIT_CONFIG[1], UNIT_CONFIG[2],
                    UNIT_CONFIG[3], env, timeout=600)
    print("output_projection unit PASSED" if ok else "output_projection unit FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
