#!/usr/bin/env python3
"""Build and run the cocotb regression for the MLP (SwiGLU) unit.

Usage:
  run_test.py            # unit test (small synthetic parameters)
  run_test.py inloop     # in-loop test with a real Granite MLP layer
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
REPO = ROOT.parent.parent
FP_RTL = REPO / "systemverilog_fp_unit" / "rtl"

RTL_SRCS = [FP_RTL / f for f in (
    "fp_pkg.sv", "fp_add.sv", "fp_add_pipe2.sv", "fp_mul.sv", "fp_mul_pipe2.sv", "fp_div.sv",
    "fp_sqrt.sv", "fp_fma.sv", "fp_minmax.sv", "fp_cmp.sv",
    "fp_totalorder.sv", "fp_roundint.sv", "fp32_to_bf16_round.sv",
    "fp_unit.sv",
)] + [
    ROOT.parent / "matrix_unit" / "matrix_unit.sv",     # gate+up / down
    ROOT.parent / "SSM_unit" / "fp_exp_seq.sv",         # accurate exp (silu)
    ROOT / "SwiGLU_unit" / "swiglu_unit.sv",
    ROOT / "mlp_unit.sv",
]


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


def run_config(params: dict, test_module: str, build_subdir: str, env: dict,
               timeout: int, show: bool = True) -> bool:
    build_dir = ROOT / build_subdir
    desc = " ".join(f"{k}={v}" for k, v in params.items())
    print(f"=== Building mlp_unit {desc} -> {build_subdir} ===", flush=True)
    runner = get_runner("verilator")
    runner.build(
        sources=RTL_SRCS,
        hdl_toplevel="mlp_unit",
        parameters=params,
        build_args=["--timing"],
        timescale=("1ns", "1ps"),
        build_dir=build_dir,
    )
    bin_path = build_dir / "mlp_unit"
    if not bin_path.exists():
        print(f"ERROR: no simulator binary at {bin_path}", flush=True)
        return False

    run_env = dict(env)
    run_env["MLP_HIDDEN"] = str(params["HIDDEN"])
    run_env["MLP_INTER"] = str(params["INTER"])

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
                        if ("Results" in line or "In-loop" in line
                                or "Max abs" in line or "Max rel" in line
                                or "Non-finite" in line or "passed" in line
                                or "mlp out:" in line or "emulation:" in line
                                or "model MLP output:" in line):
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
        # Granite 4.0-H-350M shared MLP dims (config.json)
        params = {"HIDDEN": 768, "INTER": 2048}
        ok = run_config(params, "tb_mlp_unit_inloop", "sim_build_inloop",
                        env, timeout=14400)
        print("mlp unit in-loop PASSED" if ok else "mlp unit in-loop FAILED",
              flush=True)
        return 0 if ok else 1

    params = {"HIDDEN": int(os.environ.get("MLP_HIDDEN", "16")),
              "INTER": int(os.environ.get("MLP_INTER", "8"))}
    ok = run_config(params, "tb_mlp_unit", "sim_build", env, timeout=3600)
    print("mlp unit PASSED" if ok else "mlp unit FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
