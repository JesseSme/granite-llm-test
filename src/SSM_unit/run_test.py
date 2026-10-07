#!/usr/bin/env python3
"""Build and run the cocotb regression for the SSM unit.

Usage:
  run_test.py            # unit test (small synthetic parameters)
  run_test.py controlled # directed numeric test (A=-1, D=1, dt=0.5, 3 tokens)
  run_test.py lanes      # bit-exact LANES=1 vs LANES=4 equivalence
  run_test.py inloop     # in-loop test with a real Granite Mamba2 layer
Set SSM_LANES to override the lane count (default 4).
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
    "fp_pkg.sv", "fp_add.sv", "fp_mul.sv", "fp_div.sv",
    "fp_sqrt.sv", "fp_fma.sv", "fp_minmax.sv", "fp_cmp.sv",
    "fp_totalorder.sv", "fp_roundint.sv", "fp32_to_bf16_round.sv",
    "fp_unit.sv",
    "fp_add_pipe.sv", "fp_mul_pipe.sv", "fp_mul_core.sv",
    "fp_fma_pipe.sv", "fp_minmax_pipe.sv", "fp_cmp_pipe.sv",
    "fp_totalorder_pipe.sv", "fp_roundint_pipe.sv", "fp_divsqrt_iter.sv",
)] + [
    ROOT / "fp_softplus_seq.sv",
    ROOT / "fp_exp_seq.sv",
    ROOT / "ssm_unit.sv",
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


def run_config(NH: int, HD: int, DS: int, test_module: str, build_subdir: str,
               env: dict, timeout: int, show: bool = True,
               lanes: int = 4) -> bool:
    build_dir = ROOT / build_subdir
    print(f"=== Building ssm_unit heads={NH} head_dim={HD} d_state={DS} "
          f"lanes={lanes} -> {build_subdir} ===", flush=True)
    runner = get_runner("verilator")
    runner.build(
        sources=RTL_SRCS,
        hdl_toplevel="ssm_unit",
        parameters={"NUM_HEADS": NH, "HEAD_DIM": HD, "D_STATE": DS,
                    "LANES": lanes},
        build_args=["--timing"],
        timescale=("1ns", "1ps"),
        build_dir=build_dir,
    )
    bin_path = build_dir / "ssm_unit"
    if not bin_path.exists():
        print(f"ERROR: no simulator binary at {bin_path}", flush=True)
        return False

    run_env = dict(env)
    run_env["SSM_HEADS"] = str(NH)
    run_env["SSM_HEAD_DIM"] = str(HD)
    run_env["SSM_D_STATE"] = str(DS)

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
                                or "Max abs" in line or "Max rel" in line
                                or "Bit-exact" in line or "passed" in line
                                or "chunk_scan:" in line or "recurrence:" in line
                                or "Lane equivalence" in line):
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


def run_lanes_config(env: dict) -> bool:
    """Bit-exact LANES=1 vs LANES=4 equivalence (shared-input harness)."""
    NH, HD, DS = 3, 5, 6
    build_dir = ROOT / "sim_build_lanes"
    print(f"=== Building tb_ssm_lanes_top heads={NH} head_dim={HD} "
          f"d_state={DS} -> sim_build_lanes ===", flush=True)
    runner = get_runner("verilator")
    runner.build(
        sources=RTL_SRCS + [ROOT / "tb_ssm_lanes_top.sv"],
        hdl_toplevel="tb_ssm_lanes_top",
        parameters={"NUM_HEADS": NH, "HEAD_DIM": HD, "D_STATE": DS},
        build_args=["--timing"],
        timescale=("1ns", "1ps"),
        build_dir=build_dir,
    )
    bin_path = build_dir / "tb_ssm_lanes_top"
    if not bin_path.exists():
        print(f"ERROR: no simulator binary at {bin_path}", flush=True)
        return False

    run_env = dict(env)
    run_env["SSM_HEADS"] = str(NH)
    run_env["SSM_HEAD_DIM"] = str(HD)
    run_env["SSM_D_STATE"] = str(DS)

    for attempt in range(1, 4):
        print(f"  attempt {attempt}:", end=" ", flush=True)
        rc, log, timing = _run_attempt(bin_path, run_env, 600, "tb_ssm_lanes")
        if rc == 0 and "TESTS=" in log:
            m = re.search(r"\*\* TESTS=\d+ PASS=(\d+) FAIL=(\d+)", log)
            if m:
                passed, failed = int(m.group(1)), int(m.group(2))
                print(f"TESTS PASS={passed} FAIL={failed} {timing}", flush=True)
                for line in log.splitlines():
                    if "Lane equivalence" in line or "passed" in line:
                        print("    " + line.strip(), flush=True)
                return failed == 0
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
    lanes = int(os.environ.get("SSM_LANES", "4"))

    if mode == "inloop":
        # Layer 0 Mamba2 SSM dims from config.json
        ok = run_config(48, 32, 128, "tb_ssm_unit_inloop", "sim_build_inloop",
                        env, timeout=7200, lanes=lanes)
        print("ssm unit in-loop PASSED" if ok else "ssm unit in-loop FAILED",
              flush=True)
        return 0 if ok else 1

    if mode == "lanes":
        ok = run_lanes_config(env)
        print("ssm lane equivalence PASSED" if ok else "ssm lane equivalence FAILED",
              flush=True)
        return 0 if ok else 1

    if mode == "controlled":
        ok = run_config(2, 2, 4, "tb_controlled", "sim_build", env, timeout=600,
                        lanes=lanes)
        print("ssm controlled PASSED" if ok else "ssm controlled FAILED", flush=True)
        return 0 if ok else 1

    ok = run_config(2, 2, 4, "tb_ssm_unit", "sim_build", env, timeout=600,
                    lanes=lanes)
    print("ssm unit PASSED" if ok else "ssm unit FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
