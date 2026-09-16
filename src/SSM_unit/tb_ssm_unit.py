"""Cocotb testbench for ssm_unit.

Loads A_log/D/dt_bias, streams `seq` token frames (x|B|C|dt) over the input
AXI-Stream (state carried between frames, no reset), collects the fp32 y
frames and compares against the float32 golden reference within tolerance
(the hardware exp/softplus are polynomial approximations).

After reset the unit clears its recurrent state sequentially
(NUM_HEADS*HEAD_DIM*D_STATE cycles); s_axis_tready stays low until that
finishes, so input beats are stalled on tready.

Dimensions come from the build via SSM_HEADS / SSM_HEAD_DIM / SSM_D_STATE.
"""

import os
import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

NH = int(os.environ.get("SSM_HEADS", "2"))
HD = int(os.environ.get("SSM_HEAD_DIM", "2"))
DS = int(os.environ.get("SSM_D_STATE", "4"))
XN = NH * HD
FRAME = XN + 2 * DS + NH


def read_hex(path):
    with open(path) as f:
        return [int(line.strip(), 16) for line in f if line.strip()]


def hex_to_f32(h):
    return struct.unpack("f", struct.pack("I", h & 0xFFFFFFFF))[0]


async def send_beat(dut, data, last):
    """AXI-Stream beat, stalled until tready (state clear holds it off)."""
    dut.s_axis_tvalid.value = 1
    dut.s_axis_tdata.value = data
    dut.s_axis_tlast.value = 1 if last else 0
    while int(dut.s_axis_tready.value) == 0:
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
    await RisingEdge(dut.clk)


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tdata.value = 0
    dut.s_axis_tlast.value = 0
    dut.m_axis_tready.value = 0
    dut.load_en.value = 0
    dut.load_sel.value = 0
    dut.load_idx.value = 0
    dut.load_wdata.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


@cocotb.test()
async def test_ssm_unit(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    a_log = read_hex(os.path.join(script_dir, "golden_a_log.hex"))
    d_val = read_hex(os.path.join(script_dir, "golden_d.hex"))
    dtb = read_hex(os.path.join(script_dir, "golden_dt_bias.hex"))
    inputs = read_hex(os.path.join(script_dir, "golden_inputs.hex"))
    expected = read_hex(os.path.join(script_dir, "golden_outputs.hex"))
    seq = len(inputs) // FRAME

    dut._log.info("Loading weights ...")
    for sel, arr in ((0, a_log), (1, d_val), (2, dtb)):
        for i, v in enumerate(arr):
            dut.load_en.value = 1
            dut.load_sel.value = sel
            dut.load_idx.value = i
            dut.load_wdata.value = v
            await RisingEdge(dut.clk)
    dut.load_en.value = 0
    dut._log.info(f"Weights loaded; streaming {seq} frames")

    total = 0
    max_abs = 0.0
    max_rel = 0.0
    mismatches = 0

    for t in range(seq):
        dut._log.info(f"frame {t}")
        for i in range(FRAME):
            await send_beat(dut, inputs[t * FRAME + i], i == FRAME - 1)
        dut.s_axis_tvalid.value = 0
        dut.s_axis_tlast.value = 0
        dut._log.info(f"frame {t} streamed; collecting")

        got = []
        dut.m_axis_tready.value = 1
        for _ in range(XN * 400 + 1000):
            await RisingEdge(dut.clk)
            await Timer(1, unit="ns")
            if int(dut.m_axis_tvalid.value) == 1:
                got.append(int(dut.m_axis_tdata.value) & 0xFFFFFFFF)
            if len(got) >= XN:
                break
        dut._log.info(f"frame {t} got {len(got)}")
        if len(got) < XN:
            dut._log.error(f"frame {t}: only {len(got)}/{XN} outputs")
            assert False, "missing outputs"
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        dut.m_axis_tready.value = 0

        for i in range(XN):
            actual = hex_to_f32(got[i])
            exp = hex_to_f32(expected[t * XN + i])
            abs_err = abs(actual - exp)
            rel_err = abs_err / max(abs(exp), 1e-3)
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            if abs_err > 0.05 and rel_err > 0.05:
                mismatches += 1
                if mismatches <= 10:
                    dut._log.warning(
                        f"token {t} elem {i}: got {actual:.6f}, expected {exp:.6f}"
                    )
        total += XN

    dut._log.info(f"Results: {total - mismatches}/{total} within tolerance")
    dut._log.info(f"Max abs error: {max_abs:.6f}")
    dut._log.info(f"Max rel error: {max_rel:.6f}")

    if mismatches > 0:
        dut._log.error(f"FAILED: {mismatches} mismatches")
        assert False, f"{mismatches} mismatches"
    dut._log.info("PASSED: SSM outputs within 5% / 0.05 abs of golden")
