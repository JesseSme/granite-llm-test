"""Cocotb testbench for mlp_unit (GraniteMoeHybridMLP / SwiGLU).

Loads the golden gate+up and down weights, streams `seq` input vectors over the
input AXI-Stream, collects the bf16 outputs and compares them against the fp32
golden sample. The activation reuses the LUT-based silu_unit (sigmoid LUT
accuracy < 0.5%), so the comparison uses a tolerance.

Dimensions come from the build via MLP_HIDDEN / MLP_INTER.
"""

import os
import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

HIDDEN = int(os.environ.get("MLP_HIDDEN", "16"))
INTER = int(os.environ.get("MLP_INTER", "8"))
TOL = 1e-2


def read_hex(path):
    with open(path) as f:
        return [int(line.strip(), 16) for line in f if line.strip()]


def bf16_to_f32(bits):
    """bfloat16 bits -> float32 (bits are the upper half of the fp32 word)."""
    return struct.unpack("f", struct.pack("I", (bits & 0xFFFF) << 16))[0]


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tdata.value = 0
    dut.s_axis_tlast.value = 0
    dut.m_axis_tready.value = 0
    dut.load_en.value = 0
    dut.load_sel.value = 0
    dut.load_out_idx.value = 0
    dut.load_in_idx.value = 0
    dut.load_wdata.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def send_beat(dut, data, last):
    dut.s_axis_tvalid.value = 1
    dut.s_axis_tdata.value = data
    dut.s_axis_tlast.value = 1 if last else 0
    while int(dut.s_axis_tready.value) == 0:
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_mlp_unit(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gu_w = read_hex(os.path.join(script_dir, "golden_gateup_w.hex"))
    dn_w = read_hex(os.path.join(script_dir, "golden_down_w.hex"))
    inputs = read_hex(os.path.join(script_dir, "golden_inputs.hex"))
    expected = read_hex(os.path.join(script_dir, "golden_outputs.hex"))
    seq = len(inputs) // HIDDEN

    dut._log.info("Loading weights ...")
    for o in range(2 * INTER):
        for i in range(HIDDEN):
            dut.load_en.value = 1
            dut.load_sel.value = 0
            dut.load_out_idx.value = o
            dut.load_in_idx.value = i
            dut.load_wdata.value = gu_w[o * HIDDEN + i]
            await RisingEdge(dut.clk)
    for o in range(HIDDEN):
        for i in range(INTER):
            dut.load_en.value = 1
            dut.load_sel.value = 1
            dut.load_out_idx.value = o
            dut.load_in_idx.value = i
            dut.load_wdata.value = dn_w[o * INTER + i]
            await RisingEdge(dut.clk)
    dut.load_en.value = 0

    total = 0
    mismatches = 0
    nonfinite = 0
    max_abs = 0.0
    max_rel = 0.0

    for t in range(seq):
        dut._log.info(f"vector {t}")
        for i in range(HIDDEN):
            await send_beat(dut, inputs[t * HIDDEN + i], i == HIDDEN - 1)
        dut.s_axis_tvalid.value = 0
        dut.s_axis_tlast.value = 0

        got = []
        dut.m_axis_tready.value = 1
        while len(got) < HIDDEN:
            await RisingEdge(dut.clk)
            await Timer(1, unit="ns")
            if int(dut.m_axis_tvalid.value) == 1:
                got.append(int(dut.m_axis_tdata.value) & 0xFFFF)
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        dut.m_axis_tready.value = 0

        if len(got) < HIDDEN:
            dut._log.error(f"vector {t}: only {len(got)}/{HIDDEN} outputs")
            mismatches += HIDDEN - len(got)
            total += HIDDEN
            continue

        for i in range(HIDDEN):
            actual = bf16_to_f32(got[i])
            exp = bf16_to_f32(expected[t * HIDDEN + i])
            if actual != actual or actual in (float("inf"), float("-inf")):
                nonfinite += 1
                if nonfinite <= 5:
                    dut._log.error(f"vector {t} elem {i}: non-finite 0x{got[i]:04x}")
            abs_err = abs(actual - exp)
            rel_err = abs_err / max(abs(exp), 1e-3)
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            if abs_err > TOL and rel_err > TOL:
                mismatches += 1
                if mismatches <= 10:
                    dut._log.warning(
                        f"vector {t} elem {i}: got {actual:.6f}, expected {exp:.6f}")
        total += HIDDEN

    dut._log.info(f"Results: {total - mismatches}/{total} within tolerance")
    dut._log.info(f"Max abs error: {max_abs:.6f}")
    dut._log.info(f"Max rel error: {max_rel:.6f}")
    dut._log.info(f"Non-finite outputs: {nonfinite}")
    assert nonfinite == 0, f"{nonfinite} non-finite outputs"
    if mismatches > 0:
        dut._log.error(f"FAILED: {mismatches} mismatches")
        assert False, f"{mismatches} mismatches"
    dut._log.info("PASSED: MLP outputs within tolerance of golden")
