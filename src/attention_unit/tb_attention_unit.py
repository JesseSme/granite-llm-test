"""Cocotb testbench for attention_unit.

Loads the golden Q/K/V/O weights, streams `seq` token frames (hidden states)
over the input AXI-Stream — the DUT keeps a KV cache across frames, so the
frames represent successive positions of one sequence — collects the bf16
attention outputs and compares them against the golden sample.

Only the softmax exp is approximated in hardware (poly exp, ~1e-5 relative),
so the comparison uses a tolerance rather than bit-exact matching.

Dimensions come from the build via ATT_HIDDEN / ATT_HEADS / ATT_KV_HEADS /
ATT_HEAD_DIM / ATT_MAX_SEQ.
"""

import os
import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

HIDDEN = int(os.environ.get("ATT_HIDDEN", "16"))
HEADS = int(os.environ.get("ATT_HEADS", "2"))
KV_HEADS = int(os.environ.get("ATT_KV_HEADS", "1"))
HEAD_DIM = int(os.environ.get("ATT_HEAD_DIM", "8"))
MAX_SEQ = int(os.environ.get("ATT_MAX_SEQ", "4"))
KN = KV_HEADS * HEAD_DIM
TOL = 1e-3


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
    """AXI-Stream beat, stalled until tready."""
    dut.s_axis_tvalid.value = 1
    dut.s_axis_tdata.value = data
    dut.s_axis_tlast.value = 1 if last else 0
    while int(dut.s_axis_tready.value) == 0:
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_attention_unit(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    q_w = read_hex(os.path.join(script_dir, "golden_q_w.hex"))
    k_w = read_hex(os.path.join(script_dir, "golden_k_w.hex"))
    v_w = read_hex(os.path.join(script_dir, "golden_v_w.hex"))
    o_w = read_hex(os.path.join(script_dir, "golden_o_w.hex"))
    inputs = read_hex(os.path.join(script_dir, "golden_inputs.hex"))
    expected = read_hex(os.path.join(script_dir, "golden_outputs.hex"))
    seq = len(inputs) // HIDDEN
    assert seq <= MAX_SEQ, "golden sequence longer than the KV cache"

    dut._log.info("Loading weights ...")
    for sel, arr, out_n in ((0, q_w, HIDDEN), (1, k_w, KN), (2, v_w, KN),
                            (3, o_w, HIDDEN)):
        for o in range(out_n):
            for i in range(HIDDEN):
                dut.load_en.value = 1
                dut.load_sel.value = sel
                dut.load_out_idx.value = o
                dut.load_in_idx.value = i
                dut.load_wdata.value = arr[o * HIDDEN + i]
                await RisingEdge(dut.clk)
    dut.load_en.value = 0

    total = 0
    mismatches = 0
    max_abs = 0.0
    max_rel = 0.0
    nonfinite = 0

    for t in range(seq):
        dut._log.info(f"token {t}")
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
            dut._log.error(f"token {t}: only {len(got)}/{HIDDEN} outputs")
            mismatches += HIDDEN - len(got)
            total += HIDDEN
            continue

        for i in range(HIDDEN):
            actual = bf16_to_f32(got[i])
            exp = bf16_to_f32(expected[t * HIDDEN + i])
            if actual != actual or actual in (float("inf"), float("-inf")):
                nonfinite += 1
                if nonfinite <= 5:
                    dut._log.error(f"token {t} elem {i}: non-finite 0x{got[i]:04x}")
            abs_err = abs(actual - exp)
            rel_err = abs_err / max(abs(exp), 1e-3)
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            if abs_err > TOL and rel_err > TOL:
                mismatches += 1
                if mismatches <= 10:
                    dut._log.warning(
                        f"token {t} elem {i}: got {actual:.6f}, expected {exp:.6f}")
        total += HIDDEN

    dut._log.info(f"Results: {total - mismatches}/{total} within tolerance")
    dut._log.info(f"Max abs error: {max_abs:.6f}")
    dut._log.info(f"Max rel error: {max_rel:.6f}")
    dut._log.info(f"Non-finite outputs: {nonfinite}")
    assert nonfinite == 0, f"{nonfinite} non-finite outputs"
    if mismatches > 0:
        dut._log.error(f"FAILED: {mismatches} mismatches")
        assert False, f"{mismatches} mismatches"
    dut._log.info("PASSED: attention outputs within tolerance of golden")
