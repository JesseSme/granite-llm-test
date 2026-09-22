"""Cocotb testbench for mamba2_unit.

Loads the golden weights (in_proj, out_proj, conv weight/bias, gated-norm
weight, SSM A_log/D/dt_bias), streams `seq` token frames over the input
AXI-Stream — the conv history and the SSM state carry across frames — and
compares the bf16 outputs against the golden sample.

The SSM exponential is a hardware polynomial (~1e-5 relative error), so the
comparison uses a tolerance.

Dimensions come from the build via MAMBA_HIDDEN / MAMBA_INTER / MAMBA_HEADS /
MAMBA_HEAD_DIM / MAMBA_D_STATE.
"""

import os
import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

HIDDEN = int(os.environ.get("MAMBA_HIDDEN", "8"))
HEADS = int(os.environ.get("MAMBA_HEADS", "2"))
HEAD_DIM = int(os.environ.get("MAMBA_HEAD_DIM", "2"))
D_STATE = int(os.environ.get("MAMBA_D_STATE", "2"))
INTER = HEADS * HEAD_DIM
CONV_CH = INTER + 2 * D_STATE
PROJ = INTER + CONV_CH + HEADS
# The conv1d_unit matches ATen's bf16 conv only to within ~1 bf16 ULP (its
# accumulation order differs at rounding boundaries, as verified in the conv
# unit's own in-loop test); the SSM recurrence then amplifies those ULP
# differences, so the composite comparison uses a tolerance. The post-conv
# chain is exact when driven with the DUT's own conv outputs.
TOL = 2e-2


def read_hex(path):
    with open(path) as f:
        return [int(line.strip(), 16) for line in f if line.strip()]


def bf16_to_f32(bits):
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


async def load(dut, sel, out_idx, in_idx, data):
    dut.load_en.value = 1
    dut.load_sel.value = sel
    dut.load_out_idx.value = out_idx
    dut.load_in_idx.value = in_idx
    dut.load_wdata.value = data
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_mamba2_unit(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    d = os.path.dirname(os.path.abspath(__file__))
    W_in = read_hex(os.path.join(d, "golden_in_proj_w.hex"))
    W_out = read_hex(os.path.join(d, "golden_out_proj_w.hex"))
    Wc = read_hex(os.path.join(d, "golden_conv_w.hex"))
    cb = read_hex(os.path.join(d, "golden_conv_b.hex"))
    nw = read_hex(os.path.join(d, "golden_norm_w.hex"))
    a_log = read_hex(os.path.join(d, "golden_a_log.hex"))
    dv = read_hex(os.path.join(d, "golden_d.hex"))
    dtb = read_hex(os.path.join(d, "golden_dt_bias.hex"))
    inputs = read_hex(os.path.join(d, "golden_inputs.hex"))
    expected = read_hex(os.path.join(d, "golden_outputs.hex"))
    seq = len(inputs) // HIDDEN

    dut._log.info("Loading weights ...")
    for o in range(PROJ):
        for i in range(HIDDEN):
            await load(dut, 0, o, i, W_in[o * HIDDEN + i])
    for o in range(HIDDEN):
        for i in range(INTER):
            await load(dut, 1, o, i, W_out[o * INTER + i])
    for ch in range(CONV_CH):
        for k in range(4):
            await load(dut, 2, ch, k, Wc[ch * 4 + k])
    for ch in range(CONV_CH):
        await load(dut, 3, ch, 0, cb[ch])
    for i in range(INTER):
        await load(dut, 4, i, 0, nw[i])
    for i in range(HEADS):
        await load(dut, 5, i, 0, a_log[i])
    for i in range(HEADS):
        await load(dut, 6, i, 0, dv[i])
    for i in range(HEADS):
        await load(dut, 7, i, 0, dtb[i])
    dut.load_en.value = 0

    total = 0
    mismatches = 0
    nonfinite = 0
    max_abs = 0.0
    max_rel = 0.0

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
    dut._log.info("PASSED: mamba2 outputs within tolerance of golden")
