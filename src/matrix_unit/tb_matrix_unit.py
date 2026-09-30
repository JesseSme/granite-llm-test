"""Cocotb testbench for matrix_unit (AXI-Stream in/out).

Loads the golden weights/biases, streams each input vector over the input
AXI-Stream (with random bubbles), accepts the output vector under downstream
backpressure (random stalls), and compares bit-exactly against the golden
sample. Dimensions come from the build via MATRIX_IN/MATRIX_OUT env vars.
"""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

IN = int(os.environ.get("MATRIX_IN", "32"))
OUT = int(os.environ.get("MATRIX_OUT", "16"))
GOLDEN = os.environ.get("MATRIX_GOLDEN", "golden")

BUSY_CYCLES = 0


async def _count_busy(dut):
    global BUSY_CYCLES
    while True:
        await RisingEdge(dut.clk)
        if int(dut.busy.value) == 1:
            BUSY_CYCLES += 1


def read_hex(path):
    with open(path) as f:
        return [int(line.strip(), 16) for line in f if line.strip()]


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tdata.value = 0
    dut.s_axis_tlast.value = 0
    dut.m_axis_tready.value = 0
    dut.load_en.value = 0
    dut.load_out_idx.value = 0
    dut.load_in_idx.value = 0
    dut.load_wdata.value = 0
    dut.load_is_bias.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


@cocotb.test()
async def test_matrix_unit(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    cocotb.start_soon(_count_busy(dut))
    await reset_dut(dut)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    inputs = read_hex(os.path.join(script_dir, f"{GOLDEN}_inputs.hex"))
    weights = read_hex(os.path.join(script_dir, f"{GOLDEN}_weights.hex"))
    biases = read_hex(os.path.join(script_dir, f"{GOLDEN}_biases.hex"))
    expected = read_hex(os.path.join(script_dir, f"{GOLDEN}_outputs.hex"))
    n_vectors = len(inputs) // IN

    dut._log.info(f"Loading {OUT}x{IN} weights ...")
    for o in range(OUT):
        for i in range(IN):
            dut.load_en.value = 1
            dut.load_out_idx.value = o
            dut.load_in_idx.value = i
            dut.load_wdata.value = weights[o * IN + i]
            dut.load_is_bias.value = 0
            await RisingEdge(dut.clk)
    for o in range(OUT):
        dut.load_en.value = 1
        dut.load_out_idx.value = o
        dut.load_wdata.value = biases[o]
        dut.load_is_bias.value = 1
        await RisingEdge(dut.clk)
    dut.load_en.value = 0

    random.seed(1234)
    mismatches = 0
    total = 0

    for v in range(n_vectors):
        dut._log.info(f"Vector {v}")
        # ---- input stream with random bubbles ----
        for i in range(IN):
            while random.random() < 0.3:
                dut.s_axis_tvalid.value = 0
                await RisingEdge(dut.clk)
            dut.s_axis_tvalid.value = 1
            dut.s_axis_tdata.value = inputs[v * IN + i]
            dut.s_axis_tlast.value = 1 if i == IN - 1 else 0
            await RisingEdge(dut.clk)
        dut.s_axis_tvalid.value = 0
        dut.s_axis_tlast.value = 0

        # ---- output stream with a randomized stall ----
        stall_at = random.randrange(OUT)
        stall_len = random.randrange(1, 5)
        got = []
        got_last = []
        stall_done = False
        ready = True
        while len(got) < OUT:
            if len(got) == stall_at and not stall_done:
                ready = False
                stall_len -= 1
                if stall_len <= 0:
                    stall_done = True
            else:
                ready = True
            dut.m_axis_tready.value = 1 if ready else 0
            await RisingEdge(dut.clk)
            await Timer(1, unit="ns")
            if ready and int(dut.m_axis_tvalid.value) == 1:
                got.append(int(dut.m_axis_tdata.value) & 0xFFFF)
                got_last.append(int(dut.m_axis_tlast.value))
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        dut.m_axis_tready.value = 0

        if len(got) < OUT:
            dut._log.error(f"Vector {v}: only {len(got)}/{OUT} outputs")
            mismatches += OUT - len(got)
            total += OUT
            continue

        for o in range(OUT):
            exp = expected[v * OUT + o]
            if got[o] != exp:
                mismatches += 1
                if mismatches <= 10:
                    dut._log.warning(
                        f"Vector {v} out {o}: got 0x{got[o]:04x}, expected 0x{exp:04x}"
                    )
            exp_last = 1 if o == OUT - 1 else 0
            if got_last[o] != exp_last:
                mismatches += 1
                dut._log.error(
                    f"Vector {v} out {o}: tlast={got_last[o]}, expected {exp_last}"
                )
        total += OUT

        # wait for the unit to return to idle before the next vector
        while int(dut.busy.value) == 1:
            await RisingEdge(dut.clk)

    dut._log.info(f"Results: {total - mismatches}/{total} passed")
    dut._log.info(f"Busy cycles (total, incl. weight load): {BUSY_CYCLES}")
    if mismatches > 0:
        dut._log.error(f"FAILED: {mismatches} mismatches")
        assert False, f"{mismatches} mismatches"
    dut._log.info("PASSED: all matrix outputs bit-exact")
