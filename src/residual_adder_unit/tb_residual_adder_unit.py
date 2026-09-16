"""Cocotb testbench for residual_adder_unit.

Streams random branch/residual element pairs (with an idle gap between two
vectors to exercise the valid pipeline) and compares the 3-cycle-delayed
`residual + branch * 0.246` outputs bit-exactly against the golden sample.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

WIDTH = 768
N_VECTORS = 8
GAP_AFTER_VECTOR = 1
GAP_CYCLES = 3


def read_hex(path):
    with open(path) as f:
        return [int(line.strip(), 16) for line in f if line.strip()]


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.data_in.value = 0
    dut.residual_in.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


@cocotb.test()
async def test_residual_adder_unit(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    inputs = read_hex(os.path.join(script_dir, "golden_inputs.hex"))
    residuals = read_hex(os.path.join(script_dir, "golden_residuals.hex"))
    expected = read_hex(os.path.join(script_dir, "golden_outputs.hex"))

    schedule = []
    for v in range(N_VECTORS):
        for i in range(WIDTH):
            schedule.append((1, inputs[v * WIDTH + i], residuals[v * WIDTH + i]))
        if v == GAP_AFTER_VECTOR:
            for _ in range(GAP_CYCLES):
                schedule.append((0, 0, 0))

    total = len(inputs)
    sent = 0
    got = []

    for _ in range(len(schedule) + 16):
        if sent < len(schedule):
            valid, data, res = schedule[sent]
            dut.valid_in.value = valid
            dut.data_in.value = data
            dut.residual_in.value = res
            sent += 1
        else:
            dut.valid_in.value = 0
        await RisingEdge(dut.clk)
        if int(dut.valid_out.value) == 1:
            got.append(int(dut.data_out.value) & 0xFFFF)
        if len(got) >= total:
            break

    mismatches = 0
    for i in range(min(len(got), total)):
        if got[i] != expected[i]:
            mismatches += 1
            if mismatches <= 10:
                dut._log.warning(
                    f"element {i}: got 0x{got[i]:04x}, expected 0x{expected[i]:04x}"
                )

    if len(got) != total:
        dut._log.error(f"Only collected {len(got)}/{total} outputs")
        mismatches += total - len(got)

    dut._log.info(f"Results: {total - mismatches}/{total} passed")
    if mismatches > 0:
        dut._log.error(f"FAILED: {mismatches} mismatches")
        assert False, f"{mismatches} mismatches"
    dut._log.info("PASSED: all residual adds bit-exact")
