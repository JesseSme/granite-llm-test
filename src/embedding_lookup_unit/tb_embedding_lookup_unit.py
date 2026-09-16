"""Cocotb testbench for embedding_lookup_unit.

Loads the golden table rows through the load port, then feeds each token ID
and compares the streamed 768 bfloat16 outputs bit-exactly against the
`row * 12.0` golden sample.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

DIM = 768


def read_hex(path):
    with open(path) as f:
        return [int(line.strip(), 16) for line in f if line.strip()]


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.token_id.value = 0
    dut.load_en.value = 0
    dut.load_token.value = 0
    dut.load_idx.value = 0
    dut.load_wdata.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


@cocotb.test()
async def test_embedding_lookup_unit(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tokens = read_hex(os.path.join(script_dir, "golden_tokens.hex"))
    weights = read_hex(os.path.join(script_dir, "golden_weights.hex"))
    expected = read_hex(os.path.join(script_dir, "golden_outputs.hex"))

    dut._log.info(f"Loading {len(tokens)} table rows ...")
    for k, tok in enumerate(tokens):
        for i in range(DIM):
            dut.load_en.value = 1
            dut.load_token.value = tok
            dut.load_idx.value = i
            dut.load_wdata.value = weights[k * DIM + i]
            await RisingEdge(dut.clk)
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 2)

    mismatches = 0
    total = 0

    for k, tok in enumerate(tokens):
        while int(dut.busy.value) == 1:
            await RisingEdge(dut.clk)

        dut.valid_in.value = 1
        dut.token_id.value = tok
        await RisingEdge(dut.clk)
        dut.valid_in.value = 0

        got = []
        for _ in range(DIM * 4 + 16):
            await RisingEdge(dut.clk)
            if int(dut.valid_out.value) == 1:
                got.append(int(dut.data_out.value) & 0xFFFF)
                if len(got) == DIM:
                    break

        if len(got) != DIM:
            dut._log.error(f"token 0x{tok:05x}: only {len(got)}/{DIM} outputs")
            mismatches += DIM - len(got)
            total += DIM
            continue

        for i in range(DIM):
            exp = expected[k * DIM + i]
            if got[i] != exp:
                mismatches += 1
                if mismatches <= 10:
                    dut._log.warning(
                        f"token 0x{tok:05x} elem {i}: got 0x{got[i]:04x}, expected 0x{exp:04x}"
                    )
        total += DIM

    dut._log.info(f"Results: {total - mismatches}/{total} passed")
    if mismatches > 0:
        dut._log.error(f"FAILED: {mismatches} mismatches")
        assert False, f"{mismatches} mismatches"
    dut._log.info("PASSED: all embedding lookups bit-exact")
