"""cocotb unit test for output_projection_unit.

Loads golden weights, streams SEQ hidden-state vectors and compares the
VOCAB logits per vector against the golden sample (torch bf16 Linear + /3).
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

ROOT = Path(__file__).resolve().parent
HIDDEN = int(os.environ.get("OUTPROJ_HIDDEN", "32"))
VOCAB = int(os.environ.get("OUTPROJ_VOCAB", "64"))
TOL = 2e-2
CLK_NS = 10


def read_hex(path: Path) -> list[int]:
    with open(path) as f:
        return [int(line.strip(), 16) for line in f if line.strip()]


def bf16_to_f32(bits: int) -> float:
    return struct.unpack("f", struct.pack("I", (bits & 0xFFFF) << 16))[0]


async def reset_dut(dut) -> None:
    dut.rst_n.value = 0
    dut.load_en.value = 0
    dut.load_out_idx.value = 0
    dut.load_in_idx.value = 0
    dut.load_wdata.value = 0
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tdata.value = 0
    dut.s_axis_tlast.value = 0
    dut.m_axis_tready.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_weight(dut, out_idx: int, in_idx: int, wdata: int) -> None:
    dut.load_en.value = 1
    dut.load_out_idx.value = out_idx
    dut.load_in_idx.value = in_idx
    dut.load_wdata.value = wdata
    await RisingEdge(dut.clk)
    dut.load_en.value = 0


@cocotb.test()
async def test_output_projection_unit(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_dut(dut)

    weights = read_hex(ROOT / "golden_weights.hex")
    inputs = read_hex(ROOT / "golden_inputs.hex")
    outputs = read_hex(ROOT / "golden_outputs.hex")
    assert len(weights) == VOCAB * HIDDEN
    assert len(inputs) == len(outputs) // VOCAB * HIDDEN
    seq = len(inputs) // HIDDEN

    dut._log.info(f"Loading {VOCAB}x{HIDDEN} weights ...")
    for o in range(VOCAB):
        for i in range(HIDDEN):
            await load_weight(dut, o, i, weights[o * HIDDEN + i])

    mismatches = 0
    max_abs = 0.0
    max_rel = 0.0
    exact = 0
    for t in range(seq):
        for i in range(HIDDEN):
            dut.s_axis_tvalid.value = 1
            dut.s_axis_tdata.value = inputs[t * HIDDEN + i]
            dut.s_axis_tlast.value = 1 if i == HIDDEN - 1 else 0
            while True:
                await RisingEdge(dut.clk)
                if int(dut.s_axis_tready.value) == 1:
                    break
        dut.s_axis_tvalid.value = 0
        dut.s_axis_tlast.value = 0

        got: list[int] = []
        dut.m_axis_tready.value = 1
        while len(got) < VOCAB:
            await RisingEdge(dut.clk)
            await Timer(1, unit="ns")
            if int(dut.m_axis_tvalid.value) == 1:
                got.append(int(dut.m_axis_tdata.value) & 0xFFFF)
        dut.m_axis_tready.value = 0
        await RisingEdge(dut.clk)

        if len(got) != VOCAB:
            dut._log.error(f"vector {t}: only {len(got)}/{VOCAB} outputs")
            mismatches += VOCAB - len(got)
            continue
        for i in range(VOCAB):
            g = bf16_to_f32(got[i])
            e = bf16_to_f32(outputs[t * VOCAB + i])
            if not (g == g and e == e):
                mismatches += 1
                continue
            abs_err = abs(g - e)
            rel_err = abs_err / max(abs(e), 1e-6)
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            if got[i] == outputs[t * VOCAB + i]:
                exact += 1
            elif abs_err > TOL and rel_err > TOL:
                mismatches += 1
                if mismatches <= 5:
                    dut._log.error(
                        f"vector {t} elem {i}: got {g!r} exp {e!r} "
                        f"abs {abs_err:.3e} rel {rel_err:.3e}")

    total = seq * VOCAB
    dut._log.info(f"Bit-exact: {exact}/{total}")
    dut._log.info(f"Max abs error: {max_abs:.6f}")
    dut._log.info(f"Max rel error: {max_rel:.6f}")
    assert mismatches == 0, f"FAILED: {mismatches} mismatches"
    dut._log.info("PASSED")
