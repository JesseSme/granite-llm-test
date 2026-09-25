"""In-loop cocotb testbench for output_projection_unit.

Runs the real Granite 4.0-H-350M forward pass, takes the final-norm hidden
states and the (tied) lm_head weight, and compares the DUT's logits against
`model.lm_head(hidden_states) / logits_scaling` for a sampled subset of the
100352 vocabulary rows. The full table (77M parameters) is far too large to
simulate, so the DUT is built with VOCAB = number of sampled rows and only
those weight rows are loaded; the sample includes each streamed position's
argmax row plus seeded random rows.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import cocotb
import torch
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "granite-4.0-h-350m"
HIDDEN = int(os.environ.get("OUTPROJ_HIDDEN", "768"))
SAMPLE = int(os.environ.get("OUTPROJ_VOCAB", "512"))
NTOK = 2
TOL = 2e-2
CLK_NS = 10


def bf16_bits(t: torch.Tensor) -> list[int]:
    return [int(v) for v in t.reshape(-1).view(torch.uint16).tolist()]


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


async def load_weight(dut, o: int, i: int, w: int) -> None:
    dut.load_en.value = 1
    dut.load_out_idx.value = o
    dut.load_in_idx.value = i
    dut.load_wdata.value = w
    await RisingEdge(dut.clk)
    dut.load_en.value = 0


@cocotb.test()
async def test_output_projection_inloop(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_dut(dut)

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.eval()
    ids = torch.arange(1, 10).unsqueeze(0)
    with torch.no_grad():
        out = model(ids, use_cache=False, output_hidden_states=True)
        hs = out.hidden_states[-1]                      # (1, S, 768) post final_norm
        logits = (model.lm_head(hs) / model.config.logits_scaling).to(torch.bfloat16)
    assert hs.dtype == torch.bfloat16
    seq = hs.shape[1]
    assert HIDDEN == hs.shape[-1]

    g = torch.Generator().manual_seed(1234)
    rows = torch.randperm(logits.shape[-1], generator=g)[:SAMPLE - NTOK].tolist()
    for t in range(NTOK):
        rows.append(int(logits[0, t].argmax()))
    rows = rows[:SAMPLE]
    dut._log.info(f"Captured {seq} tokens; sampling {SAMPLE} vocab rows, streaming {NTOK}")

    W = model.lm_head.weight.data                       # (100352, 768) bf16, tied
    dut._log.info("Loading sampled weight rows ...")
    for o, r in enumerate(rows):
        wbits = bf16_bits(W[r])
        for i in range(HIDDEN):
            await load_weight(dut, o, i, wbits[i])

    x_bits = bf16_bits(hs[0])
    mismatches = 0
    exact = 0
    max_abs = 0.0
    max_rel = 0.0
    for t in range(NTOK):
        for i in range(HIDDEN):
            dut.s_axis_tvalid.value = 1
            dut.s_axis_tdata.value = x_bits[t * HIDDEN + i]
            dut.s_axis_tlast.value = 1 if i == HIDDEN - 1 else 0
            while True:
                await RisingEdge(dut.clk)
                if int(dut.s_axis_tready.value) == 1:
                    break
        dut.s_axis_tvalid.value = 0
        dut.s_axis_tlast.value = 0

        got: list[int] = []
        dut.m_axis_tready.value = 1
        while len(got) < SAMPLE:
            await RisingEdge(dut.clk)
            await Timer(1, unit="ns")
            if int(dut.m_axis_tvalid.value) == 1:
                got.append(int(dut.m_axis_tdata.value) & 0xFFFF)
        dut.m_axis_tready.value = 0
        await RisingEdge(dut.clk)

        for o in range(SAMPLE):
            gv = bf16_to_f32(got[o])
            ev = float(logits[0, t, rows[o]])
            if got[o] == int(logits[0, t, rows[o]].view(torch.uint16).item()):
                exact += 1
            abs_err = abs(gv - ev)
            rel_err = abs_err / max(abs(ev), 1e-6)
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            if not (gv == gv and ev == ev) or (abs_err > TOL and rel_err > TOL):
                mismatches += 1
                if mismatches <= 5:
                    dut._log.error(f"token {t} row {rows[o]}: got {gv!r} exp {ev!r} "
                                   f"abs {abs_err:.3e} rel {rel_err:.3e}")

    dut._log.info(f"Bit-exact: {exact}/{NTOK * SAMPLE}")
    dut._log.info(f"Max abs error: {max_abs:.6f}")
    dut._log.info(f"Max rel error: {max_rel:.6f}")
    assert mismatches == 0, f"FAILED: {mismatches} mismatches"
    dut._log.info("In-loop PASSED")
