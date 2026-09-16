"""Controlled multi-token test: A_log=0 (A=-1), D=1, dt_bias=0, dt=0.5, x=B=C=1.

Token 1: h = dtp,                      y = 1 + 4*dtp
Token 2: h = dA*dtp + dtp,             y = 1 + 4*h
Token 3: h = dA*(dA*dtp + dtp) + dtp,  y = 1 + 4*h
with dtp = bf16(softplus(0.5)) and dA = exp(-dtp) (hardware polynomial exp).
"""

import os
import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer
import torch

NH = int(os.environ.get("SSM_HEADS", "2"))
HD = int(os.environ.get("SSM_HEAD_DIM", "2"))
DS = int(os.environ.get("SSM_D_STATE", "4"))
XN = NH * HD
FRAME = XN + 2 * DS + NH
NTOK = 3


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


@cocotb.test()
async def controlled(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    dut.s_axis_tvalid.value = 0
    dut.m_axis_tready.value = 0
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    for i in range(NH):
        for sel, val in ((0, 0x0000), (1, 0x3F80), (2, 0x0000)):
            dut.load_en.value = 1
            dut.load_sel.value = sel
            dut.load_idx.value = i
            dut.load_wdata.value = val
            await RisingEdge(dut.clk)
    dut.load_en.value = 0

    results = []
    for tok in range(NTOK):
        for i in range(FRAME):
            await send_beat(dut, 0x3F00 if i >= XN + 2 * DS else 0x3F80,
                            i == FRAME - 1)
        dut.s_axis_tvalid.value = 0
        dut.s_axis_tlast.value = 0
        got = []
        dut.m_axis_tready.value = 1
        for _ in range(XN * 400 + 1000):
            await RisingEdge(dut.clk)
            await Timer(1, unit="ns")
            if int(dut.m_axis_tvalid.value) == 1:
                got.append(int(dut.m_axis_tdata.value) & 0xFFFFFFFF)
            if len(got) >= XN:
                break
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        dut.m_axis_tready.value = 0
        results.append(got)

    # Python expectation
    z = torch.tensor(0.5, dtype=torch.bfloat16)
    dtp = torch.nn.functional.softplus(z).float()
    dA = torch.exp(torch.tensor(-1.0, dtype=torch.float32) * dtp)
    h = torch.zeros(1, dtype=torch.float32)
    bad = 0
    for tok in range(NTOK):
        h = dA * h + dtp * torch.tensor(1.0, dtype=torch.float32)
        y = torch.tensor(1.0, dtype=torch.float32) + 4.0 * h
        vals = [hex_to_f32(v) for v in results[tok]]
        dut._log.info(
            f"tok {tok+1}: expected {y.item():.6f}  got {[round(v,6) for v in vals]}"
        )
        for v in vals:
            err = abs(v - y.item())
            if err > 1e-3 * max(abs(y.item()), 1.0):
                bad += 1
                dut._log.error(f"tok {tok+1}: {v} vs {y.item()}")
    assert bad == 0, f"{bad} controlled values outside 1e-3"
    await ClockCycles(dut.clk, 3)
