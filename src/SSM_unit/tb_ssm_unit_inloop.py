"""In-loop cocotb testbench for ssm_unit.

Runs a real Granite 4.0-H-350M forward pass and captures the first (layer 0)
`mamba2_chunk_scan` call — its x/B/C/dt inputs, A/D/dt_bias, and its fp32
output (the model's chunked-scan result). The captured token frames are
streamed through the DUT (recurrent form, state carried across frames) and the
DUT output is compared against

  * the model's chunk_scan output (the true in-loop golden; chunked vs
    recurrent rounding only, measured ~1.6e-6 relative offline), and
  * a vectorized fp32 recurrence computed from the same captured inputs.

The RTL exp/softplus approximations contribute ~1e-5..1e-4 relative error, so
the test requires 1e-3 absolute/relative agreement per element.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer
import torch
import torch.nn.functional as F

NH = int(os.environ.get("SSM_HEADS", "48"))
HD = int(os.environ.get("SSM_HEAD_DIM", "32"))
DS = int(os.environ.get("SSM_D_STATE", "128"))
XN = NH * HD
FRAME = XN + 2 * DS + NH
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "granite-4.0-h-350m"
PROMPT = "The quick brown fox jumps over the lazy dog"
TOL_ABS = 1e-3
TOL_REL = 1e-3


def capture_layer0_scan():
    """Forward pass; return (x, dt, A, B, C, D, dt_bias, A_log, scan_out)."""
    import transformers.models.granitemoehybrid.modeling_granitemoehybrid as gmh
    from transformers import AutoModelForCausalLM, AutoTokenizer

    captured = []
    orig = gmh.mamba2_chunk_scan

    def wrapper(*args, **kwargs):
        if not captured:
            captured.append((args, kwargs, orig(*args, **kwargs)))
        return orig(*args, **kwargs)

    gmh.mamba2_chunk_scan = wrapper
    try:
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
        model.eval()
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        ids = tok(PROMPT, return_tensors="pt").input_ids
        with torch.no_grad():
            model(ids)

        args, kwargs, res = captured[0]
        x, dt, A, B, C = args[:5]
        scan_out = res[0] if isinstance(res, tuple) else res

        layer0 = None
        for mod in model.modules():
            if mod.__class__.__name__ == "GraniteMoeHybridMambaLayer":
                layer0 = mod
                break
        assert layer0 is not None, "no Mamba layer found"
        return (x, dt, A, B, C, kwargs["D"], kwargs["dt_bias"],
                layer0.A_log.detach(), scan_out.float())
    finally:
        gmh.mamba2_chunk_scan = orig


def reference(A, D, dt_bias, x, B, C, dt):
    """Vectorized fp32 recurrence with the RTL accumulation order."""
    _, L, NH, HD = x.shape
    DS = B.shape[-1]
    A32 = A.float()
    D32 = D.float()
    h = torch.zeros(NH, HD, DS, dtype=torch.float32)
    dtp = F.softplus((dt + dt_bias).to(torch.bfloat16)).float()
    dA = torch.exp(A32[None, None, :] * dtp)
    Bv = B[0].float().expand(L, NH, DS)
    Cv = C[0].float().expand(L, NH, DS)
    xv = x[0].float()
    w = dtp[0].unsqueeze(-1) * Bv
    outs = torch.zeros(L, NH, HD)
    for t in range(L):
        h = dA[0, t][:, None, None] * h + w[t][:, None, :] * xv[t][:, :, None]
        acc = D32[:, None] * xv[t]
        for s in range(DS):
            acc = acc + h[:, :, s] * Cv[t][:, s][:, None]
        outs[t] = acc
    return outs


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


def bf16_bits(t):
    return t.view(torch.uint16).flatten().tolist()


@cocotb.test()
async def test_ssm_unit_inloop(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut._log.info("Running Granite 4.0-H-350M forward pass (layer 0 capture) ...")
    x, dt, A, B, C, D, dt_bias, A_log, scan_out = capture_layer0_scan()
    L = x.shape[1]
    dut._log.info(f"Captured sequence of {L} tokens; scanning ...")

    a_log_bits = bf16_bits(A_log)
    d_bits = bf16_bits(D)
    dtb_bits = bf16_bits(dt_bias)
    assert len(a_log_bits) == NH and len(d_bits) == NH and len(dtb_bits) == NH

    for sel, arr in ((0, a_log_bits), (1, d_bits), (2, dtb_bits)):
        for i, v in enumerate(arr):
            dut.load_en.value = 1
            dut.load_sel.value = sel
            dut.load_idx.value = i
            dut.load_wdata.value = v
            await RisingEdge(dut.clk)
    dut.load_en.value = 0

    x_bits = x[0].view(torch.uint16).tolist()      # (L, NH, HD)
    b_bits = B[0].view(torch.uint16).tolist()      # (L, 1, DS)
    c_bits = C[0].view(torch.uint16).tolist()
    dt_bits = dt[0].view(torch.uint16).tolist()    # (L, NH)

    got = []
    for t in range(L):
        frame = [b for hh in range(NH) for b in x_bits[t][hh]]
        frame += b_bits[t][0]
        frame += c_bits[t][0]
        frame += dt_bits[t]
        assert len(frame) == FRAME
        for i, v in enumerate(frame):
            await send_beat(dut, v, i == FRAME - 1)
        dut.s_axis_tvalid.value = 0
        dut.s_axis_tlast.value = 0

        beats = []
        dut.m_axis_tready.value = 1
        while len(beats) < XN:
            await RisingEdge(dut.clk)
            await Timer(1, unit="ns")
            if int(dut.m_axis_tvalid.value) == 1:
                beats.append(int(dut.m_axis_tdata.value) & 0xFFFFFFFF)
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        dut.m_axis_tready.value = 0
        got.append(beats)
        dut._log.info(f"token {t}: {len(beats)} output beats")

    import numpy as np

    dut_out = torch.from_numpy(
        np.array(got, dtype=np.uint32).view(np.float32).reshape(L, NH, HD)
    )

    ref = reference(A, D, dt_bias, x, B, C, dt)
    model_out = scan_out.reshape(L, NH, HD)

    for name, golden in (("model chunk_scan", model_out), ("fp32 recurrence", ref)):
        diff = (dut_out - golden).abs()
        rel = diff / golden.abs().clamp_min(TOL_ABS)
        bad = ((diff > TOL_ABS) & (rel > TOL_REL)).sum().item()
        dut._log.info(
            f"{name}: max_abs={diff.max():.3e} max_rel={rel.max():.3e} "
            f"outside {TOL_ABS:g}/{TOL_REL:g}: {bad}/{diff.numel()}"
        )
        assert bad == 0, f"{bad} elements outside tolerance vs {name}"

    dut._log.info("PASSED: SSM in-loop outputs within 1e-3 of model chunk_scan")
