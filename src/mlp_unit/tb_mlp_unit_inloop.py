"""In-loop cocotb testbench for mlp_unit.

Runs a real Granite 4.0-H-350M forward pass, hooks the first shared MLP
(GraniteMoeHybridMLP), and captures its input hidden states and output. Each
token's MLP is independent, so the first NTOK captured vectors are streamed
through the DUT (with the model's gate+up and down weights) and compared
against the model's MLP output.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer
import torch

HIDDEN = int(os.environ.get("MLP_HIDDEN", "768"))
INTER = int(os.environ.get("MLP_INTER", "2048"))
NTOK = int(os.environ.get("MLP_NTOK", "4"))
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "granite-4.0-h-350m"
PROMPT = "The quick brown fox jumps over the lazy dog"
# The DUT accumulates each dot product sequentially in fp32, while ATen uses a
# blocked GEMM order; over the 2048-term down projection the orders diverge by
# up to ~1e-2 absolute (measured ~8e-3), so the model comparison is looser. The
# comparison against the sequential-order emulation is tight.
TOL_MODEL = 2e-2
TOL_EMU = 1e-3


def capture_mlp():
    """Return (W_gu, W_dn, x, y, layer_idx) for the first shared MLP."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.eval()

    mlp_mod, layer_idx = None, None
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "GraniteMoeHybridMLP":
            mlp_mod = mod
            layer_idx = int(name.split(".")[2])
            break
    assert mlp_mod is not None, "no MLP found"

    cap = {}

    def hook(module, args, kwargs, output):
        hs = args[0] if args else kwargs.get("hidden_states")
        assert hs is not None, "hidden_states not found in MLP call"
        cap["x"] = hs.detach().clone()               # keep bf16 for the DUT
        cap["y"] = output.detach().float().clone()

    handle = mlp_mod.register_forward_hook(hook, with_kwargs=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    try:
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()

    assert cap, "MLP not executed"
    x = cap["x"][0]        # (S, hidden)
    y = cap["y"][0]        # (S, hidden)
    assert x.shape[1] == HIDDEN, f"unexpected hidden size {x.shape}"
    return (mlp_mod.input_linear.weight, mlp_mod.output_linear.weight,
            x, y, layer_idx)


def bf16_bits(t):
    return t.view(torch.uint16).flatten().tolist()


def to_f32(bits):
    import numpy as np

    arr = np.array(bits, dtype=np.uint32) << 16
    return torch.from_numpy(arr.view(np.float32))


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
async def test_mlp_unit_inloop(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut._log.info("Running Granite 4.0-H-350M forward pass (MLP capture) ...")
    W_gu, W_dn, x, y, layer_idx = capture_mlp()
    seq = min(x.shape[0], NTOK)
    dut._log.info(f"Captured layer {layer_idx} MLP: {x.shape[0]} tokens, "
                  f"streaming {seq}")

    for sel, W in ((0, W_gu), (1, W_dn)):
        out_n, in_n = W.shape
        bits = bf16_bits(W)
        for o in range(out_n):
            for i in range(in_n):
                dut.load_en.value = 1
                dut.load_sel.value = sel
                dut.load_out_idx.value = o
                dut.load_in_idx.value = i
                dut.load_wdata.value = bits[o * in_n + i]
                await RisingEdge(dut.clk)
    dut.load_en.value = 0
    dut._log.info("Weights loaded; streaming tokens ...")

    assert x.dtype == torch.bfloat16, "captured inputs must stay bf16"
    x_bits = bf16_bits(x)
    assert len(x_bits) == x.shape[0] * HIDDEN

    got = []
    for t in range(seq):
        for i in range(HIDDEN):
            await send_beat(dut, x_bits[t * HIDDEN + i], i == HIDDEN - 1)
        dut.s_axis_tvalid.value = 0
        dut.s_axis_tlast.value = 0

        beats = []
        dut.m_axis_tready.value = 1
        while len(beats) < HIDDEN:
            await RisingEdge(dut.clk)
            await Timer(1, unit="ns")
            if int(dut.m_axis_tvalid.value) == 1:
                beats.append(int(dut.m_axis_tdata.value) & 0xFFFF)
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        dut.m_axis_tready.value = 0
        got.append(beats)
        dut._log.info(f"token {t}: {len(beats)} output beats")

    dut_out = to_f32(got).reshape(seq, HIDDEN)

    # Sequential-order emulation of the RTL datapath (tight comparison).
    import gen_golden

    ref_seq = gen_golden.reference(x[:seq].bfloat16(), W_gu.bfloat16(),
                                   W_dn.bfloat16(), INTER).float()
    model_out = y[:seq].reshape(seq, HIDDEN)

    for name, ref, tol in (("sequential emulation", ref_seq, TOL_EMU),
                           ("model MLP output", model_out, TOL_MODEL)):
        diff = (dut_out - ref).abs()
        rel = diff / ref.abs().clamp_min(tol)
        bad = ((diff > tol) & (rel > tol)).sum().item()
        nonfinite = (~torch.isfinite(dut_out)).sum().item()
        dut._log.info(
            f"{name}: max_abs={diff.max():.3e} max_rel={rel.max():.3e} "
            f"outside {tol:g}/{tol:g}: {bad}/{diff.numel()} "
            f"non-finite: {nonfinite}")
        assert nonfinite == 0, f"{nonfinite} non-finite outputs"
        assert bad == 0, f"{bad} elements outside tolerance vs {name}"

    dut._log.info("PASSED: MLP in-loop outputs match the emulation and the model")
