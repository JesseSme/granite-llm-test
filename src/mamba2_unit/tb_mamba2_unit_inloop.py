"""In-loop cocotb testbench for mamba2_unit.

Runs a real Granite 4.0-H-350M forward pass, hooks the first Mamba mixer
(GraniteMoeHybridMambaLayer, layer 0), and captures its input hidden states and
output. The first NTOK tokens are streamed through the DUT (the conv history and
SSM state carry across frames, and each token's output depends only on tokens
up to it, so a prefix of the captured sequence is a valid in-loop test) and
compared against the model's mixer output.

The model's conv runs through ATen (the RTL conv matches it to ~1 bf16 ULP) and
its SSM is the chunked scan (the RTL is the recurrence with a polynomial exp),
so the comparison uses a tolerance.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer
import torch

HIDDEN = int(os.environ.get("MAMBA_HIDDEN", "768"))
HEADS = int(os.environ.get("MAMBA_HEADS", "48"))
HEAD_DIM = int(os.environ.get("MAMBA_HEAD_DIM", "32"))
D_STATE = int(os.environ.get("MAMBA_D_STATE", "128"))
INTER = HEADS * HEAD_DIM
CONV_CH = INTER + 2 * D_STATE
NTOK = int(os.environ.get("MAMBA_NTOK", "5"))
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "granite-4.0-h-350m"
PROMPT = "The quick brown fox jumps over the lazy dog"
TOL = 2e-2


def capture_mamba():
    """Return the layer-0 mixer weights, its input and its output."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.eval()

    mixer, layer_idx = None, None
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "GraniteMoeHybridMambaLayer":
            mixer = mod
            layer_idx = int(name.split(".")[2])
            break
    assert mixer is not None, "no Mamba layer found"

    cap = {}

    def hook(module, args, kwargs, output):
        hs = args[0] if args else kwargs.get("hidden_states")
        assert hs is not None, "hidden_states not found in mixer call"
        cap["x"] = hs.detach().clone()
        out = output[0] if isinstance(output, tuple) else output
        cap["y"] = out.detach().float().clone()

    handle = mixer.register_forward_hook(hook, with_kwargs=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    try:
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()

    assert cap, "mixer not executed"
    x = cap["x"][0]
    y = cap["y"][0]
    assert x.shape[1] == HIDDEN, f"unexpected hidden size {x.shape}"
    return mixer, x, y, layer_idx


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


async def load(dut, sel, out_idx, in_idx, data):
    dut.load_en.value = 1
    dut.load_sel.value = sel
    dut.load_out_idx.value = out_idx
    dut.load_in_idx.value = in_idx
    dut.load_wdata.value = data
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_mamba2_unit_inloop(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut._log.info("Running Granite 4.0-H-350M forward pass (Mamba capture) ...")
    mixer, x, y, layer_idx = capture_mamba()
    seq = min(x.shape[0], NTOK)
    dut._log.info(f"Captured layer {layer_idx} mixer: {x.shape[0]} tokens, "
                  f"streaming {seq}")

    W_in = mixer.in_proj.weight          # (PROJ, HIDDEN)
    W_out = mixer.out_proj.weight        # (HIDDEN, INTER)
    Wc = mixer.conv1d.weight.squeeze(1)  # (CONV_CH, KERNEL)
    cb = mixer.conv1d.bias
    nw = mixer.norm.weight
    a_log = mixer.A_log
    dv = mixer.D
    dtb = mixer.dt_bias

    # in_proj
    bits = bf16_bits(W_in)
    for o in range(W_in.shape[0]):
        for i in range(HIDDEN):
            await load(dut, 0, o, i, bits[o * HIDDEN + i])
    # out_proj
    bits = bf16_bits(W_out)
    for o in range(W_out.shape[0]):
        for i in range(INTER):
            await load(dut, 1, o, i, bits[o * INTER + i])
    # conv weight: the RTL tap k multiplies x[t-k], torch tap j multiplies
    # x[t-3+j], so load tap k with the model's weight[..., 3-k]
    bits = bf16_bits(Wc)
    for ch in range(CONV_CH):
        for k in range(4):
            await load(dut, 2, ch, k, bits[ch * 4 + (3 - k)])
    bits = bf16_bits(cb)
    for ch in range(CONV_CH):
        await load(dut, 3, ch, 0, bits[ch])
    bits = bf16_bits(nw)
    for i in range(INTER):
        await load(dut, 4, i, 0, bits[i])
    for sel, t in ((5, a_log), (6, dv), (7, dtb)):
        bits = bf16_bits(t)
        for i in range(HEADS):
            await load(dut, sel, i, 0, bits[i])
    dut.load_en.value = 0
    dut._log.info("Weights loaded; streaming tokens ...")

    x_bits = bf16_bits(x)
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
    ref = y[:seq].reshape(seq, HIDDEN)

    diff = (dut_out - ref).abs()
    rel = diff / ref.abs().clamp_min(TOL)
    bad = ((diff > TOL) & (rel > TOL)).sum().item()
    nonfinite = (~torch.isfinite(dut_out)).sum().item()
    dut._log.info(f"mamba out: max_abs={diff.max():.3e} max_rel={rel.max():.3e} "
                  f"outside {TOL:g}/{TOL:g}: {bad}/{diff.numel()} "
                  f"non-finite: {nonfinite}")
    assert nonfinite == 0, f"{nonfinite} non-finite outputs"
    assert bad == 0, f"{bad} elements outside tolerance"

    dut._log.info("PASSED: mamba2 in-loop outputs within 2e-2 of the model")
