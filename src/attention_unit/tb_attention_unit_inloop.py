"""In-loop cocotb testbench for attention_unit.

Runs a real Granite 4.0-H-350M forward pass with the eager attention
implementation, hooks the first attention layer (layer 10, one of the
GraniteMoeHybridAttention layers), and captures its input hidden states and
its output. The captured token vectors are streamed through the DUT in order
(the DUT keeps the KV cache, so the frames are the successive positions of the
sequence) and the DUT output is compared against the model's attention output.

The projection weights are taken from the model's q/k/v/o_proj parameters, so
the projections are bit-comparable; the remaining difference comes from the
softmax exp approximation and ATen's matmul accumulation order.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer
import torch

HIDDEN = int(os.environ.get("ATT_HIDDEN", "768"))
HEADS = int(os.environ.get("ATT_HEADS", "12"))
KV_HEADS = int(os.environ.get("ATT_KV_HEADS", "4"))
HEAD_DIM = int(os.environ.get("ATT_HEAD_DIM", "64"))
MAX_SEQ = int(os.environ.get("ATT_MAX_SEQ", "64"))
KN = KV_HEADS * HEAD_DIM
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "granite-4.0-h-350m"
PROMPT = "The quick brown fox jumps over the lazy dog"
TOL_ABS = 1e-3
TOL_REL = 1e-3


def capture_attention():
    """Return (q_w, k_w, v_w, o_w, x, y, layer_idx) for the first attention layer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.eval()

    layer_mod, layer_idx = None, None
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "GraniteMoeHybridAttention":
            layer_mod = mod
            layer_idx = int(name.split(".")[2])
            break
    assert layer_mod is not None, "no attention layer found"

    cap = {}

    def hook(module, args, kwargs, output):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        assert hs is not None, "hidden_states not found in attention call"
        cap["x"] = hs.detach().clone()                 # keep bf16 for the DUT
        cap["y"] = output[0].detach().float().clone()

    handle = layer_mod.register_forward_hook(hook, with_kwargs=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    try:
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()

    assert cap, "attention layer not executed"
    x = cap["x"][0]        # (S, hidden)
    y = cap["y"][0]        # (S, hidden)
    assert x.shape[1] == HIDDEN, f"unexpected hidden size {x.shape}"
    return (layer_mod.q_proj.weight, layer_mod.k_proj.weight,
            layer_mod.v_proj.weight, layer_mod.o_proj.weight,
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
async def test_attention_unit_inloop(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut._log.info("Running Granite 4.0-H-350M forward pass (eager attention) ...")
    q_w, k_w, v_w, o_w, x, y, layer_idx = capture_attention()
    seq = x.shape[0]
    dut._log.info(f"Captured layer {layer_idx} attention: {seq} tokens")

    for sel, W, out_n in ((0, q_w, HIDDEN), (1, k_w, KN), (2, v_w, KN),
                          (3, o_w, HIDDEN)):
        assert W.shape == (out_n, HIDDEN), f"unexpected weight shape {W.shape}"
        bits = bf16_bits(W)
        for o in range(out_n):
            for i in range(HIDDEN):
                dut.load_en.value = 1
                dut.load_sel.value = sel
                dut.load_out_idx.value = o
                dut.load_in_idx.value = i
                dut.load_wdata.value = bits[o * HIDDEN + i]
                await RisingEdge(dut.clk)
    dut.load_en.value = 0
    dut._log.info("Weights loaded; streaming tokens ...")

    assert x.dtype == torch.bfloat16, "captured inputs must stay bf16"
    x_bits = bf16_bits(x)
    assert len(x_bits) == seq * HIDDEN, "input bit count mismatch"
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
    ref = y.reshape(seq, HIDDEN)

    diff = (dut_out - ref).abs()
    rel = diff / ref.abs().clamp_min(TOL_ABS)
    bad = ((diff > TOL_ABS) & (rel > TOL_REL)).sum().item()
    dut._log.info(f"attention out: max_abs={diff.max():.3e} "
                  f"max_rel={rel.max():.3e} outside {TOL_ABS:g}/{TOL_REL:g}: "
                  f"{bad}/{diff.numel()}")
    assert bad == 0, f"{bad} elements outside tolerance"

    dut._log.info("PASSED: attention in-loop outputs within 1e-3 of the model")
