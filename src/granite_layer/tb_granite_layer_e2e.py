"""End-to-end hybrid test: one RTL layer, 31 layers in software.

Runs the real Granite 4.0-H-350M forward pass, captures layer 0's input and
output, loads layer 0's real weights into the DUT (input_layernorm, mamba2
mixer, post_attention_layernorm, MLP) and streams exactly ONE token. The DUT
output is compared against the model's layer-0 output for that token.
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
HIDDEN = int(os.environ.get("MAMBA_HIDDEN", "768"))
INTER = int(os.environ.get("MAMBA_INTER", "1536"))
MLP_INTER = 2048
TOL = 2e-2
CLK_NS = 10


def bf16_bits(t: torch.Tensor) -> list[int]:
    return [int(v) for v in t.reshape(-1).view(torch.uint16).tolist()]


def bf16_to_f32(bits: int) -> float:
    return struct.unpack("f", struct.pack("I", (bits & 0xFFFF) << 16))[0]


async def reset_dut(dut) -> None:
    dut.rst_n.value = 0
    dut.load_en.value = 0
    dut.load_sel.value = 0
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


async def load(dut, sel: int, o: int, i: int, w: int) -> None:
    dut.load_en.value = 1
    dut.load_sel.value = sel
    dut.load_out_idx.value = o
    dut.load_in_idx.value = i
    dut.load_wdata.value = w
    await RisingEdge(dut.clk)
    dut.load_en.value = 0


@cocotb.test()
async def test_granite_layer_inloop(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_dut(dut)

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.eval()
    layer = model.model.layers[0]
    assert type(layer).__name__ == "GraniteMoeHybridDecoderLayer"
    cap = {}
    layer.register_forward_pre_hook(
        lambda m, a, k: cap.update(x=k.get("hidden_states", a[0] if a else None).detach().clone()),
        with_kwargs=True)
    layer.register_forward_hook(lambda m, a, k, o: cap.update(y=o[0].detach().clone() if isinstance(o, tuple) else o.detach().clone()), with_kwargs=True)
    ids = torch.arange(1, 10).unsqueeze(0)
    with torch.no_grad():
        out = model(ids[:, :1], use_cache=False)     # single-token baseline
    logits_base = out.logits.detach().clone()
    x = cap["x"][0, 0]                      # token 0 input, (768,) bf16
    y_ref = cap["y"][0, 0]                  # token 0 output, (768,) bf16
    assert x.dtype == torch.bfloat16 and x.numel() == HIDDEN

    m = layer.mamba
    W_in = m.in_proj.weight.data
    Wc = m.conv1d.weight.data.squeeze(1).contiguous()
    cb = m.conv1d.bias.data
    nw = m.norm.weight.data
    W_out = m.out_proj.weight.data
    a_log, D, dtb = m.A_log.data, m.D.data, m.dt_bias.data
    n1w = layer.input_layernorm.weight.data
    n2w = layer.post_attention_layernorm.weight.data
    mlp_mod = next(m for m in layer.modules() if type(m).__name__ == "GraniteMoeHybridMLP")
    mlp_in = mlp_mod.input_linear.weight.data
    mlp_out = mlp_mod.output_linear.weight.data

    dut._log.info("Loading layer-0 weights ...")
    for o in range(W_in.shape[0]):
        bits = bf16_bits(W_in[o])
        for i in range(HIDDEN):
            await load(dut, 0, o, i, bits[i])
    for o in range(W_out.shape[0]):
        bits = bf16_bits(W_out[o])
        for i in range(INTER):
            await load(dut, 1, o, i, bits[i])
    for ch in range(Wc.shape[0]):
        for k in range(4):
            await load(dut, 2, ch, k, bf16_bits(Wc[ch, 3 - k])[0])
    for ch in range(cb.shape[0]):
        await load(dut, 3, ch, 0, bf16_bits(cb[ch])[0])
    for i in range(INTER):
        await load(dut, 4, i, 0, bf16_bits(nw[i])[0])
    for i in range(a_log.numel()):
        await load(dut, 5, i, 0, bf16_bits(a_log[i])[0])
        await load(dut, 6, i, 0, bf16_bits(D[i])[0])
        await load(dut, 7, i, 0, bf16_bits(dtb[i])[0])
    for i in range(HIDDEN):
        await load(dut, 8, i, 0, bf16_bits(n1w[i])[0])
        await load(dut, 9, i, 0, bf16_bits(n2w[i])[0])
    for o in range(mlp_in.shape[0]):
        bits = bf16_bits(mlp_in[o])
        for i in range(HIDDEN):
            await load(dut, 10, o, i, bits[i])
    for o in range(mlp_out.shape[0]):
        bits = bf16_bits(mlp_out[o])
        for i in range(MLP_INTER):
            await load(dut, 11, o, i, bits[i])
    dut._log.info("Weights loaded; streaming one token ...")

    xb = bf16_bits(x)
    for i in range(HIDDEN):
        dut.s_axis_tvalid.value = 1
        dut.s_axis_tdata.value = xb[i]
        dut.s_axis_tlast.value = 1 if i == HIDDEN - 1 else 0
        await RisingEdge(dut.clk)
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tlast.value = 0

    got: list[int] = []
    dut.m_axis_tready.value = 1
    while len(got) < HIDDEN:
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.m_axis_tvalid.value) == 1:
            got.append(int(dut.m_axis_tdata.value) & 0xFFFF)
    dut.m_axis_tready.value = 0

    mism = 0
    exact = 0
    max_abs = 0.0
    max_rel = 0.0
    for i in range(HIDDEN):
        gv = bf16_to_f32(got[i])
        ev = float(y_ref[i])
        if got[i] == int(y_ref[i].view(torch.uint16).item()):
            exact += 1
        ae = abs(gv - ev)
        re_ = ae / max(abs(ev), 1e-6)
        max_abs = max(max_abs, ae)
        max_rel = max(max_rel, re_)
        if not (gv == gv and ev == ev) or (ae > TOL and re_ > TOL):
            mism += 1
            if mism <= 5:
                dut._log.error(f"elem {i}: got {gv!r} exp {ev!r} abs {ae:.3e} rel {re_:.3e}")
    dut._log.info(f"Bit-exact: {exact}/{HIDDEN}")
    dut._log.info(f"Max abs error: {max_abs:.6f}")
    dut._log.info(f"Max rel error: {max_rel:.6f}")
    assert mism == 0, f"FAILED: {mism} mismatches"
    dut._log.info("Layer output matches the model")

    # ---- soft layer 0, run the 31 remaining layers + LM head in software ----
    y_dut = torch.tensor([bf16_to_f32(v) for v in got], dtype=torch.bfloat16).reshape(1, 1, HIDDEN)
    y_ref = y_ref.reshape(1, 1, HIDDEN)

    def run_substituted(sub):
        def hook(m, a, o):
            if isinstance(o, tuple):
                return (sub,) + tuple(o[1:])
            return sub
        h = layer.register_forward_hook(hook)
        with torch.no_grad():
            o = model(ids[:, :1], use_cache=False)
        h.remove()
        return o.logits.detach().clone()

    lg_ref = run_substituted(y_ref)   # sanity: substituting the model's own
    lg_dut = run_substituted(y_dut)   # output, then the RTL output
    d_ref = (lg_ref.float() - logits_base.float()).abs()
    d_dut = (lg_dut.float() - logits_base.float()).abs()
    dut._log.info(f"substitute model output : max logits diff {float(d_ref.max()):.3e} "
                  f"(must be 0), argmax {int(lg_ref.argmax())} vs {int(logits_base.argmax())}")
    dut._log.info(f"substitute RTL output   : max logits diff {float(d_dut.max()):.3e}, "
                  f"argmax {int(lg_dut.argmax())} vs {int(logits_base.argmax())}")
    top_base = torch.topk(logits_base.float(), 5).indices.tolist()
    top_dut = torch.topk(lg_dut.float(), 5).indices.tolist()
    dut._log.info(f"top-5 base {top_base} | top-5 RTL {top_dut}")
    assert float(d_ref.max()) == 0.0, "substitution sanity failed"
    dut._log.info("End-to-end hybrid PASSED")
