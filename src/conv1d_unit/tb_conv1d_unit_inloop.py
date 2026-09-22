"""In-loop cocotb testbench for conv1d_unit.

Runs a real Granite 4.0-H-350M forward pass and captures the first Mamba
layer's projected states (via a hook on `in_proj`). The raw causal depthwise
conv is then recomputed from the captured states and the checkpoint's conv1d
weights exactly as the model's CPU fallback does
(`causal_conv1d_fn` -> F.conv1d(padding=3, groups=conv_dim)[...][:, :, :S]`,
without the SiLU activation, which is a separate unit).

Note: the model's Conv1d module is NOT invoked on CPU (the functional
fallback is used), so hooking `mamba.conv1d` would never fire. Only the first
1536 of the 1792 conv_dim channels are exercised — that slice is the state
branch implemented by this unit.
"""

import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import struct
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

CHANNELS = 1536
CONV_DIM = 1792
INTERMEDIATE = 1536
KERNEL = 4
PROMPT = "The quick brown fox jumps over the lazy dog"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "granite-4.0-h-350m")


def bfloat16_to_float(val):
    val = val & 0xFFFF
    float32_val = val << 16
    return struct.unpack("f", struct.pack("I", float32_val))[0]


def float_to_bfloat16(val):
    f32 = struct.pack("f", val)
    i32 = struct.unpack("I", f32)[0]
    rounding_bias = (1 << 15) + ((i32 >> 16) & 1)
    i32 += rounding_bias
    bf16 = i32 >> 16
    return bf16 & 0xFFFF


@cocotb.test()
async def test_conv1d_inloop(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    dut.valid_i.value = 0
    dut.data_i.value = 0
    dut.load_en.value = 0
    dut.load_ch.value = 0
    dut.load_tap.value = 0
    dut.load_wdata.value = 0
    dut.load_is_bias.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    dut._log.info("Loading Granite model ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.eval()

    mamba = model.model.layers[0].mamba
    captured = {}

    def hook_fn(module, inputs, output):
        captured["proj"] = output.detach().clone()

    handle = mamba.in_proj.register_forward_hook(hook_fn)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        model(input_ids)
    handle.remove()

    projected = captured["proj"]  # (1, S, 3376)
    seq_len = projected.shape[1]
    hidden_states_B_C = projected[..., INTERMEDIATE:INTERMEDIATE + CONV_DIM]
    x = hidden_states_B_C.transpose(1, 2)  # (1, CONV_DIM, S)

    conv_weight = mamba.conv1d.weight.data  # (CONV_DIM, 1, KERNEL)
    conv_bias = mamba.conv1d.bias.data  # (CONV_DIM,)

    with torch.no_grad():
        conv_ref = F.conv1d(
            x.to(conv_weight.dtype),
            weight=conv_weight,
            bias=conv_bias,
            padding=KERNEL - 1,
            groups=CONV_DIM,
        )[:, :, :seq_len]  # raw causal conv, pre-SiLU

    dut._log.info(f"seq_len={seq_len}, conv_ref shape={tuple(conv_ref.shape)}")

    dut._log.info("Loading conv1d weights into DUT ...")
    for c in range(CHANNELS):
        for k in range(KERNEL):
            w_val = float_to_bfloat16(conv_weight[c, 0, KERNEL - 1 - k].item())
            dut.load_en.value = 1
            dut.load_ch.value = c
            dut.load_tap.value = k
            dut.load_wdata.value = w_val
            dut.load_is_bias.value = 0
            await RisingEdge(dut.clk)
    dut.load_en.value = 0
    await RisingEdge(dut.clk)

    dut._log.info("Loading conv1d biases into DUT ...")
    for c in range(CHANNELS):
        b_val = float_to_bfloat16(conv_bias[c].item())
        dut.load_en.value = 1
        dut.load_ch.value = c
        dut.load_wdata.value = b_val
        dut.load_is_bias.value = 1
        await RisingEdge(dut.clk)
    dut.load_en.value = 0
    await RisingEdge(dut.clk)

    dut._log.info(f"Running in-loop test ({seq_len} timesteps x {CHANNELS} channels) ...")
    mismatches = 0
    max_abs_error = 0.0

    for t in range(seq_len):
        for c in range(CHANNELS):
            inp_hex = float_to_bfloat16(x[0, c, t].item())
            exp_hex = float_to_bfloat16(conv_ref[0, c, t].item())

            while int(dut.ready_o.value) == 0:
                await RisingEdge(dut.clk)

            dut.data_i.value = inp_hex
            dut.valid_i.value = 1
            await RisingEdge(dut.clk)
            dut.valid_i.value = 0

            while int(dut.valid_o.value) == 0:
                await RisingEdge(dut.clk)

            actual = int(dut.data_o.value) & 0xFFFF
            actual_float = bfloat16_to_float(actual)
            expected_float = bfloat16_to_float(exp_hex)
            abs_err = abs(actual_float - expected_float)

            if abs_err > 0.1:
                mismatches += 1
                if mismatches <= 10:
                    dut._log.warning(
                        f"t={t} c={c}: expected=0x{exp_hex:04X} ({expected_float:.6f}), "
                        f"actual=0x{actual:04X} ({actual_float:.6f}), err={abs_err:.6f}"
                    )
            max_abs_error = max(max_abs_error, abs_err)

    total = seq_len * CHANNELS
    dut._log.info(f"In-loop results: {total - mismatches}/{total} passed")
    dut._log.info(f"Max absolute error: {max_abs_error:.6f}")

    if mismatches > 0:
        dut._log.error(f"FAILED: {mismatches} mismatches")
        assert False, f"In-loop test failed with {mismatches} mismatches"
    dut._log.info("PASSED: All in-loop conv1d outputs matched")
