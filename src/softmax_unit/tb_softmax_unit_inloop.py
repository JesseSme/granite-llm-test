"""In-loop cocotb testbench for softmax_unit.

Runs a real Granite 4.0-H-350M forward pass with the eager attention
implementation and captures the actual attention softmax operands by
monkeypatching torch.nn.functional.softmax during the forward pass. The
captured (B, heads, S, S) score matrices are the real scaled QK^T logits
(includes the causal mask), and each row is fed through the DUT. Outputs are
compared against the model's own float32 softmax result.
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import torch
import torch.nn.functional as F

PROMPT = "The quick brown fox jumps over the lazy dog"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "granite-4.0-h-350m")


def float_to_hex(val):
    return struct.unpack("I", struct.pack("f", val))[0]


def hex_to_float(h):
    return struct.unpack("f", struct.pack("I", h & 0xFFFFFFFF))[0]


@cocotb.test()
async def test_softmax_inloop(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.data_in.value = 0
    dut.last_in.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    dut._log.info("Loading Granite model (eager attention) ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.eval()

    captured = []
    original_softmax = F.softmax

    def patched_softmax(input, dim=None, *args, **kwargs):
        out = original_softmax(input, dim=dim, *args, **kwargs)
        if input.dim() == 4:
            captured.append((input.detach().float().clone(), out.detach().float().clone()))
        return out

    torch.nn.functional.softmax = patched_softmax
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        torch.nn.functional.softmax = original_softmax

    if not captured:
        dut._log.error("No attention softmax operands captured")
        assert False, "capture failed"

    row_len = captured[0][0].shape[-1]
    num_rows = sum(s.reshape(-1, row_len).shape[0] for s, _ in captured)
    dut._log.info(f"Captured {len(captured)} attention score matrices, {num_rows} rows of {row_len}")

    total = 0
    mismatches = 0
    max_abs_error = 0.0
    max_sum_error = 0.0

    for scores, ref in captured:
        s = scores.shape[-1]
        s_rows = scores.reshape(-1, s)
        r_rows = ref.reshape(-1, s)

        for r in range(s_rows.shape[0]):
            row = s_rows[r]
            expected = r_rows[r]

            for i in range(s):
                dut.data_in.value = float_to_hex(row[i].item())
                dut.valid_in.value = 1
                dut.last_in.value = 1 if i == s - 1 else 0
                await RisingEdge(dut.clk)
            dut.valid_in.value = 0
            dut.last_in.value = 0

            row_outputs = []
            for _ in range(s * 40):
                await RisingEdge(dut.clk)
                if int(dut.valid_out.value) == 1:
                    row_outputs.append(int(dut.data_out.value))
                if int(dut.done.value) == 1:
                    break
            await RisingEdge(dut.clk)

            if len(row_outputs) < s:
                dut._log.error(f"Row {total // s}: only {len(row_outputs)}/{s} outputs")
                mismatches += s - len(row_outputs)
                total += s
                continue

            row_sum = 0.0
            for i in range(s):
                actual = hex_to_float(row_outputs[i])
                exp = expected[i].item()
                row_sum += actual
                abs_err = abs(actual - exp)
                max_abs_error = max(max_abs_error, abs_err)
                if abs_err > 0.05:
                    mismatches += 1
                    if mismatches <= 10:
                        dut._log.warning(
                            f"Row {total // s} elem {i}: expected={exp:.6f}, "
                            f"actual={actual:.6f}, err={abs_err:.6f}"
                        )
            max_sum_error = max(max_sum_error, abs(row_sum - 1.0))
            total += s

    dut._log.info(f"In-loop results: {total - mismatches}/{total} passed")
    dut._log.info(f"Max absolute error: {max_abs_error:.6f}")
    dut._log.info(f"Max row-sum error: {max_sum_error:.6f}")

    if mismatches > 0:
        dut._log.error(f"FAILED: {mismatches} mismatches")
        assert False, f"In-loop test failed with {mismatches} mismatches"
    dut._log.info("PASSED: All in-loop softmax rows matched real attention scores")
