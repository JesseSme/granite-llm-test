"""In-loop cocotb testbench for rmsnorm_unit.

Runs a real Granite 4.0-H-350M forward pass, captures the inputs, weights and
outputs of the non-gated RMSNorm layers (input_layernorm,
post_attention_layernorm, final norm), feeds the captured activations through
the DUT and checks two things:

  1. DUT output == bit-exact bfloat16 emulation of the RTL algorithm applied to
     the real captured inputs/weights (PRIMARY assertion).
  2. Report the deviation of the DUT against the actual PyTorch layer output,
     which computes the variance in float32 internally (the RTL accumulates in
     bfloat16), so a few percent of relative error is expected and is only
     logged, not asserted.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge
import torch

WIDTH = 768
EPS = 1e-5
PROMPT = "The quick brown fox jumps over the lazy dog"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "granite-4.0-h-350m")


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.data_in.value = 0
    dut.weight_valid.value = 0
    dut.weight_in.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def load_weight(dut, weight_u16):
    for w in weight_u16:
        dut.weight_valid.value = 1
        dut.weight_in.value = w
        await RisingEdge(dut.clk)
    dut.weight_valid.value = 0
    dut.weight_in.value = 0
    await ClockCycles(dut.clk, 10)


async def feed_input(dut, input_u16):
    for x in input_u16:
        dut.valid_in.value = 1
        dut.data_in.value = x
        await RisingEdge(dut.clk)
    dut.valid_in.value = 0
    dut.data_in.value = 0


async def collect_output(dut, expected_count, max_cycles=20000):
    collected = []
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if int(dut.valid_out.value) == 1:
            collected.append(int(dut.data_out.value))
            if len(collected) >= expected_count:
                break
    return collected


def rtl_reference(row_bf16, weight_bf16):
    """Bit-exact bfloat16 emulation of the RTL RMSNorm algorithm."""
    x = row_bf16.bfloat16()
    w = weight_bf16.bfloat16()
    sum_sq = torch.tensor(0.0, dtype=torch.bfloat16)
    for i in range(x.shape[0]):
        sum_sq = sum_sq + x[i] * x[i]
    mean = sum_sq / torch.tensor(float(x.shape[0]), dtype=torch.bfloat16)
    rms = torch.sqrt(mean + torch.tensor(EPS, dtype=torch.bfloat16))
    return (x / rms) * w


def u16_to_bf16(values):
    return torch.tensor(values, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)


@cocotb.test()
async def test_rmsnorm_inloop(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    dut._log.info("Loading Granite model ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.eval()

    targets = [
        (f"layers.{i}.input_layernorm", model.model.layers[i].input_layernorm)
        for i in (0, 1, 10, 31)
    ]
    targets.append(("layers.0.post_attention_layernorm",
                    model.model.layers[0].post_attention_layernorm))
    targets.append(("final_norm", model.model.norm))

    captures = []

    def make_hook(name):
        def hook(module, inputs, output):
            captures.append((
                name,
                inputs[0].detach().clone(),
                output.detach().clone(),
                module.weight.detach().clone(),
            ))
        return hook

    handles = [mod.register_forward_hook(make_hook(name)) for name, mod in targets]
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        model(input_ids)
    for h in handles:
        h.remove()

    num_rows = input_ids.shape[1]
    dut._log.info(f"Captured {len(captures)} RMSNorm invocations x {num_rows} rows")

    total = 0
    emu_mismatches = 0
    max_model_abs = 0.0
    max_model_rel = 0.0

    for name, x, y, w in captures:
        x_rows = x.reshape(-1, WIDTH).contiguous()
        y_rows = y.reshape(-1, WIDTH).contiguous()
        w_row = w.reshape(WIDTH).contiguous()
        w_u16 = w_row.view(torch.uint16).tolist()

        for r in range(x_rows.shape[0]):
            row = x_rows[r]
            expected = y_rows[r]

            await reset_dut(dut)
            await load_weight(dut, w_u16)
            await feed_input(dut, row.view(torch.uint16).tolist())
            got = await collect_output(dut, WIDTH)

            if len(got) < WIDTH:
                dut._log.error(f"{name} row {r}: only {len(got)}/{WIDTH} outputs")
                emu_mismatches += WIDTH - len(got)
                total += WIDTH
                continue

            got_t = u16_to_bf16(got)

            emu = rtl_reference(row, w_row)
            if not torch.equal(got_t, emu):
                emu_mismatches += int((got_t != emu).sum().item())
                idx = int((got_t != emu).nonzero()[0].item())
                dut._log.warning(
                    f"{name} row {r}: emulation mismatch at {idx}: "
                    f"got=0x{got[idx]:04X} expected=0x{emu[idx].view(torch.uint16).item():04X}"
                )

            gf = got_t.float()
            rf = expected.float()
            abs_err = (gf - rf).abs().max().item()
            rel_err = ((gf - rf).abs() / rf.abs().clamp_min(1e-3)).max().item()
            max_model_abs = max(max_model_abs, abs_err)
            max_model_rel = max(max_model_rel, rel_err)
            total += WIDTH

    dut._log.info(f"In-loop results: {total - emu_mismatches}/{total} bit-exact vs RTL emulation")
    dut._log.info(f"Max abs error vs PyTorch layer: {max_model_abs:.6f}")
    dut._log.info(f"Max rel error vs PyTorch layer: {max_model_rel:.4f}")

    if max_model_rel > 0.15:
        dut._log.warning("PyTorch deviation exceeds 15% (bf16 accumulation vs fp32 variance)")

    if emu_mismatches > 0:
        dut._log.error(f"FAILED: {emu_mismatches} mismatches vs RTL emulation")
        assert False, f"In-loop test failed with {emu_mismatches} mismatches"
    dut._log.info("PASSED: RTL matches bit-exact emulation on real model activations")
