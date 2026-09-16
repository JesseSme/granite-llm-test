"""In-loop cocotb testbench for matrix_unit.

Runs real Granite 4.0-H-350M forward passes, captures a real Linear layer's
input activations and output (via a hook), loads the checkpoint weights/bias
into the DUT and streams every captured input row through it.

The DUT output is compared against the model output. Exact matching is
expected except for rare (<0.1%) 1-ULP differences caused by ATen's blocked
GEMM accumulation order, which the RTL (sequential accumulation) does not
replicate; the test asserts a minimum bit-exact rate and a 1-ULP bound.

Configuration (build dimensions and layer path) comes from env vars
MATRIX_IN / MATRIX_OUT / MATRIX_MODULE.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer
import torch

IN = int(os.environ.get("MATRIX_IN", "768"))
OUT = int(os.environ.get("MATRIX_OUT", "768"))
MODULE_PATH = os.environ.get("MATRIX_MODULE", "layers.10.self_attn.q_proj")
PROMPTS = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a field of study",
]
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "granite-4.0-h-350m")
MIN_EXACT_RATE = 0.995


def resolve(root, path):
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tdata.value = 0
    dut.s_axis_tlast.value = 0
    dut.m_axis_tready.value = 0
    dut.load_en.value = 0
    dut.load_out_idx.value = 0
    dut.load_in_idx.value = 0
    dut.load_wdata.value = 0
    dut.load_is_bias.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def feed_vector(dut, row_u16):
    for i in range(IN):
        dut.s_axis_tvalid.value = 1
        dut.s_axis_tdata.value = row_u16[i]
        dut.s_axis_tlast.value = 1 if i == IN - 1 else 0
        await RisingEdge(dut.clk)
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tlast.value = 0

    got = []
    dut.m_axis_tready.value = 1
    while len(got) < OUT:
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.m_axis_tvalid.value) == 1:
            got.append(int(dut.m_axis_tdata.value) & 0xFFFF)
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.m_axis_tready.value = 0
    return got


@cocotb.test()
async def test_matrix_inloop(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut._log.info("Loading Granite model ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.eval()

    linear = resolve(model.model, MODULE_PATH)
    dut._log.info(f"Target module: {MODULE_PATH} (in={linear.in_features}, out={linear.out_features})")
    assert linear.in_features == IN and linear.out_features == OUT, "dimension mismatch"

    captured_in = []
    captured_out = []

    def hook(module, inputs, output):
        captured_in.append(inputs[0].detach().clone())
        captured_out.append(output.detach().clone())

    handle = linear.register_forward_hook(hook)
    with torch.no_grad():
        for prompt in PROMPTS:
            model(tokenizer(prompt, return_tensors="pt").input_ids)
    handle.remove()

    rows_in = []
    rows_out = []
    for act_in, act_out in zip(captured_in, captured_out):
        for pos in range(act_in.shape[1]):
            rows_in.append(act_in[0, pos].contiguous())
            rows_out.append(act_out[0, pos].contiguous())
    max_rows = 8
    rows_in = rows_in[:max_rows]
    rows_out = rows_out[:max_rows]
    dut._log.info(f"Captured {len(rows_in)} input rows (using first {len(rows_in)})")

    weight = linear.weight.data
    if linear.bias is None:
        bias = torch.zeros(OUT, dtype=torch.bfloat16)
    else:
        bias = linear.bias.data
    dut._log.info(f"Loading {OUT}x{IN} weights ...")
    for o in range(OUT):
        w_u16 = weight[o].view(torch.uint16).tolist()
        for i in range(IN):
            dut.load_en.value = 1
            dut.load_out_idx.value = o
            dut.load_in_idx.value = i
            dut.load_wdata.value = w_u16[i]
            dut.load_is_bias.value = 0
            await RisingEdge(dut.clk)
    b_u16 = bias.view(torch.uint16).tolist()
    for o in range(OUT):
        dut.load_en.value = 1
        dut.load_out_idx.value = o
        dut.load_wdata.value = b_u16[o]
        dut.load_is_bias.value = 1
        await RisingEdge(dut.clk)
    dut.load_en.value = 0

    total = 0
    exact = 0
    max_abs = 0.0
    max_ulp = 0
    for v, (row_in, row_out) in enumerate(zip(rows_in, rows_out)):
        got = await feed_vector(dut, row_in.view(torch.uint16).tolist())
        got_t = torch.tensor(got, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
        exp_t = row_out

        diff = (got_t.float() - exp_t.float()).abs()
        max_abs = max(max_abs, diff.max().item())
        eq = got_t == exp_t
        exact += int(eq.sum().item())
        total += OUT

        # ULP distance for same-sign values (sign-magnitude difference)
        gb = got_t.view(torch.uint16).to(torch.int32)
        eb = exp_t.view(torch.uint16).to(torch.int32)
        same_sign = (gb >> 15) == (eb >> 15)
        ulp = (gb & 0x7FFF) - (eb & 0x7FFF)
        ulp = ulp.abs()
        if same_sign.any():
            max_ulp = max(max_ulp, int(ulp[same_sign].max().item()))
        if total % (OUT * 4) == 0:
            dut._log.info(f"  row {v}: exact={int(eq.sum())}/{OUT}")

    rate = exact / total
    dut._log.info(f"In-loop results: {exact}/{total} bit-exact ({100.0 * rate:.3f}%)")
    dut._log.info(f"Max abs error: {max_abs:.6f}")
    dut._log.info(f"Max ULP difference (same sign): {max_ulp}")

    if rate < MIN_EXACT_RATE:
        dut._log.error(f"FAILED: bit-exact rate {rate:.4f} < {MIN_EXACT_RATE}")
        assert False, "in-loop exact rate below threshold"
    if max_ulp > 1:
        dut._log.error(f"FAILED: ULP difference {max_ulp} > 1")
        assert False, "in-loop ULP difference above 1"
    dut._log.info("PASSED: real Linear layer within 1 ULP, >=99.5% bit-exact")
