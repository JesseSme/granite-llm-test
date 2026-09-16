"""In-loop cocotb testbench for embedding_lookup_unit.

Runs real Granite 4.0-H-350M tokenization/forward passes, captures the actual
scaled embedding tensor that feeds decoder layer 0 (input of its
input_layernorm), loads the corresponding table rows from the model's
`embed_tokens.weight`, and compares the DUT output bit-exactly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge
import torch

DIM = 768
PROMPTS = [
    "The quick brown fox jumps over the lazy dog",
    "The capital of France is",
    "Machine learning is a fascinating field of study",
]
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "granite-4.0-h-350m")


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.token_id.value = 0
    dut.load_en.value = 0
    dut.load_token.value = 0
    dut.load_idx.value = 0
    dut.load_wdata.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


@cocotb.test()
async def test_embedding_lookup_inloop(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut._log.info("Loading Granite model ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.eval()
    weight = model.model.embed_tokens.weight.data  # (VOCAB, DIM) bfloat16

    captured = []

    def hook(module, inputs, output):
        captured.append(inputs[0].detach().clone())

    handle = model.model.layers[0].input_layernorm.register_forward_hook(hook)
    with torch.no_grad():
        for prompt in PROMPTS:
            ids = tokenizer(prompt, return_tensors="pt").input_ids
            model(ids)
    handle.remove()

    # captured activations are (1, S, DIM) - the scaled embedding fed to layer 0
    token_ids = []
    expected_rows = []
    for prompt, act in zip(PROMPTS, captured):
        ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        for pos, tok in enumerate(ids):
            token_ids.append(tok)
            expected_rows.append(act[0, pos].contiguous())

    num_tokens = len(token_ids)
    dut._log.info(f"Captured {num_tokens} token positions from {len(PROMPTS)} prompts")

    unique = sorted(set(token_ids))
    dut._log.info(f"Loading {len(unique)} unique embedding rows ...")
    for tok in unique:
        row = weight[tok]
        row_u16 = row.view(torch.uint16).tolist()
        for i in range(DIM):
            dut.load_en.value = 1
            dut.load_token.value = tok
            dut.load_idx.value = i
            dut.load_wdata.value = row_u16[i]
            await RisingEdge(dut.clk)
    dut.load_en.value = 0
    await ClockCycles(dut.clk, 2)

    mismatches = 0
    total = 0
    max_abs_error = 0.0

    for k, tok in enumerate(token_ids):
        exp_t = expected_rows[k].contiguous()
        exp = exp_t.view(torch.uint16).tolist()

        while int(dut.busy.value) == 1:
            await RisingEdge(dut.clk)

        dut.valid_in.value = 1
        dut.token_id.value = tok
        await RisingEdge(dut.clk)
        dut.valid_in.value = 0

        got = []
        for _ in range(DIM * 4 + 16):
            await RisingEdge(dut.clk)
            if int(dut.valid_out.value) == 1:
                got.append(int(dut.data_out.value) & 0xFFFF)
                if len(got) == DIM:
                    break

        if len(got) != DIM:
            dut._log.error(f"token 0x{tok:05x}: only {len(got)}/{DIM} outputs")
            mismatches += DIM - len(got)
            total += DIM
            continue

        got_t = torch.tensor(got, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
        abs_err = (got_t.float() - exp_t.float()).abs().max().item()
        max_abs_error = max(max_abs_error, abs_err)

        for i in range(DIM):
            if got[i] != exp[i]:
                mismatches += 1
                if mismatches <= 10:
                    dut._log.warning(
                        f"token 0x{tok:05x} pos {k} elem {i}: "
                        f"got 0x{got[i]:04x}, expected 0x{exp[i]:04x}"
                    )
        total += DIM

    dut._log.info(f"In-loop results: {total - mismatches}/{total} bit-exact")
    dut._log.info(f"Max absolute error: {max_abs_error:.6f}")

    if mismatches > 0:
        dut._log.error(f"FAILED: {mismatches} mismatches")
        assert False, f"In-loop test failed with {mismatches} mismatches"
    dut._log.info("PASSED: all real-token lookups bit-exact")
