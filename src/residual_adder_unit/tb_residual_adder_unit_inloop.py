"""In-loop cocotb testbench for residual_adder_unit.

Runs real Granite 4.0-H-350M forward passes and captures the operands of both
residual adds per decoder layer (mixer residual and MLP residual):

  residual (layer input / post-norm hidden states)
  branch   (mixer output / shared_mlp output)
  expected (input of post_attention_layernorm / decoder-layer output)

Each captured element pair is streamed through the DUT. The DUT output is
checked bit-exactly against an emulation of the RTL algorithm (binary32
multiply/add with the fp32 constant, one bf16 rounding per operation) and
against the model's actual tensor, whose scalar arithmetic this datapath
reproduces.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge
import torch

DIM = 768
LAYER_INDICES = (0, 10)  # one Mamba layer, one attention layer
PROMPTS = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a fascinating field of study",
]
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "granite-4.0-h-350m")
MULT_FP32 = torch.tensor(0.246, dtype=torch.float32)


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.data_in.value = 0
    dut.residual_in.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


def capture_samples(model, tokenizer, prompt):
    """Returns list of (tag, residual, branch, model_out) row tensors (DIM,)."""
    samples = []

    for layer_idx in LAYER_INDICES:
        layer = model.model.layers[layer_idx]
        mixer = layer.mamba if layer.mamba is not None else layer.self_attn
        store = {}

        def pre_layer(mod, inp, st=store):
            st["layer_in"] = inp[0].detach().clone()

        def mixer_out(mod, inp, out, st=store):
            st["mixer_out"] = (out[0] if isinstance(out, tuple) else out).detach().clone()

        def norm_in(mod, inp, st=store):
            st["norm_in"] = inp[0].detach().clone()

        def mlp_out(mod, inp, out, st=store):
            st["mlp_out"] = out.detach().clone()

        def layer_out(mod, inp, out, st=store):
            st["layer_out"] = (out[0] if isinstance(out, tuple) else out).detach().clone()

        handles = [
            layer.register_forward_pre_hook(pre_layer),
            mixer.register_forward_hook(mixer_out),
            layer.post_attention_layernorm.register_forward_pre_hook(norm_in),
            layer.shared_mlp.register_forward_hook(mlp_out),
            layer.register_forward_hook(layer_out),
        ]
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            model(ids)
        for h in handles:
            h.remove()

        seq = store["layer_in"].shape[1]
        for pos in range(seq):
            samples.append((
                f"L{layer_idx}.mixer",
                store["layer_in"][0, pos].contiguous(),
                store["mixer_out"][0, pos].contiguous(),
                store["norm_in"][0, pos].contiguous(),
            ))
            samples.append((
                f"L{layer_idx}.mlp",
                store["norm_in"][0, pos].contiguous(),
                store["mlp_out"][0, pos].contiguous(),
                store["layer_out"][0, pos].contiguous(),
            ))
    return samples


@cocotb.test()
async def test_residual_adder_inloop(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut._log.info("Loading Granite model ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.eval()

    samples = []
    for prompt in PROMPTS:
        samples.extend(capture_samples(model, tokenizer, prompt))
    dut._log.info(f"Captured {len(samples)} residual-add vectors")

    # Stream all elements through the DUT (branch -> data_in, residual -> residual_in)
    schedule = []
    for _, residual, branch, _ in samples:
        r_u16 = residual.view(torch.uint16).tolist()
        b_u16 = branch.view(torch.uint16).tolist()
        for i in range(DIM):
            schedule.append((b_u16[i], r_u16[i]))

    total = len(schedule)
    got = []
    for k in range(total + 8):
        if k < total:
            data, res = schedule[k]
            dut.valid_in.value = 1
            dut.data_in.value = data
            dut.residual_in.value = res
        else:
            dut.valid_in.value = 0
        await RisingEdge(dut.clk)
        if int(dut.valid_out.value) == 1:
            got.append(int(dut.data_out.value) & 0xFFFF)

    if len(got) < total:
        dut._log.error(f"Only collected {len(got)}/{total} outputs")
        assert False, "output collection incomplete"

    emu_mismatches = 0
    max_model_abs = 0.0
    max_model_rel = 0.0
    exact_model = 0

    for v, (tag, residual, branch, model_out) in enumerate(samples):
        chunk = got[v * DIM:(v + 1) * DIM]
        got_t = torch.tensor(chunk, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)

        # RTL algorithm: fp32 multiply by fp32(0.246), bf16 round, fp32 add, bf16 round
        prod = (branch.float() * MULT_FP32).bfloat16()
        emu = (residual.float() + prod.float()).bfloat16()
        if not torch.equal(got_t, emu):
            emu_mismatches += int((got_t != emu).sum().item())
            idx = int((got_t != emu).nonzero()[0].item())
            dut._log.warning(
                f"{tag} vec {v}: RTL-emulation mismatch at {idx}: "
                f"got 0x{chunk[idx]:04x} expected 0x{emu[idx].view(torch.uint16).item():04x}"
            )

        gf = got_t.float()
        mf = model_out.float()
        abs_err = (gf - mf).abs().max().item()
        rel_err = ((gf - mf).abs() / mf.abs().clamp_min(1e-3)).max().item()
        max_model_abs = max(max_model_abs, abs_err)
        max_model_rel = max(max_model_rel, rel_err)
        exact_model += int((got_t == model_out).sum().item())

    n = total
    dut._log.info(f"In-loop results: {n - emu_mismatches}/{n} bit-exact vs RTL emulation")
    dut._log.info(f"Bit-exact vs model output: {exact_model}/{n} ({100.0 * exact_model / n:.1f}%)")
    dut._log.info(f"Max abs error vs model: {max_model_abs:.6f}")
    dut._log.info(f"Max rel error vs model: {max_model_rel:.6f}")

    if exact_model < n:
        dut._log.error(f"{n - exact_model} elements differ from the model output")
        assert False, f"In-loop model mismatch: {n - exact_model} elements"

    if emu_mismatches > 0:
        dut._log.error(f"FAILED: {emu_mismatches} mismatches vs RTL emulation")
        assert False, f"In-loop test failed with {emu_mismatches} mismatches"
    dut._log.info("PASSED: RTL matches the model output bit-exactly on real tensors")
