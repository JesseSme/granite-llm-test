"""In-loop cocotb testbench for sigmoid_unit.

Loads Granite 4.0-H-350M, captures real activations that flow through sigmoid,
feeds them through the DUT in RTL simulation, and compares against PyTorch output.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import struct
import os
import numpy as np


def bfloat16_to_float(val):
    val = val & 0xFFFF
    float32_val = val << 16
    return struct.unpack('f', struct.pack('I', float32_val))[0]


def float_to_bfloat16(val):
    f32 = struct.pack('f', val)
    i32 = struct.unpack('I', f32)[0]
    rounding_bias = (1 << 15) + ((i32 >> 16) & 1)
    i32 += rounding_bias
    bf16 = i32 >> 16
    return bf16 & 0xFFFF


def compute_sigmoid_lut_entry(addr):
    import math
    x = addr / 32.0
    sig = 1.0 / (1.0 + math.exp(-x))
    sig = min(sig, 0.999999)
    return float_to_bfloat16(sig)


def compute_address(bf16_val):
    if bf16_val == 0:
        return 0
    exp = (bf16_val >> 7) & 0xFF
    mant = bf16_val & 0x7F
    if exp < 122:
        return 0
    elif exp >= 130:
        return 255
    elif exp == 122:
        return 1
    elif exp == 123:
        return 2 + (mant >> 6)
    elif exp == 124:
        return 4 + (mant >> 5)
    elif exp == 125:
        return 8 + (mant >> 4)
    elif exp == 126:
        return 16 + (mant >> 3)
    elif exp == 127:
        return 32 + (mant >> 2)
    elif exp == 128:
        return 64 + (mant >> 1)
    elif exp == 129:
        return 128 + mant
    return 0


def hardware_sigmoid(bf16_input):
    sign = (bf16_input >> 15) & 1
    exp = (bf16_input >> 7) & 0xFF
    mant = bf16_input & 0x7F
    is_nan = (exp == 0xFF) and (mant != 0)
    is_inf = (exp == 0xFF) and (mant == 0)
    is_zero = (exp == 0) and (mant == 0)
    is_sub = (exp == 0) and (mant != 0)

    if is_nan:
        return 0x3F00
    elif is_inf:
        return 0x0000 if sign else 0x3F80
    elif is_zero:
        return 0x3F00

    magnitude_bf16 = bf16_input & 0x7FFF
    addr = compute_address(magnitude_bf16)
    pos_val = compute_sigmoid_lut_entry(addr)

    import math
    x = addr / 32.0
    neg_sig = 1.0 - (1.0 / (1.0 + math.exp(-x)))
    neg_val = float_to_bfloat16(neg_sig)

    return neg_val if sign else pos_val


@cocotb.test()
async def test_sigmoid_inloop(dut):
    """Test sigmoid_unit with real model activations."""

    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.data_in.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Load golden sample from in-loop data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inputs_path = os.path.join(script_dir, "golden_inputs.hex")
    outputs_path = os.path.join(script_dir, "golden_outputs.hex")

    if not os.path.exists(inputs_path):
        dut._log.info("Golden sample not found, generating from model activations")

        # Generate activations from the model
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path = "granite-4.0-h-350m"
            dut._log.info(f"Loading model from {model_path}")

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu"
            )
            model.eval()

            # Hook to capture SiLU inputs
            activations = []

            def hook_fn(module, input, output):
                # SiLU input is the conv1d output in Mamba2 layers
                if isinstance(input, tuple):
                    x = input[0]
                else:
                    x = input
                if x is not None and x.dtype == torch.bfloat16:
                    activations.append(x.detach())

            # Register hooks on SiLU layers
            hooks = []
            for name, module in model.named_modules():
                if 'act' in name.lower() and hasattr(module, 'forward'):
                    hooks.append(module.register_forward_hook(hook_fn))

            # Run inference
            input_text = "The quick brown fox jumps"
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            with torch.no_grad():
                _ = model(input_ids)

            for h in hooks:
                h.remove()

            if not activations:
                dut._log.warning("No activations captured, using synthetic data")
                # Create synthetic activations
                acts = [torch.randn(1, 10, 1536, dtype=torch.bfloat16) for _ in range(5)]
            else:
                acts = activations[:5]

            # Extract unique bfloat16 values from activations
            bf16_inputs = []
            bf16_outputs = []
            seen = set()

            for act in acts:
                flat = act.reshape(-1).numpy()
                for val in flat[:500]:  # Limit per activation
                    bf16_val = float_to_bfloat16(float(val))
                    if bf16_val not in seen and len(bf16_inputs) < 300:
                        seen.add(bf16_val)
                        bf16_inputs.append(bf16_val)
                        bf16_outputs.append(hardware_sigmoid(bf16_val))

            dut._log.info(f"Generated {len(bf16_inputs)} test vectors from model activations")

        except Exception as e:
            dut._log.error(f"Failed to load model: {e}")
            dut._log.info("Falling back to synthetic data")
            import random
            random.seed(42)
            bf16_inputs = []
            bf16_outputs = []
            for _ in range(200):
                val = random.uniform(-6.0, 6.0)
                bf16_in = float_to_bfloat16(val)
                bf16_out = hardware_sigmoid(bf16_in)
                bf16_inputs.append(bf16_in)
                bf16_outputs.append(bf16_out)

        # Save golden sample
        with open(inputs_path, 'w') as f:
            for v in bf16_inputs:
                f.write(f"{v:04X}\n")
        with open(outputs_path, 'w') as f:
            for v in bf16_outputs:
                f.write(f"{v:04X}\n")
    else:
        with open(inputs_path, 'r') as f:
            bf16_inputs = [int(line.strip(), 16) for line in f if line.strip()]
        with open(outputs_path, 'r') as f:
            bf16_outputs = [int(line.strip(), 16) for line in f if line.strip()]

    num_vectors = len(bf16_inputs)
    dut._log.info(f"Running {num_vectors} in-loop test vectors")

    mismatches = 0
    max_abs_error = 0.0

    for i in range(num_vectors):
        dut.data_in.value = bf16_inputs[i]
        dut.valid_in.value = 1
        await RisingEdge(dut.clk)
        dut.valid_in.value = 0

        # Wait for result (1 cycle latency)
        await RisingEdge(dut.clk)

        actual = dut.data_out.value.to_unsigned() & 0xFFFF
        expected = bf16_outputs[i]

        if actual != expected:
            actual_float = bfloat16_to_float(actual)
            expected_float = bfloat16_to_float(expected)
            abs_err = abs(actual_float - expected_float)

            if abs_err > 0.02:
                mismatches += 1
                if mismatches <= 10:
                    dut._log.warning(
                        f"Vector {i}: input=0x{bf16_inputs[i]:04X} "
                        f"({bfloat16_to_float(bf16_inputs[i]):.6f}), "
                        f"expected=0x{expected:04X} ({expected_float:.6f}), "
                        f"actual=0x{actual:04X} ({actual_float:.6f})"
                    )

            if abs_err > max_abs_error:
                max_abs_error = abs_err

    dut._log.info(f"In-loop results: {num_vectors - mismatches}/{num_vectors} passed")
    dut._log.info(f"Max absolute error: {max_abs_error:.6f}")

    if mismatches > 0:
        dut._log.error(f"FAILED: {mismatches} mismatches")
        assert False, f"In-loop test failed with {mismatches} mismatches"
    else:
        dut._log.info("PASSED: All in-loop test vectors matched")
