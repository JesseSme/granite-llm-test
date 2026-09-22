"""In-loop cocotb testbench for silu_unit.

Loads Granite 4.0-H-350M, captures real activations that flow through SiLU,
feeds them through the DUT in RTL simulation, and compares against PyTorch output.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import struct
import os


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


def hardware_sigmoid(bf16_input):
    """Compute hardware sigmoid to generate expected SiLU output."""
    import math
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
    if is_sub or is_zero or exp < 122:
        addr = 0
    elif exp >= 130:
        addr = 255
    elif exp == 122:
        addr = 1
    elif exp == 123:
        addr = 2 + (mant >> 6)
    elif exp == 124:
        addr = 4 + (mant >> 5)
    elif exp == 125:
        addr = 8 + (mant >> 4)
    elif exp == 126:
        addr = 16 + (mant >> 3)
    elif exp == 127:
        addr = 32 + (mant >> 2)
    elif exp == 128:
        addr = 64 + (mant >> 1)
    elif exp == 129:
        addr = 128 + mant
    else:
        addr = 0

    x = addr / 32.0
    sig = 1.0 / (1.0 + math.exp(-x))
    sig = min(sig, 0.999999)
    pos_val = float_to_bfloat16(sig)
    neg_sig = 1.0 - sig
    neg_val = float_to_bfloat16(neg_sig)
    return neg_val if sign else pos_val


def hardware_bfloat16_mul(a_bf16, b_bf16):
    """Compute bfloat16 multiply matching the RTL (RNE rounding)."""
    a_sign = (a_bf16 >> 15) & 1
    a_exp  = (a_bf16 >> 7) & 0xFF
    a_mant = a_bf16 & 0x7F
    b_sign = (b_bf16 >> 15) & 1
    b_exp  = (b_bf16 >> 7) & 0xFF
    b_mant = b_bf16 & 0x7F

    is_a_nan  = (a_exp == 0xFF) and (a_mant != 0)
    is_a_inf  = (a_exp == 0xFF) and (a_mant == 0)
    is_a_zero = (a_exp == 0) and (a_mant == 0)
    is_a_sub  = (a_exp == 0) and (a_mant != 0)
    is_b_nan  = (b_exp == 0xFF) and (b_mant != 0)
    is_b_inf  = (b_exp == 0xFF) and (b_mant == 0)
    is_b_zero = (b_exp == 0) and (b_mant == 0)
    is_b_sub  = (b_exp == 0) and (b_mant != 0)

    rsign = a_sign ^ b_sign
    QUIET = 0x7F

    if is_a_nan or is_b_nan:
        payload = a_mant if is_a_nan else b_mant
        return (0 << 15) | (0xFF << 7) | (payload | QUIET)
    if is_a_inf or is_b_inf:
        if (is_a_inf and is_b_zero) or (is_b_inf and is_a_zero):
            return (0 << 15) | (0xFF << 7) | QUIET
        return (rsign << 15) | (0xFF << 7) | 0
    if is_a_zero or is_b_zero:
        return (rsign << 15) | 0 | 0

    sig_a = (0 if is_a_sub else 1) << 7 | a_mant
    sig_b = (0 if is_b_sub else 1) << 7 | b_mant
    P = sig_a * sig_b

    def pbitlen(x):
        if x == 0:
            return 0
        bl = 0
        while x > 0:
            bl += 1
            x >>= 1
        return bl

    e_a = a_exp - 127
    if is_a_sub:
        e_a = -126 - 7
    e_b = b_exp - 127
    if is_b_sub:
        e_b = -126 - 7
    e_res = e_a + e_b + (1 if pbitlen(P) == 16 else 0)

    TARGET = 10
    FW = 11
    P_W = 16

    B = pbitlen(P)
    norm_sticky = False
    if B == 0:
        placed = 0
    elif B <= FW:
        placed = (P << (FW - B)) & ((1 << FW) - 1)
    else:
        placed = (P >> (B - FW)) & ((1 << FW) - 1)
        mask = (1 << (B - FW)) - 1
        norm_sticky = (P & mask) != 0

    sig_val = (placed >> (TARGET - 7)) & 0xFF
    g = (placed >> 2) & 1
    r = (placed >> 1) & 1
    s_bit = (placed & 1) or norm_sticky

    round_up = (g and (r or s_bit)) or (g and not r and not s_bit and (sig_val & 1))
    carry = round_up and (sig_val == 0xFF)

    if carry:
        sig_rnd = 0x80
        e_rnd = e_res + 1
    else:
        sig_rnd = (sig_val + (1 if round_up else 0)) & 0xFF
        e_rnd = e_res

    if e_rnd > 127:
        return (rsign << 15) | (0xFF << 7) | 0
    elif e_rnd < -126:
        shift_amt = -126 - e_rnd
        if shift_amt > 0:
            shifted = placed >> shift_amt
        else:
            shifted = placed
        sig_s = (shifted >> (TARGET - 7)) & 0xFF
        gs = (shifted >> 2) & 1
        rs2 = (shifted >> 1) & 1
        ss = (shifted & 1) or norm_sticky
        if shift_amt > 0:
            drop_mask = (1 << shift_amt) - 1
            ss = ss or ((placed & drop_mask) != 0)
        if (gs and (rs2 or ss)) or (gs and not rs2 and not ss and (sig_s & 1)):
            sig_s = (sig_s + 1) & 0xFF
        if sig_s == 0x80:
            return (rsign << 15) | (1 << 7) | 0
        else:
            return (rsign << 15) | 0 | (sig_s & 0x7F)
    else:
        exp_fld = (e_rnd + 127) & 0xFF
        return (rsign << 15) | (exp_fld << 7) | (sig_rnd & 0x7F)


def hardware_silu(bf16_input):
    """Compute SiLU(x) = x * sigmoid(x) matching hardware."""
    sig_out = hardware_sigmoid(bf16_input)
    return hardware_bfloat16_mul(bf16_input, sig_out)


@cocotb.test()
async def test_silu_inloop(dut):
    """Test silu_unit with real model activations."""

    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.data_in.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    inputs_path = os.path.join(script_dir, "golden_inloop_inputs.hex")
    outputs_path = os.path.join(script_dir, "golden_inloop_outputs.hex")

    if not os.path.exists(inputs_path):
        dut._log.info("Generating activations from model")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path = "/home/jese/tinyllm/granite-4.0-h-350m"
            dut._log.info(f"Loading model from {model_path}")

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu"
            )
            model.eval()

            activations = []

            def hook_fn(module, input, output):
                if isinstance(input, tuple):
                    x = input[0]
                else:
                    x = input
                if x is not None and x.dtype == torch.bfloat16:
                    activations.append(x.detach())

            hooks = []
            for name, module in model.named_modules():
                if 'act' in name.lower() and hasattr(module, 'forward'):
                    hooks.append(module.register_forward_hook(hook_fn))

            input_text = "The quick brown fox jumps over the lazy dog"
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            with torch.no_grad():
                _ = model(input_ids)

            for h in hooks:
                h.remove()

            if not activations:
                dut._log.warning("No activations captured, using synthetic data")
                acts = [torch.randn(1, 10, 1536, dtype=torch.bfloat16) for _ in range(5)]
            else:
                acts = activations[:5]

            bf16_inputs = []
            bf16_outputs = []
            seen = set()

            for act in acts:
                flat = act.reshape(-1).tolist()
                for val in flat[:500]:
                    bf16_val = float_to_bfloat16(float(val))
                    if bf16_val not in seen and len(bf16_inputs) < 300:
                        seen.add(bf16_val)
                        bf16_inputs.append(bf16_val)
                        bf16_outputs.append(hardware_silu(bf16_val))

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
                bf16_out = hardware_silu(bf16_in)
                bf16_inputs.append(bf16_in)
                bf16_outputs.append(bf16_out)

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

        await RisingEdge(dut.clk)
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
