#!/usr/bin/env python3
"""Generate golden sample data for silu_unit cocotb testbench."""

import struct
import random
import math
import os

def float_to_bfloat16(val):
    """Convert Python float to 16-bit bfloat16 pattern."""
    f32 = struct.pack('f', val)
    i32 = struct.unpack('I', f32)[0]
    rounding_bias = (1 << 15) + ((i32 >> 16) & 1)
    i32 += rounding_bias
    bf16 = i32 >> 16
    return bf16 & 0xFFFF

def bfloat16_to_float(val):
    """Convert 16-bit bfloat16 pattern to Python float."""
    val = val & 0xFFFF
    float32_val = val << 16
    return struct.unpack('f', struct.pack('I', float32_val))[0]

def hardware_sigmoid(bf16_input):
    """Compute the expected hardware sigmoid output for a given bfloat16 input."""
    sign = (bf16_input >> 15) & 1
    exp = (bf16_input >> 7) & 0xFF
    mant = bf16_input & 0x7F

    is_nan = (exp == 0xFF) and (mant != 0)
    is_inf = (exp == 0xFF) and (mant == 0)
    is_zero = (exp == 0) and (mant == 0)
    is_sub = (exp == 0) and (mant != 0)

    if is_nan:
        return 0x3F00  # 0.5
    elif is_inf:
        return 0x0000 if sign else 0x3F80
    elif is_zero:
        return 0x3F00  # ±0 → 0.5

    # LUT address from magnitude
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
    QUIET = 0x7F  # all-ones mantissa = quiet NaN

    # Special cases
    if is_a_nan or is_b_nan:
        payload = a_mant if is_a_nan else b_mant
        return (0 << 15) | (0xFF << 7) | (payload | QUIET)
    if is_a_inf or is_b_inf:
        if (is_a_inf and is_b_zero) or (is_b_inf and is_a_zero):
            return (0 << 15) | (0xFF << 7) | QUIET
        return (rsign << 15) | (0xFF << 7) | 0
    if is_a_zero or is_b_zero:
        return (rsign << 15) | 0 | 0

    # Build significands (8 bits)
    s_a = (0, a_mant) if is_a_sub else (1, a_mant)
    s_b = (0, b_mant) if is_b_sub else (1, b_mant)

    sig_a = s_a[0] << 7 | s_a[1]  # 8-bit significand
    sig_b = s_b[0] << 7 | s_b[1]

    # Exact product (16 bits)
    P = sig_a * sig_b

    # Product bit length
    def pbitlen(x):
        if x == 0:
            return 0
        bl = 0
        while x > 0:
            bl += 1
            x >>= 1
        return bl

    # Leading-bit exponents
    e_a = a_exp - 127
    if is_a_sub:
        e_a = -126 - 7
    e_b = b_exp - 127
    if is_b_sub:
        e_b = -126 - 7

    # Result exponent
    e_res = e_a + e_b + (1 if pbitlen(P) == 16 else 0)

    # Place product: MSB at position TARGET = 10
    TARGET = 10
    FW = 11
    P_W = 16

    B = pbitlen(P)
    norm_sticky = False
    if B == 0:
        placed = 0
        norm_sticky = False
    elif B <= FW:
        placed = (P << (FW - B)) & ((1 << FW) - 1)
        norm_sticky = False
    else:
        placed = (P >> (B - FW)) & ((1 << FW) - 1)
        mask = (1 << (B - FW)) - 1
        norm_sticky = (P & mask) != 0

    # Extract sig, G, R, S
    sig_val = (placed >> (TARGET - 7)) & 0xFF  # bits [TARGET-1:TARGET-8]
    g = (placed >> 2) & 1
    r = (placed >> 1) & 1
    s_bit = (placed & 1) or norm_sticky

    # RNE rounding
    round_up = (g and (r or s_bit)) or (g and not r and not s_bit and (sig_val & 1))
    carry = round_up and (sig_val == 0xFF)

    if carry:
        sig_rnd = 0x80
        e_rnd = e_res + 1
    else:
        sig_rnd_val = sig_val + (1 if round_up else 0)
        sig_rnd = sig_rnd_val & 0xFF
        e_rnd = e_res

    # Re-encode
    if e_rnd > 127:
        return (rsign << 15) | (0xFF << 7) | 0
    elif e_rnd < -126:
        # Subnormal result
        shift_amt = -126 - e_rnd
        if shift_amt > 0:
            shifted = placed >> shift_amt
        else:
            shifted = placed
        sig_s = (shifted >> (TARGET - 7)) & 0xFF
        gs = (shifted >> 2) & 1
        rs2 = (shifted >> 1) & 1
        ss = (shifted & 1) or norm_sticky
        # OR of dropped bits
        if shift_amt > 0:
            drop_mask = ((1 << shift_amt) - 1)
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


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    random.seed(42)

    inputs = []
    outputs = []

    # Test cases: specific values + random values
    test_values = [
        # Special values
        0.0, -0.0,
        float('nan'),
        float('inf'), float('-inf'),
        # Small values
        0.001, -0.001, 0.01, -0.01, 0.1, -0.1,
        # Medium values
        0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0,
        4.0, -4.0, 5.0, -5.0,
        # Large values (near saturation)
        6.0, -6.0, 7.0, -7.0, 8.0, -8.0,
        10.0, -10.0, 100.0, -100.0,
    ]

    # Add random bfloat16 values
    for _ in range(200):
        exp = random.randint(110, 134)
        mant = random.randint(0, 127)
        bf16_val = (exp << 7) | mant
        val = bfloat16_to_float(bf16_val)
        if -8.0 <= val <= 8.0:
            test_values.append(val)
            test_values.append(-val)

    for val in test_values:
        if val != val:  # NaN check
            inputs.append(0x7FC0)
            outputs.append(hardware_silu(0x7FC0))
        else:
            bf16_in = float_to_bfloat16(val)
            bf16_out = hardware_silu(bf16_in)
            inputs.append(bf16_in)
            outputs.append(bf16_out)

    # Write hex files
    inputs_path = os.path.join(script_dir, "golden_inputs.hex")
    outputs_path = os.path.join(script_dir, "golden_outputs.hex")

    with open(inputs_path, 'w') as f:
        for val in inputs:
            f.write(f"{val:04X}\n")

    with open(outputs_path, 'w') as f:
        for val in outputs:
            f.write(f"{val:04X}\n")

    print(f"Generated {len(inputs)} test vectors")
    print(f"Inputs:  {inputs_path}")
    print(f"Outputs: {outputs_path}")

    # Verify a few values
    print("\nVerification:")
    for i in range(min(20, len(inputs))):
        x = bfloat16_to_float(inputs[i])
        expected = bfloat16_to_float(outputs[i])
        actual_silu = x * (1.0 / (1.0 + math.exp(-x))) if x == x else 0.0
        print(f"  x={x:8.4f}: hw_out=0x{outputs[i]:04X} ({expected:.6f}), py_silu={actual_silu:.6f}")


if __name__ == "__main__":
    main()
