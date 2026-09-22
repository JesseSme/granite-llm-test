#!/usr/bin/env python3
"""Generate golden sample data for sigmoid_unit cocotb testbench."""

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

def compute_sigmoid_lut_entry(addr):
    """Compute the bfloat16 sigmoid value for a given LUT address.
    Address i corresponds to x = i / 32.0."""
    x = addr / 32.0
    sig = 1.0 / (1.0 + math.exp(-x))
    # Clamp to [0, 1) - never exactly 1.0
    sig = min(sig, 0.999999)
    return float_to_bfloat16(sig)

def compute_address(bf16_val):
    """Compute the LUT address from a bfloat16 magnitude.
    addr = |x| * 32, clamped to [0, 255]."""
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
    """Compute the expected hardware output for a given bfloat16 input."""
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
        return 0x0000 if sign else 0x3F80  # -inf→0, +inf→1
    elif is_zero:
        return 0x3F00  # ±0 → 0.5

    # Compute LUT address from magnitude
    magnitude_bf16 = bf16_input & 0x7FFF  # clear sign
    addr = compute_address(magnitude_bf16)

    # Look up sigmoid value
    pos_val = compute_sigmoid_lut_entry(addr)

    # Compute neg_lut entry: 1 - sigmoid(addr/32) as bfloat16
    x = addr / 32.0
    neg_sig = 1.0 - (1.0 / (1.0 + math.exp(-x)))
    neg_val = float_to_bfloat16(neg_sig)

    if sign:
        return neg_val
    else:
        return pos_val

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
            outputs.append(hardware_sigmoid(0x7FC0))
        else:
            bf16_in = float_to_bfloat16(val)
            bf16_out = hardware_sigmoid(bf16_in)
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
        actual_hardware = 1.0 / (1.0 + math.exp(-x)) if x == x else 0.5
        print(f"  x={x:8.4f}: hw_out=0x{outputs[i]:04X} ({expected:.6f}), py_sig={actual_hardware:.6f}")

if __name__ == "__main__":
    main()
