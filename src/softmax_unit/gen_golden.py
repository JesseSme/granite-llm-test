#!/usr/bin/env python3
"""Generate golden sample data for softmax_unit cocotb testbench."""

import struct
import random
import math
import os

def float_to_hex(val):
    """Convert Python float to float32 hex pattern."""
    return struct.unpack('I', struct.pack('f', val))[0]

def hex_to_float(hex_val):
    """Convert float32 hex pattern to Python float."""
    return struct.unpack('f', struct.pack('I', hex_val))[0]

def compute_softmax(row):
    """Compute numerically stable softmax for a row of float32 values."""
    max_val = max(row)
    exp_vals = [math.exp(x - max_val) for x in row]
    sum_exp = sum(exp_vals)
    return [e / sum_exp for e in exp_vals]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    random.seed(42)

    all_inputs = []
    all_outputs = []

    # Test case 1: All zeros
    row = [0.0] * 8
    out = compute_softmax(row)
    all_inputs.append(row)
    all_outputs.append(out)

    # Test case 2: Uniform values
    row = [1.0] * 8
    out = compute_softmax(row)
    all_inputs.append(row)
    all_outputs.append(out)

    # Test case 3: Large negative values
    row = [-100.0, -200.0, -300.0, -400.0, -500.0, -600.0, -700.0, -800.0]
    out = compute_softmax(row)
    all_inputs.append(row)
    all_outputs.append(out)

    # Test case 4: Large positive values
    row = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
    out = compute_softmax(row)
    all_inputs.append(row)
    all_outputs.append(out)

    # Test case 5: Mixed positive and negative
    row = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 7.0]
    out = compute_softmax(row)
    all_inputs.append(row)
    all_outputs.append(out)

    # Test case 6: One dominant value
    row = [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    out = compute_softmax(row)
    all_inputs.append(row)
    all_outputs.append(out)

    # Test case 7: Negative dominant
    row = [0.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0]
    out = compute_softmax(row)
    all_inputs.append(row)
    all_outputs.append(out)

    # Test case 8: Typical attention score patterns
    for _ in range(10):
        row = [random.gauss(0, 1) for _ in range(8)]
        out = compute_softmax(row)
        all_inputs.append(row)
        all_outputs.append(out)

    # Test case 9: Random rows of exactly 8 elements
    for _ in range(20):
        row = [random.gauss(0, 2) for _ in range(8)]
        out = compute_softmax(row)
        all_inputs.append(row)
        all_outputs.append(out)

    # Flatten all rows into a single sequence
    flat_inputs = []
    flat_outputs = []
    for row_in, row_out in zip(all_inputs, all_outputs):
        flat_inputs.extend(row_in)
        flat_outputs.extend(row_out)

    # Convert to hex
    hex_inputs = [float_to_hex(v) for v in flat_inputs]
    hex_outputs = [float_to_hex(v) for v in flat_outputs]

    # Write hex files
    inputs_path = os.path.join(script_dir, "golden_inputs.hex")
    outputs_path = os.path.join(script_dir, "golden_outputs.hex")

    with open(inputs_path, 'w') as f:
        for h in hex_inputs:
            f.write(f"{h:08X}\n")

    with open(outputs_path, 'w') as f:
        for h in hex_outputs:
            f.write(f"{h:08X}\n")

    print(f"Generated {len(flat_inputs)} test values ({len(all_inputs)} rows)")
    print(f"Inputs:  {inputs_path}")
    print(f"Outputs: {outputs_path}")

    # Verify a few values
    print("\nVerification:")
    for i in range(min(8, len(flat_inputs))):
        x = flat_inputs[i]
        expected = flat_outputs[i]
        actual_hex = hex_outputs[i]
        actual = hex_to_float(actual_hex)
        print(f"  [{i:3d}] x={x:10.4f}  expected={expected:.6f}  actual={actual:.6f}  "
              f"hex_in={hex_inputs[i]:08X}  hex_out={actual_hex:08X}")

    # Verify softmax property: outputs sum to ~1.0
    print("\nRow sum verification:")
    idx = 0
    for row_in, row_out in zip(all_inputs, all_outputs):
        row_sum = sum(row_out)
        print(f"  Row {all_inputs.index(row_in):3d}: sum={row_sum:.10f}  "
              f"(should be ~1.0, err={abs(row_sum - 1.0):.2e})")
        idx += 1

if __name__ == "__main__":
    main()
