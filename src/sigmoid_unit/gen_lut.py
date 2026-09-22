#!/usr/bin/env python3
"""Generate sigmoid lookup table values for the hardware sigmoid unit.

The LUT is indexed by the top 8 bits of the bfloat16 magnitude.
Each entry is 12-bit fixed-point: [11:4] = integer part, [3:0] = fraction.
The table covers the range [0, ~2.0] which is where sigmoid transitions
from 0.5 to ~0.88. For larger values, sigmoid saturates toward 1.0.
"""

import struct
import math

def bfloat16_to_float(val):
    """Convert 16-bit bfloat16 pattern to Python float."""
    # bfloat16: sign(1) + exp(8) + mantissa(7)
    # Reinterpret as float32 by shifting left 16 bits
    val = val & 0xFFFF
    # Convert to float32 by zero-extending (bfloat16 is upper 16 bits of float32)
    float32_val = val << 16
    return struct.unpack('f', struct.pack('I', float32_val))[0]

def float_to_bfloat16(val):
    """Convert Python float to 16-bit bfloat16 pattern."""
    f32 = struct.pack('f', val)
    i32 = struct.unpack('I', f32)[0]
    # Round to nearest even for bfloat16
    # bfloat16 has 7 mantissa bits, float32 has 23
    # Round at bit 16 (the lowest bfloat16 bit)
    rounding_bias = (1 << 15) + ((i32 >> 16) & 1)
    i32 += rounding_bias
    bf16 = i32 >> 16
    return bf16 & 0xFFFF

def compute_sigmoid_lut():
    """Compute sigmoid values for LUT entries 0-255.
    
    Address bits [7:1] = exponent field (8-bit)
    Address bit [0] = mantissa MSB (bit 6)
    
    For each entry, we compute sigmoid(midpoint_value) where midpoint_value
    is the bfloat16 value at the center of the segment.
    """
    lut = []
    
    for addr in range(256):
        exp = (addr >> 1) & 0xFF  # 8-bit exponent
        mant_msb = addr & 0x1     # mantissa MSB
        
        # Construct a bfloat16 magnitude with this exp and mantissa
        # mantissa = {mant_msb, 0, 0, 0, 0, 0, 0} for the midpoint
        mantissa = mant_msb << 6  # bit 6 set, rest 0
        bf16_val = (exp << 7) | mantissa
        
        # Convert to float
        float_val = bfloat16_to_float(bf16_val)
        
        # Compute sigmoid
        sig = 1.0 / (1.0 + math.exp(-float_val))
        
        # Clamp to [0, 1)
        sig = max(0.0, min(sig, 0.99999))
        
        # Convert to 12-bit fixed-point: [11:4] = integer, [3:0] = fraction
        # Scale: multiply by 256 (8-bit fraction equivalent for 12-bit format)
        fixed_val = int(sig * 256.0 + 0.5)
        fixed_val = min(fixed_val, 0xFFF)  # Clamp to 12 bits
        
        lut.append(fixed_val)
    
    return lut

def compute_sigmoid_lut_256():
    """Compute sigmoid values for 256-entry LUT.
    
    Each entry stores the sigmoid value for the midpoint of its segment.
    The segment is defined by the 8-bit address mapping to bfloat16 values.
    """
    lut = []
    
    for addr in range(256):
        # Map address to a bfloat16 value
        # Address 0: bfloat16(0.0) = 0x0000
        # Address 127: bfloat16 with exp=0, mantissa=127 << 1 = 0x007E
        # Address 128: bfloat16 with exp=1, mantissa=0 = 0x3F80... wait
        
        # Actually, let me map address to a meaningful range
        # Use address as the upper 8 bits of a 12-bit fixed-point value
        # in the range [0, 16.0) with 4 fractional bits
        # Address 0 = 0.0, Address 255 = 15.9375
        
        # Better: use a non-linear mapping that focuses on the sigmoid transition
        # Map address to input value using: x = address * 8.0 / 256 = address / 32.0
        # This gives range [0, 7.96875] which covers the sigmoid transition
        
        x = addr * 8.0 / 256.0  # Range [0, ~8]
        
        sig = 1.0 / (1.0 + math.exp(-x))
        
        # Convert to 12-bit fixed-point (8.4 format)
        fixed_val = int(sig * 256.0 + 0.5)
        fixed_val = min(fixed_val, 0xFFF)
        
        lut.append(fixed_val)
    
    return lut

def generate_verilog_header(lut, filename):
    """Generate a SystemVerilog header file with the LUT."""
    with open(filename, 'w') as f:
        f.write("// Auto-generated sigmoid lookup table.\n")
        f.write("// 256 entries, 12-bit fixed-point (8.4 format).\n")
        f.write("// Index = input_value * 32.0 (range [0, ~8]).\n")
        f.write("//\n")
        f.write("// For negative inputs, use symmetry: sigmoid(-x) = 1 - sigmoid(x).\n")
        f.write("// For inputs > 8.0, output = 1.0 (0xFFF).\n\n")
        f.write("`define SIGMOID_LUT_WIDTH 12\n")
        f.write("`define SIGMOID_LUT_DEPTH 256\n\n")
        f.write("function automatic logic [11:0] sigmoid_lut(\n")
        f.write("  input logic [7:0] addr\n")
        f.write(");\n")
        f.write("  case (addr)\n")
        for i, val in enumerate(lut):
            f.write(f"    8'h{i:02X}: sigmoid_lut = 12'h{val:03X};\n")
        f.write("    default: sigmoid_lut = 12'hFFF;\n")
        f.write("  endcase\n")
        f.write("endfunction\n")

if __name__ == "__main__":
    lut = compute_sigmoid_lut_256()
    
    # Print first few entries for verification
    print("First 20 LUT entries:")
    for i in range(20):
        x = i * 8.0 / 256.0
        expected = 1.0 / (1.0 + math.exp(-x))
        fixed = lut[i] / 256.0
        print(f"  addr={i:3d}: x={x:6.3f}, expected={expected:.6f}, fixed={fixed:.6f}, hex=0x{lut[i]:03X}")
    
    print(f"\nEntry at x=1.0 (addr=32): {lut[32]/256:.6f} (expected: {1/(1+math.exp(-1)):.6f})")
    print(f"Entry at x=2.0 (addr=64): {lut[64]/256:.6f} (expected: {1/(1+math.exp(-2)):.6f})")
    print(f"Entry at x=4.0 (addr=128): {lut[128]/256:.6f} (expected: {1/(1+math.exp(-4)):.6f})")
    
    # Generate header file
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    header_path = os.path.join(script_dir, "sigmoid_lut.svh")
    generate_verilog_header(lut, header_path)
    print(f"\nGenerated LUT header: {header_path}")
    
    # Also save the LUT as a Python file for the testbench
    py_path = os.path.join(script_dir, "sigmoid_lut_values.py")
    with open(py_path, 'w') as f:
        f.write("# Auto-generated sigmoid LUT values (12-bit fixed-point, 8.4 format)\n")
        f.write("# Index = input_value * 32.0\n")
        f.write("SIGMOID_LUT = [\n")
        for i in range(0, 256, 8):
            row = lut[i:i+8]
            f.write("    " + ", ".join(f"0x{v:03X}" for v in row) + ",\n")
        f.write("]\n")
    print(f"Generated LUT Python file: {py_path}")
