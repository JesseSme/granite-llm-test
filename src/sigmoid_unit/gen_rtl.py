#!/usr/bin/env python3
"""Generate sigmoid_unit.sv with correct LUT values."""

import struct
import math

def float_to_bfloat16(val):
    f32 = struct.pack('f', val)
    i32 = struct.unpack('I', f32)[0]
    rounding_bias = (1 << 15) + ((i32 >> 16) & 1)
    i32 += rounding_bias
    bf16 = i32 >> 16
    return bf16 & 0xFFFF

# Generate LUT entries
lut_lines = []
for addr in range(256):
    x = addr / 32.0
    sig = 1.0 / (1.0 + math.exp(-x))
    sig = min(sig, 0.999999)
    bf16 = float_to_bfloat16(sig)
    lut_lines.append(f'      8\'h{addr:02X}: lut_val = 16\'h{bf16:04X};')

lut_block = '\n'.join(lut_lines)

sv_content = f'''// Sigmoid activation function: σ(x) = 1 / (1 + exp(-x))
//
// Single-cycle combinational design using a 256-entry lookup table.
// For bfloat16 I/O. Uses symmetry: σ(-x) = 1 - σ(x).
//
// LUT approach:
//   - LUT has 256 entries, addr = |x| * 32 (covers range [0, ~8])
//   - Entry i stores sigmoid(i/32) as bfloat16
//   - For negative inputs: result = 1 - lut(|x|)
//   - For |x| > 8: result = 1.0 (positive) or 0.0 (negative)
//
// Latency: 0 cycles (purely combinational, registered output).

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */

module sigmoid_unit (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        valid_in,
  input  logic [15:0] data_in,
  output logic [15:0] data_out,
  output logic        valid_out
);

  localparam int W      = 16;
  localparam int W_EXP  = 8;
  localparam int W_MANT = 7;
  localparam int BIAS   = 127;
  localparam int EXP_ALL = (1 << W_EXP) - 1;

  logic [W-1:0] result_reg;

  // --------------------------------------------------------------- unpack
  logic        sign;
  logic [W_EXP-1:0]  exp;
  logic [W_MANT-1:0] mant;
  assign sign = data_in[W-1];
  assign exp  = data_in[W-2 -: W_EXP];
  assign mant = data_in[W_MANT-1:0];

  logic is_nan, is_inf, is_zero, is_sub;
  assign is_nan  = (exp == EXP_ALL[W_EXP-1:0]) && (mant != '0);
  assign is_inf  = (exp == EXP_ALL[W_EXP-1:0]) && (mant == '0);
  assign is_zero = (exp == '0) && (mant == '0);
  assign is_sub  = (exp == '0) && (mant != '0);

  // ----------------------------------------------------------- LUT addr
  // addr = |x| * 32, clamped to [0, 255]
  //
  // For a normal bfloat16 with exponent e and mantissa m:
  //   |x| = 2^(e - 127) * (1 + m/128)
  //   addr = |x| * 32 = 2^(e - 122) * (1 + m/128)

  logic [7:0] lut_addr;

  always_comb begin
    if (is_sub || is_zero || exp < 8'd122) begin
      lut_addr = 8'd0;
    end else if (exp >= 8'd130) begin
      lut_addr = 8'd255;
    end else begin
      unique case (exp)
        8'd122: lut_addr = 8'd1;
        8'd123: lut_addr = {6'b000000, 1'b1, mant[6]};
        8'd124: lut_addr = {5'b00000, 1'b1, mant[6:5]};
        8'd125: lut_addr = {4'b0000, 1'b1, mant[6:4]};
        8'd126: lut_addr = {3'b000, 1'b1, mant[6:3]};
        8'd127: lut_addr = {2'b00, 1'b1, mant[6:2]};
        8'd128: lut_addr = {1'b0, 1'b1, mant[6:1]};
        8'd129: lut_addr = {1'b1, mant[6:0]};
        default: lut_addr = 8'd0;
      endcase
    end
  end

  // --------------------------------------------------------------- LUT
  // Pre-computed bfloat16 sigmoid values.
  // Entry i = sigmoid(i / 32.0) as bfloat16.

  logic [W-1:0] lut_val;
  always_comb begin
    unique case (lut_addr)
{lut_block}
    endcase
  end

  // --------------------------------------------------------- apply sign
  // For negative inputs: σ(-x) = 1 - σ(x)
  // 1.0 = 0x3F80 in bfloat16

  logic [W-1:0] abs_result;
  logic [W-1:0] neg_result;
  logic [W-1:0] final_result;

  assign abs_result = lut_val;
  assign neg_result = 16'h3F80 - lut_val;

  always_comb begin
    if (is_nan) begin
      final_result = 16'h3F00;
    end else if (is_inf) begin
      final_result = sign ? 16'h0000 : 16'h3F80;
    end else if (is_zero) begin
      final_result = 16'h3F00;
    end else begin
      final_result = sign ? neg_result : abs_result;
    end
  end

  // ---------------------------------------------------------- pipeline
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      result_reg <= '0;
      valid_out  <= 1'b0;
    end else begin
      result_reg <= final_result;
      valid_out  <= valid_in;
    end
  end

  assign data_out = result_reg;

endmodule
'''

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, 'sigmoid_unit.sv')
with open(output_path, 'w') as f:
    f.write(sv_content)
print(f'Generated {output_path}')
