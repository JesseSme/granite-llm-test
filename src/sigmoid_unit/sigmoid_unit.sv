// Sigmoid activation function: σ(x) = 1 / (1 + exp(-x))
//
// Single-cycle combinational design using two 256-entry lookup tables.
// For bfloat16 I/O. Uses symmetry: σ(-x) = 1 - σ(x).
//
// LUT approach:
//   - pos_lut: stores sigmoid(i/32) for i in [0,255]
//   - neg_lut: stores 1 - sigmoid(i/32) for i in [0,255]
//   - For negative inputs, use neg_lut; for positive, use pos_lut
//   - addr = |x| * 32, clamped to [0, 255]
//   - For |x| > 8: result = 1.0 (positive) or 0.0 (negative)
//
// Latency: 1 cycle (registered output).

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

  // --------------------------------------------------------------- LUTs
  logic [W-1:0] pos_lut_val;
  always_comb begin
    unique case (lut_addr)
      8'h00: pos_lut_val = 16'h3F00;
      8'h01: pos_lut_val = 16'h3F02;
      8'h02: pos_lut_val = 16'h3F04;
      8'h03: pos_lut_val = 16'h3F06;
      8'h04: pos_lut_val = 16'h3F08;
      8'h05: pos_lut_val = 16'h3F0A;
      8'h06: pos_lut_val = 16'h3F0C;
      8'h07: pos_lut_val = 16'h3F0E;
      8'h08: pos_lut_val = 16'h3F10;
      8'h09: pos_lut_val = 16'h3F12;
      8'h0A: pos_lut_val = 16'h3F14;
      8'h0B: pos_lut_val = 16'h3F16;
      8'h0C: pos_lut_val = 16'h3F18;
      8'h0D: pos_lut_val = 16'h3F1A;
      8'h0E: pos_lut_val = 16'h3F1C;
      8'h0F: pos_lut_val = 16'h3F1D;
      8'h10: pos_lut_val = 16'h3F1F;
      8'h11: pos_lut_val = 16'h3F21;
      8'h12: pos_lut_val = 16'h3F23;
      8'h13: pos_lut_val = 16'h3F25;
      8'h14: pos_lut_val = 16'h3F27;
      8'h15: pos_lut_val = 16'h3F29;
      8'h16: pos_lut_val = 16'h3F2A;
      8'h17: pos_lut_val = 16'h3F2C;
      8'h18: pos_lut_val = 16'h3F2E;
      8'h19: pos_lut_val = 16'h3F30;
      8'h1A: pos_lut_val = 16'h3F31;
      8'h1B: pos_lut_val = 16'h3F33;
      8'h1C: pos_lut_val = 16'h3F35;
      8'h1D: pos_lut_val = 16'h3F36;
      8'h1E: pos_lut_val = 16'h3F38;
      8'h1F: pos_lut_val = 16'h3F3A;
      8'h20: pos_lut_val = 16'h3F3B;
      8'h21: pos_lut_val = 16'h3F3D;
      8'h22: pos_lut_val = 16'h3F3E;
      8'h23: pos_lut_val = 16'h3F40;
      8'h24: pos_lut_val = 16'h3F41;
      8'h25: pos_lut_val = 16'h3F43;
      8'h26: pos_lut_val = 16'h3F44;
      8'h27: pos_lut_val = 16'h3F46;
      8'h28: pos_lut_val = 16'h3F47;
      8'h29: pos_lut_val = 16'h3F48;
      8'h2A: pos_lut_val = 16'h3F4A;
      8'h2B: pos_lut_val = 16'h3F4B;
      8'h2C: pos_lut_val = 16'h3F4C;
      8'h2D: pos_lut_val = 16'h3F4E;
      8'h2E: pos_lut_val = 16'h3F4F;
      8'h2F: pos_lut_val = 16'h3F50;
      8'h30: pos_lut_val = 16'h3F51;
      8'h31: pos_lut_val = 16'h3F52;
      8'h32: pos_lut_val = 16'h3F54;
      8'h33: pos_lut_val = 16'h3F55;
      8'h34: pos_lut_val = 16'h3F56;
      8'h35: pos_lut_val = 16'h3F57;
      8'h36: pos_lut_val = 16'h3F58;
      8'h37: pos_lut_val = 16'h3F59;
      8'h38: pos_lut_val = 16'h3F5A;
      8'h39: pos_lut_val = 16'h3F5B;
      8'h3A: pos_lut_val = 16'h3F5C;
      8'h3B: pos_lut_val = 16'h3F5D;
      8'h3C: pos_lut_val = 16'h3F5E;
      8'h3D: pos_lut_val = 16'h3F5F;
      8'h3E: pos_lut_val = 16'h3F60;
      8'h3F: pos_lut_val = 16'h3F61;
      8'h40: pos_lut_val = 16'h3F61;
      8'h41: pos_lut_val = 16'h3F62;
      8'h42: pos_lut_val = 16'h3F63;
      8'h43: pos_lut_val = 16'h3F64;
      8'h44: pos_lut_val = 16'h3F65;
      8'h45: pos_lut_val = 16'h3F65;
      8'h46: pos_lut_val = 16'h3F66;
      8'h47: pos_lut_val = 16'h3F67;
      8'h48: pos_lut_val = 16'h3F68;
      8'h49: pos_lut_val = 16'h3F68;
      8'h4A: pos_lut_val = 16'h3F69;
      8'h4B: pos_lut_val = 16'h3F6A;
      8'h4C: pos_lut_val = 16'h3F6A;
      8'h4D: pos_lut_val = 16'h3F6B;
      8'h4E: pos_lut_val = 16'h3F6B;
      8'h4F: pos_lut_val = 16'h3F6C;
      8'h50: pos_lut_val = 16'h3F6D;
      8'h51: pos_lut_val = 16'h3F6D;
      8'h52: pos_lut_val = 16'h3F6E;
      8'h53: pos_lut_val = 16'h3F6E;
      8'h54: pos_lut_val = 16'h3F6F;
      8'h55: pos_lut_val = 16'h3F6F;
      8'h56: pos_lut_val = 16'h3F70;
      8'h57: pos_lut_val = 16'h3F70;
      8'h58: pos_lut_val = 16'h3F71;
      8'h59: pos_lut_val = 16'h3F71;
      8'h5A: pos_lut_val = 16'h3F71;
      8'h5B: pos_lut_val = 16'h3F72;
      8'h5C: pos_lut_val = 16'h3F72;
      8'h5D: pos_lut_val = 16'h3F73;
      8'h5E: pos_lut_val = 16'h3F73;
      8'h5F: pos_lut_val = 16'h3F73;
      8'h60: pos_lut_val = 16'h3F74;
      8'h61: pos_lut_val = 16'h3F74;
      8'h62: pos_lut_val = 16'h3F75;
      8'h63: pos_lut_val = 16'h3F75;
      8'h64: pos_lut_val = 16'h3F75;
      8'h65: pos_lut_val = 16'h3F76;
      8'h66: pos_lut_val = 16'h3F76;
      8'h67: pos_lut_val = 16'h3F76;
      8'h68: pos_lut_val = 16'h3F76;
      8'h69: pos_lut_val = 16'h3F77;
      8'h6A: pos_lut_val = 16'h3F77;
      8'h6B: pos_lut_val = 16'h3F77;
      8'h6C: pos_lut_val = 16'h3F78;
      8'h6D: pos_lut_val = 16'h3F78;
      8'h6E: pos_lut_val = 16'h3F78;
      8'h6F: pos_lut_val = 16'h3F78;
      8'h70: pos_lut_val = 16'h3F78;
      8'h71: pos_lut_val = 16'h3F79;
      8'h72: pos_lut_val = 16'h3F79;
      8'h73: pos_lut_val = 16'h3F79;
      8'h74: pos_lut_val = 16'h3F79;
      8'h75: pos_lut_val = 16'h3F7A;
      8'h76: pos_lut_val = 16'h3F7A;
      8'h77: pos_lut_val = 16'h3F7A;
      8'h78: pos_lut_val = 16'h3F7A;
      8'h79: pos_lut_val = 16'h3F7A;
      8'h7A: pos_lut_val = 16'h3F7A;
      8'h7B: pos_lut_val = 16'h3F7B;
      8'h7C: pos_lut_val = 16'h3F7B;
      8'h7D: pos_lut_val = 16'h3F7B;
      8'h7E: pos_lut_val = 16'h3F7B;
      8'h7F: pos_lut_val = 16'h3F7B;
      8'h80: pos_lut_val = 16'h3F7B;
      8'h81: pos_lut_val = 16'h3F7C;
      8'h82: pos_lut_val = 16'h3F7C;
      8'h83: pos_lut_val = 16'h3F7C;
      8'h84: pos_lut_val = 16'h3F7C;
      8'h85: pos_lut_val = 16'h3F7C;
      8'h86: pos_lut_val = 16'h3F7C;
      8'h87: pos_lut_val = 16'h3F7C;
      8'h88: pos_lut_val = 16'h3F7C;
      8'h89: pos_lut_val = 16'h3F7D;
      8'h8A: pos_lut_val = 16'h3F7D;
      8'h8B: pos_lut_val = 16'h3F7D;
      8'h8C: pos_lut_val = 16'h3F7D;
      8'h8D: pos_lut_val = 16'h3F7D;
      8'h8E: pos_lut_val = 16'h3F7D;
      8'h8F: pos_lut_val = 16'h3F7D;
      8'h90: pos_lut_val = 16'h3F7D;
      8'h91: pos_lut_val = 16'h3F7D;
      8'h92: pos_lut_val = 16'h3F7D;
      8'h93: pos_lut_val = 16'h3F7D;
      8'h94: pos_lut_val = 16'h3F7E;
      8'h95: pos_lut_val = 16'h3F7E;
      8'h96: pos_lut_val = 16'h3F7E;
      8'h97: pos_lut_val = 16'h3F7E;
      8'h98: pos_lut_val = 16'h3F7E;
      8'h99: pos_lut_val = 16'h3F7E;
      8'h9A: pos_lut_val = 16'h3F7E;
      8'h9B: pos_lut_val = 16'h3F7E;
      8'h9C: pos_lut_val = 16'h3F7E;
      8'h9D: pos_lut_val = 16'h3F7E;
      8'h9E: pos_lut_val = 16'h3F7E;
      8'h9F: pos_lut_val = 16'h3F7E;
      8'hA0: pos_lut_val = 16'h3F7E;
      8'hA1: pos_lut_val = 16'h3F7E;
      8'hA2: pos_lut_val = 16'h3F7E;
      8'hA3: pos_lut_val = 16'h3F7E;
      8'hA4: pos_lut_val = 16'h3F7E;
      8'hA5: pos_lut_val = 16'h3F7F;
      8'hA6: pos_lut_val = 16'h3F7F;
      8'hA7: pos_lut_val = 16'h3F7F;
      8'hA8: pos_lut_val = 16'h3F7F;
      8'hA9: pos_lut_val = 16'h3F7F;
      8'hAA: pos_lut_val = 16'h3F7F;
      8'hAB: pos_lut_val = 16'h3F7F;
      8'hAC: pos_lut_val = 16'h3F7F;
      8'hAD: pos_lut_val = 16'h3F7F;
      8'hAE: pos_lut_val = 16'h3F7F;
      8'hAF: pos_lut_val = 16'h3F7F;
      8'hB0: pos_lut_val = 16'h3F7F;
      8'hB1: pos_lut_val = 16'h3F7F;
      8'hB2: pos_lut_val = 16'h3F7F;
      8'hB3: pos_lut_val = 16'h3F7F;
      8'hB4: pos_lut_val = 16'h3F7F;
      8'hB5: pos_lut_val = 16'h3F7F;
      8'hB6: pos_lut_val = 16'h3F7F;
      8'hB7: pos_lut_val = 16'h3F7F;
      8'hB8: pos_lut_val = 16'h3F7F;
      8'hB9: pos_lut_val = 16'h3F7F;
      8'hBA: pos_lut_val = 16'h3F7F;
      8'hBB: pos_lut_val = 16'h3F7F;
      8'hBC: pos_lut_val = 16'h3F7F;
      8'hBD: pos_lut_val = 16'h3F7F;
      8'hBE: pos_lut_val = 16'h3F7F;
      8'hBF: pos_lut_val = 16'h3F7F;
      8'hC0: pos_lut_val = 16'h3F7F;
      8'hC1: pos_lut_val = 16'h3F7F;
      8'hC2: pos_lut_val = 16'h3F7F;
      8'hC3: pos_lut_val = 16'h3F7F;
      8'hC4: pos_lut_val = 16'h3F7F;
      8'hC5: pos_lut_val = 16'h3F7F;
      8'hC6: pos_lut_val = 16'h3F7F;
      8'hC7: pos_lut_val = 16'h3F7F;
      8'hC8: pos_lut_val = 16'h3F80;
      8'hC9: pos_lut_val = 16'h3F80;
      8'hCA: pos_lut_val = 16'h3F80;
      8'hCB: pos_lut_val = 16'h3F80;
      8'hCC: pos_lut_val = 16'h3F80;
      8'hCD: pos_lut_val = 16'h3F80;
      8'hCE: pos_lut_val = 16'h3F80;
      8'hCF: pos_lut_val = 16'h3F80;
      8'hD0: pos_lut_val = 16'h3F80;
      8'hD1: pos_lut_val = 16'h3F80;
      8'hD2: pos_lut_val = 16'h3F80;
      8'hD3: pos_lut_val = 16'h3F80;
      8'hD4: pos_lut_val = 16'h3F80;
      8'hD5: pos_lut_val = 16'h3F80;
      8'hD6: pos_lut_val = 16'h3F80;
      8'hD7: pos_lut_val = 16'h3F80;
      8'hD8: pos_lut_val = 16'h3F80;
      8'hD9: pos_lut_val = 16'h3F80;
      8'hDA: pos_lut_val = 16'h3F80;
      8'hDB: pos_lut_val = 16'h3F80;
      8'hDC: pos_lut_val = 16'h3F80;
      8'hDD: pos_lut_val = 16'h3F80;
      8'hDE: pos_lut_val = 16'h3F80;
      8'hDF: pos_lut_val = 16'h3F80;
      8'hE0: pos_lut_val = 16'h3F80;
      8'hE1: pos_lut_val = 16'h3F80;
      8'hE2: pos_lut_val = 16'h3F80;
      8'hE3: pos_lut_val = 16'h3F80;
      8'hE4: pos_lut_val = 16'h3F80;
      8'hE5: pos_lut_val = 16'h3F80;
      8'hE6: pos_lut_val = 16'h3F80;
      8'hE7: pos_lut_val = 16'h3F80;
      8'hE8: pos_lut_val = 16'h3F80;
      8'hE9: pos_lut_val = 16'h3F80;
      8'hEA: pos_lut_val = 16'h3F80;
      8'hEB: pos_lut_val = 16'h3F80;
      8'hEC: pos_lut_val = 16'h3F80;
      8'hED: pos_lut_val = 16'h3F80;
      8'hEE: pos_lut_val = 16'h3F80;
      8'hEF: pos_lut_val = 16'h3F80;
      8'hF0: pos_lut_val = 16'h3F80;
      8'hF1: pos_lut_val = 16'h3F80;
      8'hF2: pos_lut_val = 16'h3F80;
      8'hF3: pos_lut_val = 16'h3F80;
      8'hF4: pos_lut_val = 16'h3F80;
      8'hF5: pos_lut_val = 16'h3F80;
      8'hF6: pos_lut_val = 16'h3F80;
      8'hF7: pos_lut_val = 16'h3F80;
      8'hF8: pos_lut_val = 16'h3F80;
      8'hF9: pos_lut_val = 16'h3F80;
      8'hFA: pos_lut_val = 16'h3F80;
      8'hFB: pos_lut_val = 16'h3F80;
      8'hFC: pos_lut_val = 16'h3F80;
      8'hFD: pos_lut_val = 16'h3F80;
      8'hFE: pos_lut_val = 16'h3F80;
      8'hFF: pos_lut_val = 16'h3F80;    endcase
  end

  logic [W-1:0] neg_lut_val;
  always_comb begin
    unique case (lut_addr)
      8'h00: neg_lut_val = 16'h3F00;
      8'h01: neg_lut_val = 16'h3EFC;
      8'h02: neg_lut_val = 16'h3EF8;
      8'h03: neg_lut_val = 16'h3EF4;
      8'h04: neg_lut_val = 16'h3EF0;
      8'h05: neg_lut_val = 16'h3EEC;
      8'h06: neg_lut_val = 16'h3EE8;
      8'h07: neg_lut_val = 16'h3EE4;
      8'h08: neg_lut_val = 16'h3EE0;
      8'h09: neg_lut_val = 16'h3EDC;
      8'h0A: neg_lut_val = 16'h3ED8;
      8'h0B: neg_lut_val = 16'h3ED4;
      8'h0C: neg_lut_val = 16'h3ED1;
      8'h0D: neg_lut_val = 16'h3ECD;
      8'h0E: neg_lut_val = 16'h3EC9;
      8'h0F: neg_lut_val = 16'h3EC5;
      8'h10: neg_lut_val = 16'h3EC1;
      8'h11: neg_lut_val = 16'h3EBE;
      8'h12: neg_lut_val = 16'h3EBA;
      8'h13: neg_lut_val = 16'h3EB6;
      8'h14: neg_lut_val = 16'h3EB3;
      8'h15: neg_lut_val = 16'h3EAF;
      8'h16: neg_lut_val = 16'h3EAB;
      8'h17: neg_lut_val = 16'h3EA8;
      8'h18: neg_lut_val = 16'h3EA4;
      8'h19: neg_lut_val = 16'h3EA1;
      8'h1A: neg_lut_val = 16'h3E9D;
      8'h1B: neg_lut_val = 16'h3E9A;
      8'h1C: neg_lut_val = 16'h3E97;
      8'h1D: neg_lut_val = 16'h3E93;
      8'h1E: neg_lut_val = 16'h3E90;
      8'h1F: neg_lut_val = 16'h3E8D;
      8'h20: neg_lut_val = 16'h3E8A;
      8'h21: neg_lut_val = 16'h3E87;
      8'h22: neg_lut_val = 16'h3E83;
      8'h23: neg_lut_val = 16'h3E80;
      8'h24: neg_lut_val = 16'h3E7B;
      8'h25: neg_lut_val = 16'h3E75;
      8'h26: neg_lut_val = 16'h3E6F;
      8'h27: neg_lut_val = 16'h3E6A;
      8'h28: neg_lut_val = 16'h3E64;
      8'h29: neg_lut_val = 16'h3E5F;
      8'h2A: neg_lut_val = 16'h3E59;
      8'h2B: neg_lut_val = 16'h3E54;
      8'h2C: neg_lut_val = 16'h3E4F;
      8'h2D: neg_lut_val = 16'h3E4A;
      8'h2E: neg_lut_val = 16'h3E45;
      8'h2F: neg_lut_val = 16'h3E40;
      8'h30: neg_lut_val = 16'h3E3B;
      8'h31: neg_lut_val = 16'h3E36;
      8'h32: neg_lut_val = 16'h3E31;
      8'h33: neg_lut_val = 16'h3E2D;
      8'h34: neg_lut_val = 16'h3E28;
      8'h35: neg_lut_val = 16'h3E24;
      8'h36: neg_lut_val = 16'h3E20;
      8'h37: neg_lut_val = 16'h3E1C;
      8'h38: neg_lut_val = 16'h3E18;
      8'h39: neg_lut_val = 16'h3E14;
      8'h3A: neg_lut_val = 16'h3E10;
      8'h3B: neg_lut_val = 16'h3E0C;
      8'h3C: neg_lut_val = 16'h3E08;
      8'h3D: neg_lut_val = 16'h3E05;
      8'h3E: neg_lut_val = 16'h3E01;
      8'h3F: neg_lut_val = 16'h3DFB;
      8'h40: neg_lut_val = 16'h3DF4;
      8'h41: neg_lut_val = 16'h3DED;
      8'h42: neg_lut_val = 16'h3DE7;
      8'h43: neg_lut_val = 16'h3DE1;
      8'h44: neg_lut_val = 16'h3DDB;
      8'h45: neg_lut_val = 16'h3DD4;
      8'h46: neg_lut_val = 16'h3DCF;
      8'h47: neg_lut_val = 16'h3DC9;
      8'h48: neg_lut_val = 16'h3DC3;
      8'h49: neg_lut_val = 16'h3DBE;
      8'h4A: neg_lut_val = 16'h3DB9;
      8'h4B: neg_lut_val = 16'h3DB3;
      8'h4C: neg_lut_val = 16'h3DAE;
      8'h4D: neg_lut_val = 16'h3DA9;
      8'h4E: neg_lut_val = 16'h3DA5;
      8'h4F: neg_lut_val = 16'h3DA0;
      8'h50: neg_lut_val = 16'h3D9B;
      8'h51: neg_lut_val = 16'h3D97;
      8'h52: neg_lut_val = 16'h3D93;
      8'h53: neg_lut_val = 16'h3D8E;
      8'h54: neg_lut_val = 16'h3D8A;
      8'h55: neg_lut_val = 16'h3D86;
      8'h56: neg_lut_val = 16'h3D82;
      8'h57: neg_lut_val = 16'h3D7D;
      8'h58: neg_lut_val = 16'h3D76;
      8'h59: neg_lut_val = 16'h3D6F;
      8'h5A: neg_lut_val = 16'h3D68;
      8'h5B: neg_lut_val = 16'h3D61;
      8'h5C: neg_lut_val = 16'h3D5B;
      8'h5D: neg_lut_val = 16'h3D54;
      8'h5E: neg_lut_val = 16'h3D4E;
      8'h5F: neg_lut_val = 16'h3D48;
      8'h60: neg_lut_val = 16'h3D42;
      8'h61: neg_lut_val = 16'h3D3D;
      8'h62: neg_lut_val = 16'h3D37;
      8'h63: neg_lut_val = 16'h3D32;
      8'h64: neg_lut_val = 16'h3D2C;
      8'h65: neg_lut_val = 16'h3D27;
      8'h66: neg_lut_val = 16'h3D22;
      8'h67: neg_lut_val = 16'h3D1E;
      8'h68: neg_lut_val = 16'h3D19;
      8'h69: neg_lut_val = 16'h3D14;
      8'h6A: neg_lut_val = 16'h3D10;
      8'h6B: neg_lut_val = 16'h3D0C;
      8'h6C: neg_lut_val = 16'h3D08;
      8'h6D: neg_lut_val = 16'h3D03;
      8'h6E: neg_lut_val = 16'h3CFF;
      8'h6F: neg_lut_val = 16'h3CF8;
      8'h70: neg_lut_val = 16'h3CF0;
      8'h71: neg_lut_val = 16'h3CE9;
      8'h72: neg_lut_val = 16'h3CE2;
      8'h73: neg_lut_val = 16'h3CDB;
      8'h74: neg_lut_val = 16'h3CD5;
      8'h75: neg_lut_val = 16'h3CCE;
      8'h76: neg_lut_val = 16'h3CC8;
      8'h77: neg_lut_val = 16'h3CC2;
      8'h78: neg_lut_val = 16'h3CBC;
      8'h79: neg_lut_val = 16'h3CB7;
      8'h7A: neg_lut_val = 16'h3CB1;
      8'h7B: neg_lut_val = 16'h3CAC;
      8'h7C: neg_lut_val = 16'h3CA7;
      8'h7D: neg_lut_val = 16'h3CA2;
      8'h7E: neg_lut_val = 16'h3C9D;
      8'h7F: neg_lut_val = 16'h3C98;
      8'h80: neg_lut_val = 16'h3C93;
      8'h81: neg_lut_val = 16'h3C8F;
      8'h82: neg_lut_val = 16'h3C8B;
      8'h83: neg_lut_val = 16'h3C86;
      8'h84: neg_lut_val = 16'h3C82;
      8'h85: neg_lut_val = 16'h3C7D;
      8'h86: neg_lut_val = 16'h3C75;
      8'h87: neg_lut_val = 16'h3C6E;
      8'h88: neg_lut_val = 16'h3C66;
      8'h89: neg_lut_val = 16'h3C5F;
      8'h8A: neg_lut_val = 16'h3C59;
      8'h8B: neg_lut_val = 16'h3C52;
      8'h8C: neg_lut_val = 16'h3C4C;
      8'h8D: neg_lut_val = 16'h3C45;
      8'h8E: neg_lut_val = 16'h3C3F;
      8'h8F: neg_lut_val = 16'h3C3A;
      8'h90: neg_lut_val = 16'h3C34;
      8'h91: neg_lut_val = 16'h3C2F;
      8'h92: neg_lut_val = 16'h3C29;
      8'h93: neg_lut_val = 16'h3C24;
      8'h94: neg_lut_val = 16'h3C1F;
      8'h95: neg_lut_val = 16'h3C1A;
      8'h96: neg_lut_val = 16'h3C16;
      8'h97: neg_lut_val = 16'h3C11;
      8'h98: neg_lut_val = 16'h3C0D;
      8'h99: neg_lut_val = 16'h3C08;
      8'h9A: neg_lut_val = 16'h3C04;
      8'h9B: neg_lut_val = 16'h3C00;
      8'h9C: neg_lut_val = 16'h3BF8;
      8'h9D: neg_lut_val = 16'h3BF1;
      8'h9E: neg_lut_val = 16'h3BE9;
      8'h9F: neg_lut_val = 16'h3BE2;
      8'hA0: neg_lut_val = 16'h3BDB;
      8'hA1: neg_lut_val = 16'h3BD5;
      8'hA2: neg_lut_val = 16'h3BCE;
      8'hA3: neg_lut_val = 16'h3BC8;
      8'hA4: neg_lut_val = 16'h3BC2;
      8'hA5: neg_lut_val = 16'h3BBC;
      8'hA6: neg_lut_val = 16'h3BB6;
      8'hA7: neg_lut_val = 16'h3BB0;
      8'hA8: neg_lut_val = 16'h3BAB;
      8'hA9: neg_lut_val = 16'h3BA6;
      8'hAA: neg_lut_val = 16'h3BA1;
      8'hAB: neg_lut_val = 16'h3B9C;
      8'hAC: neg_lut_val = 16'h3B97;
      8'hAD: neg_lut_val = 16'h3B92;
      8'hAE: neg_lut_val = 16'h3B8E;
      8'hAF: neg_lut_val = 16'h3B8A;
      8'hB0: neg_lut_val = 16'h3B85;
      8'hB1: neg_lut_val = 16'h3B81;
      8'hB2: neg_lut_val = 16'h3B7B;
      8'hB3: neg_lut_val = 16'h3B73;
      8'hB4: neg_lut_val = 16'h3B6C;
      8'hB5: neg_lut_val = 16'h3B64;
      8'hB6: neg_lut_val = 16'h3B5D;
      8'hB7: neg_lut_val = 16'h3B57;
      8'hB8: neg_lut_val = 16'h3B50;
      8'hB9: neg_lut_val = 16'h3B4A;
      8'hBA: neg_lut_val = 16'h3B43;
      8'hBB: neg_lut_val = 16'h3B3D;
      8'hBC: neg_lut_val = 16'h3B38;
      8'hBD: neg_lut_val = 16'h3B32;
      8'hBE: neg_lut_val = 16'h3B2C;
      8'hBF: neg_lut_val = 16'h3B27;
      8'hC0: neg_lut_val = 16'h3B22;
      8'hC1: neg_lut_val = 16'h3B1D;
      8'hC2: neg_lut_val = 16'h3B18;
      8'hC3: neg_lut_val = 16'h3B14;
      8'hC4: neg_lut_val = 16'h3B0F;
      8'hC5: neg_lut_val = 16'h3B0B;
      8'hC6: neg_lut_val = 16'h3B06;
      8'hC7: neg_lut_val = 16'h3B02;
      8'hC8: neg_lut_val = 16'h3AFD;
      8'hC9: neg_lut_val = 16'h3AF5;
      8'hCA: neg_lut_val = 16'h3AED;
      8'hCB: neg_lut_val = 16'h3AE6;
      8'hCC: neg_lut_val = 16'h3ADF;
      8'hCD: neg_lut_val = 16'h3AD8;
      8'hCE: neg_lut_val = 16'h3AD1;
      8'hCF: neg_lut_val = 16'h3ACB;
      8'hD0: neg_lut_val = 16'h3AC5;
      8'hD1: neg_lut_val = 16'h3ABF;
      8'hD2: neg_lut_val = 16'h3AB9;
      8'hD3: neg_lut_val = 16'h3AB3;
      8'hD4: neg_lut_val = 16'h3AAE;
      8'hD5: neg_lut_val = 16'h3AA8;
      8'hD6: neg_lut_val = 16'h3AA3;
      8'hD7: neg_lut_val = 16'h3A9E;
      8'hD8: neg_lut_val = 16'h3A99;
      8'hD9: neg_lut_val = 16'h3A95;
      8'hDA: neg_lut_val = 16'h3A90;
      8'hDB: neg_lut_val = 16'h3A8C;
      8'hDC: neg_lut_val = 16'h3A87;
      8'hDD: neg_lut_val = 16'h3A83;
      8'hDE: neg_lut_val = 16'h3A7E;
      8'hDF: neg_lut_val = 16'h3A76;
      8'hE0: neg_lut_val = 16'h3A6F;
      8'hE1: neg_lut_val = 16'h3A67;
      8'hE2: neg_lut_val = 16'h3A60;
      8'hE3: neg_lut_val = 16'h3A59;
      8'hE4: neg_lut_val = 16'h3A53;
      8'hE5: neg_lut_val = 16'h3A4C;
      8'hE6: neg_lut_val = 16'h3A46;
      8'hE7: neg_lut_val = 16'h3A40;
      8'hE8: neg_lut_val = 16'h3A3A;
      8'hE9: neg_lut_val = 16'h3A34;
      8'hEA: neg_lut_val = 16'h3A2F;
      8'hEB: neg_lut_val = 16'h3A29;
      8'hEC: neg_lut_val = 16'h3A24;
      8'hED: neg_lut_val = 16'h3A1F;
      8'hEE: neg_lut_val = 16'h3A1A;
      8'hEF: neg_lut_val = 16'h3A16;
      8'hF0: neg_lut_val = 16'h3A11;
      8'hF1: neg_lut_val = 16'h3A0C;
      8'hF2: neg_lut_val = 16'h3A08;
      8'hF3: neg_lut_val = 16'h3A04;
      8'hF4: neg_lut_val = 16'h3A00;
      8'hF5: neg_lut_val = 16'h39F8;
      8'hF6: neg_lut_val = 16'h39F0;
      8'hF7: neg_lut_val = 16'h39E9;
      8'hF8: neg_lut_val = 16'h39E2;
      8'hF9: neg_lut_val = 16'h39DB;
      8'hFA: neg_lut_val = 16'h39D4;
      8'hFB: neg_lut_val = 16'h39CE;
      8'hFC: neg_lut_val = 16'h39C7;
      8'hFD: neg_lut_val = 16'h39C1;
      8'hFE: neg_lut_val = 16'h39BB;
      8'hFF: neg_lut_val = 16'h39B5;    endcase
  end

  // --------------------------------------------------------- select result
  logic [W-1:0] final_result;

  always_comb begin
    if (is_nan) begin
      final_result = 16'h3F00;
    end else if (is_inf) begin
      final_result = sign ? 16'h0000 : 16'h3F80;
    end else if (is_zero) begin
      final_result = 16'h3F00;
    end else begin
      final_result = sign ? neg_lut_val : pos_lut_val;
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
