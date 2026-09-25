// Sequential accurate fp32 exp: y = exp(x), x clamped to [-16, 16].
//
// Used by the SSM unit for A = -exp(A_log) and the per-step decay
// dA = exp(A * dtp). The shared `fp_exp` LUT units are only ~8-bit accurate
// (max relative error ~20% measured), which is not sufficient inside a
// recurrent state update, so this unit uses range reduction plus a degree-6
// polynomial and scaling by squaring:
//
//   exp(x) = exp(x/32)^32,   exp(r) ~= 1 + r*(1 + r*(1/2 + r*(1/6 +
//                                        r*(1/24 + r*(1/120 + r/720)))))
// with |x/32| <= 0.5, giving ~1e-6 relative error.
//
// The caller pulses `start` with `x`; `done` pulses one cycle with `y` valid
// (held until the next start). ~21 cycles per evaluation. Arguments below
// -16 are clamped (exp(-16) = 1.1e-7, effectively zero for the SSM decay).

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */

module fp_exp_seq (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  input  logic [31:0] x,
  output logic [31:0] y,
  output logic        done
);

  localparam logic [31:0] C_SCALE = 32'h3D000000;  // 2^-5
  localparam logic [31:0] C_6     = 32'h3AB60B61;  // 1/720
  localparam logic [31:0] C_5     = 32'h3C088889;  // 1/120
  localparam logic [31:0] C_4     = 32'h3D2AAAAB;  // 1/24
  localparam logic [31:0] C_3     = 32'h3E2AAAAB;  // 1/6
  localparam logic [31:0] C_2     = 32'h3F000000;  // 1/2
  localparam logic [31:0] C_1     = 32'h3F800000;  // 1.0
  localparam logic [31:0] C_MAX   = 32'h41800000;  // 16.0

  logic [31:0] x_reg, r_reg, h_reg;
  logic [4:0]  step;
  logic [31:0] clamped_mag;

  // Clamp |x| to 16.0, preserving the sign.
  assign clamped_mag = (x[30:0] > C_MAX[30:0]) ? {x[31], C_MAX[30:0]} : x;

  // ------------------------------------------------------- MUL
  fp_pkg::op_t       mul_mode;
  fp_pkg::rounding_t mul_rm;
  logic [31:0]       mul_a, mul_b, mul_y;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]        mul_cmp;
  logic [4:0]        mul_flags;
  logic              unused_out_valid_mul;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_mul (
    .clk(clk), .rst_n(rst_n),
    .mode(mul_mode), .rm(mul_rm),
    .a(mul_a), .b(mul_b), .c('0),
    .y(mul_y), .cmp(mul_cmp), .flags(mul_flags),
    .in_valid(1'b1), .out_valid(unused_out_valid_mul)
  );

  // ------------------------------------------------------- ADD
  fp_pkg::op_t       add_mode;
  fp_pkg::rounding_t add_rm;
  logic [31:0]       add_a, add_b, add_y;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]        add_cmp;
  logic [4:0]        add_flags;
  logic              unused_out_valid_add;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_add (
    .clk(clk), .rst_n(rst_n),
    .mode(add_mode), .rm(add_rm),
    .a(add_a), .b(add_b), .c('0),
    .y(add_y), .cmp(add_cmp), .flags(add_flags),
    .in_valid(1'b1), .out_valid(unused_out_valid_add)
  );

  assign mul_mode = fp_pkg::OP_MUL;
  assign mul_rm   = fp_pkg::RM_RNE;
  assign add_mode = fp_pkg::OP_ADD;
  assign add_rm   = fp_pkg::RM_RNE;

  // ------------------------------------------------------- op inputs
  always_comb begin
    mul_a = '0;
    mul_b = '0;
    add_a = '0;
    add_b = '0;

    case (step)
      5'd1:  begin mul_a = x_reg;  mul_b = C_SCALE; end  // r = x/32
      5'd3:  begin mul_a = h_reg;  mul_b = r_reg;   end  // h *= r
      5'd4:  begin add_a = mul_y;  add_b = C_5;     end  // + 1/120
      5'd5:  begin mul_a = add_y;  mul_b = r_reg;   end
      5'd6:  begin add_a = mul_y;  add_b = C_4;     end  // + 1/24
      5'd7:  begin mul_a = add_y;  mul_b = r_reg;   end
      5'd8:  begin add_a = mul_y;  add_b = C_3;     end  // + 1/6
      5'd9:  begin mul_a = add_y;  mul_b = r_reg;   end
      5'd10: begin add_a = mul_y;  add_b = C_2;     end  // + 1/2
      5'd11: begin mul_a = add_y;  mul_b = r_reg;   end
      5'd12: begin add_a = mul_y;  add_b = C_1;     end  // + 1
      5'd13: begin mul_a = r_reg;  mul_b = add_y;   end  // r*h
      5'd14: begin add_a = C_1;    add_b = mul_y;   end  // p = 1 + r*h
      5'd15: begin mul_a = add_y;  mul_b = add_y;   end  // squarings
      5'd16: begin mul_a = mul_y;  mul_b = mul_y;   end
      5'd17: begin mul_a = mul_y;  mul_b = mul_y;   end
      5'd18: begin mul_a = mul_y;  mul_b = mul_y;   end
      5'd19: begin mul_a = mul_y;  mul_b = mul_y;   end
      default: ;
    endcase
  end

  // ------------------------------------------------------- FSM
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      step  <= 5'd0;
      x_reg <= '0;
      r_reg <= '0;
      h_reg <= '0;
      y     <= '0;
      done  <= 1'b0;
    end else begin
      done <= 1'b0;
      if (step == 5'd0) begin
        if (start) begin
          x_reg <= clamped_mag;
          step  <= 5'd1;
        end
      end else begin
        case (step)
          5'd2: begin
            r_reg <= mul_y;           // r = x/32
            h_reg <= C_6;             // Horner starts at 1/720
          end
          5'd5:  h_reg <= add_y;      // h = h*r + 1/120
          5'd7:  h_reg <= add_y;      // + 1/24
          5'd9:  h_reg <= add_y;      // + 1/6
          5'd11: h_reg <= add_y;      // + 1/2
          5'd13: h_reg <= add_y;      // + 1
          5'd20: begin
            y    <= mul_y;            // exp(x) = poly^32
            done <= 1'b1;
          end
          default: ;
        endcase
        step <= (step == 5'd20) ? 5'd0 : (step + 5'd1);
      end
    end
  end

endmodule
