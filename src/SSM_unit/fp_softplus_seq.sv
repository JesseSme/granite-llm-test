// Sequential fp32 softplus: y = log(1 + exp(z)).
//
// Used by the SSM unit to discretize the time step:
//   dtp = softplus(dt + dt_bias)
//
// Implementation (one operation per cycle, explicit result registers):
//   softplus(z) = max(z, 0) + log1p(exp(-|z|))
//   log1p(u)    = 2 * atanh(v),  v = u / (2 + u)
//               = 2 * (v + v^3/3 + v^5/5 + v^7/7 + v^9/9) + O(1e-6)
// The exponential uses fp_exp_seq (accurate polynomial exp).
// After the bf16 rounding applied by the SSM unit (matching the model, which
// stores dtp in bfloat16), the result reproduces torch F.softplus closely.
//
// The caller pulses `start` with `z`; `done` pulses one cycle with `y` valid
// (held until the next start). ~38 cycles per evaluation.

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */

module fp_softplus_seq (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  input  logic [31:0] z,
  output logic [31:0] y,
  output logic        done
);

  // fp32 constants.
  localparam logic [31:0] C2 = 32'h40000000;  // 2.0
  localparam logic [31:0] C9 = 32'h3DE38E39;  // 1/9
  localparam logic [31:0] C7 = 32'h3E124925;  // 1/7
  localparam logic [31:0] C5 = 32'h3E4CCCCD;  // 1/5
  localparam logic [31:0] C3 = 32'h3EAAAAAB;  // 1/3

  logic [31:0] z_reg;
  logic [5:0]  step;

  logic [31:0] u_reg, v_reg, v2_reg, h_reg;

  // ------------------------------------------------------- EXP (accurate)
  logic [31:0] exp_x, exp_y;
  logic        exp_start, exp_done;

  fp_exp_seq u_exp (
    .clk(clk), .rst_n(rst_n),
    .start(exp_start), .x(exp_x), .y(exp_y), .done(exp_done)
  );

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

  // ------------------------------------------------------- DIV
  fp_pkg::op_t       div_mode;
  fp_pkg::rounding_t div_rm;
  logic [31:0]       div_a, div_b, div_y;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]        div_cmp;
  logic [4:0]        div_flags;
  logic              unused_out_valid_div;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_div (
    .clk(clk), .rst_n(rst_n),
    .mode(div_mode), .rm(div_rm),
    .a(div_a), .b(div_b), .c('0),
    .y(div_y), .cmp(div_cmp), .flags(div_flags),
    .in_valid(1'b1), .out_valid(unused_out_valid_div)
  );

  assign mul_mode = fp_pkg::OP_MUL;
  assign mul_rm   = fp_pkg::RM_RNE;
  assign add_mode = fp_pkg::OP_ADD;
  assign add_rm   = fp_pkg::RM_RNE;
  assign div_mode = fp_pkg::OP_DIV;
  assign div_rm   = fp_pkg::RM_RNE;

  assign exp_start = (step == 6'd1);
  assign exp_x     = {1'b1, z_reg[30:0]};   // -|z|

  // ------------------------------------------------------- op inputs
  always_comb begin
    mul_a = '0;
    mul_b = '0;
    add_a = '0;
    add_b = '0;
    div_a = '0;
    div_b = '0;

    case (step)
      6'd3: begin add_a = u_reg; add_b = C2; end        // d = u + 2
      6'd4: begin div_a = u_reg; div_b = add_y; end     // v = u / d
      6'd5: begin mul_a = div_y; mul_b = div_y; end     // v2 = v * v
      6'd6: begin mul_a = C9;   mul_b = mul_y; end      // h = (1/9) * v2
      6'd7: begin add_a = mul_y; add_b = C7; end        // h += 1/7
      6'd8: begin mul_a = add_y; mul_b = v2_reg; end    // h *= v2
      6'd9: begin add_a = mul_y; add_b = C5; end        // h += 1/5
      6'd10: begin mul_a = add_y; mul_b = v2_reg; end   // h *= v2
      6'd11: begin add_a = mul_y; add_b = C3; end       // h += 1/3
      6'd12: begin mul_a = v_reg; mul_b = v2_reg; end   // term = v * v2
      6'd13: begin mul_a = mul_y; mul_b = h_reg; end    // lg = term * h
      6'd14: begin add_a = v_reg; add_b = mul_y; end    // s = v + lg
      6'd15: begin add_a = add_y; add_b = add_y; end    // 2 * s
      6'd16: begin add_a = z_reg[31] ? 32'h0 : z_reg;
                   add_b = add_y; end                   // sp = max(z,0) + ...
      default: ;
    endcase
  end

  // ------------------------------------------------------- FSM
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      step  <= 6'd0;
      z_reg <= '0;
      u_reg <= '0;
      v_reg <= '0;
      v2_reg <= '0;
      h_reg <= '0;
      y     <= '0;
      done  <= 1'b0;
    end else begin
      done <= 1'b0;
      if (step == 6'd0) begin
        if (start) begin
          z_reg <= z;
          step  <= 6'd1;
        end
      end else begin
        case (step)
          6'd1: step <= 6'd2;                 // exp started combinationally
          6'd2: begin                         // wait for the accurate exp
            if (exp_done) begin
              u_reg <= exp_y;
              step  <= 6'd3;
            end
          end
          6'd5:  v_reg  <= div_y;             // v
          6'd6:  v2_reg <= mul_y;             // v2
          6'd7:  h_reg  <= mul_y;             // (1/9)*v2
          6'd8:  h_reg  <= add_y;             // + 1/7
          6'd9:  h_reg  <= mul_y;             // * v2
          6'd10: h_reg  <= add_y;             // + 1/5
          6'd11: h_reg  <= mul_y;             // * v2
          6'd12: h_reg  <= add_y;             // + 1/3
          6'd17: begin
            y    <= add_y;                    // sp
            done <= 1'b1;
          end
          default: ;
        endcase
        if (step != 6'd2)                     // hold at the exp wait state
          step <= (step == 6'd17) ? 6'd0 : (step + 6'd1);
      end
    end
  end

endmodule
