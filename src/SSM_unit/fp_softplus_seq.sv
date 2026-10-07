// Sequential fp32 softplus: y = log(1 + exp(z)).
//
// Used by the SSM unit to discretize the time step:
//   dtp = softplus(dt + dt_bias)
//
// Implementation (one fp_unit transaction at a time):
//   softplus(z) = max(z, 0) + log1p(exp(-|z|))
//   log1p(u)    = 2 * atanh(v),  v = u / (2 + u)
//               = 2 * (v + v^3/3 + v^5/5 + v^7/7 + v^9/9) + O(1e-6)
// The exponential uses fp_exp_seq (accurate polynomial exp).
// After the bf16 rounding applied by the SSM unit (matching the model, which
// stores dtp in bfloat16), the result reproduces torch F.softplus closely.
//
// fp_unit protocol: each operation is started by a 1-cycle in_valid pulse on
// its instance (MUL / ADD / DIV) and its result is latched on out_valid. The
// operation sequence and operand order are unchanged, so the result is
// bit-identical to the previous fixed-schedule version.
//
// The caller pulses `start` with `z`; `done` pulses one cycle with `y` valid
// (held until the next start).

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

  logic [31:0] z_reg, u_reg, v_reg, v2_reg, h_reg, d_reg, prev_y, y_reg;
  logic [3:0]  op;        // 0..13, see the operand table below
  logic        pend;      // the current op is in flight

  typedef enum logic [1:0] { S_IDLE, S_EXP, S_EXPW, S_OP } stage_t;
  stage_t stage;

  // ------------------------------------------------------- EXP (accurate)
  logic [31:0] exp_x, exp_y;
  logic        exp_start, exp_done;

  fp_exp_seq u_exp (
    .clk(clk), .rst_n(rst_n),
    .start(exp_start), .x(exp_x), .y(exp_y), .done(exp_done)
  );

  assign exp_start = (stage == S_EXP);
  assign exp_x     = {1'b1, z_reg[30:0]};   // -|z|

  // ------------------------------------------------------- MUL
  fp_pkg::op_t       mul_mode;
  fp_pkg::rounding_t mul_rm;
  logic [31:0]       mul_a, mul_b, mul_y;
  logic              mul_start, mul_ov;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]        mul_cmp;
  logic [4:0]        mul_flags;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_mul (
    .clk(clk), .rst_n(rst_n),
    .mode(mul_mode), .rm(mul_rm),
    .a(mul_a), .b(mul_b), .c('0),
    .y(mul_y), .cmp(mul_cmp), .flags(mul_flags),
    .in_valid(mul_start), .out_valid(mul_ov)
  );

  // ------------------------------------------------------- ADD
  fp_pkg::op_t       add_mode;
  fp_pkg::rounding_t add_rm;
  logic [31:0]       add_a, add_b, add_y;
  logic              add_start, add_ov;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]        add_cmp;
  logic [4:0]        add_flags;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_add (
    .clk(clk), .rst_n(rst_n),
    .mode(add_mode), .rm(add_rm),
    .a(add_a), .b(add_b), .c('0),
    .y(add_y), .cmp(add_cmp), .flags(add_flags),
    .in_valid(add_start), .out_valid(add_ov)
  );

  // ------------------------------------------------------- DIV
  fp_pkg::op_t       div_mode;
  fp_pkg::rounding_t div_rm;
  logic [31:0]       div_a, div_b, div_y;
  logic              div_start, div_ov;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]        div_cmp;
  logic [4:0]        div_flags;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_div (
    .clk(clk), .rst_n(rst_n),
    .mode(div_mode), .rm(div_rm),
    .a(div_a), .b(div_b), .c('0),
    .y(div_y), .cmp(div_cmp), .flags(div_flags),
    .in_valid(div_start), .out_valid(div_ov)
  );

  assign mul_mode = fp_pkg::OP_MUL;
  assign mul_rm   = fp_pkg::RM_RNE;
  assign add_mode = fp_pkg::OP_ADD;
  assign add_rm   = fp_pkg::RM_RNE;
  assign div_mode = fp_pkg::OP_DIV;
  assign div_rm   = fp_pkg::RM_RNE;

  // ------------------------------------------------------- op inputs
  wire op_mul = (op == 4'd2) || (op == 4'd3) || (op == 4'd5) ||
                (op == 4'd7) || (op == 4'd9) || (op == 4'd10);
  wire op_div = (op == 4'd1);
  wire op_add = !op_mul && !op_div;
  wire op_go  = (stage == S_OP) && !pend;

  always_comb begin
    mul_a = '0; mul_b = '0;
    add_a = '0; add_b = '0;
    div_a = '0; div_b = '0;

    case (op)
      4'd0:  begin add_a = u_reg;  add_b = C2;    end  // d = u + 2
      4'd1:  begin div_a = u_reg;  div_b = d_reg; end  // v = u / d
      4'd2:  begin mul_a = v_reg;  mul_b = v_reg; end  // v2 = v * v
      4'd3:  begin mul_a = C9;     mul_b = v2_reg; end // h = (1/9) * v2
      4'd4:  begin add_a = prev_y; add_b = C7;    end  // h += 1/7
      4'd5:  begin mul_a = prev_y; mul_b = v2_reg; end // h *= v2
      4'd6:  begin add_a = prev_y; add_b = C5;    end  // h += 1/5
      4'd7:  begin mul_a = prev_y; mul_b = v2_reg; end // h *= v2
      4'd8:  begin add_a = prev_y; add_b = C3;    end  // h += 1/3
      4'd9:  begin mul_a = v_reg;  mul_b = v2_reg; end // term = v * v2
      4'd10: begin mul_a = prev_y; mul_b = h_reg;  end // lg = term * h
      4'd11: begin add_a = v_reg;  add_b = prev_y; end // s = v + lg
      4'd12: begin add_a = prev_y; add_b = prev_y; end // 2 * s
      4'd13: begin add_a = z_reg[31] ? 32'h0 : z_reg;  // sp = max(z,0) + ...
                   add_b = prev_y; end
      default: ;
    endcase
  end

  assign mul_start = op_go && op_mul;
  assign add_start = op_go && op_add;
  assign div_start = op_go && op_div;

  wire op_done = pend && (op_mul ? mul_ov : op_div ? div_ov : add_ov);
  wire [31:0] op_y = op_mul ? mul_y : op_div ? div_y : add_y;

  // ------------------------------------------------------- FSM
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      stage  <= S_IDLE;
      op     <= 4'd0;
      pend   <= 1'b0;
      z_reg  <= '0;
      u_reg  <= '0;
      v_reg  <= '0;
      v2_reg <= '0;
      h_reg  <= '0;
      d_reg  <= '0;
      prev_y <= '0;
      y_reg  <= '0;
      done   <= 1'b0;
    end else begin
      done <= 1'b0;
      case (stage)
        S_IDLE: begin
          if (start) begin
            z_reg <= z;
            op    <= 4'd0;
            pend  <= 1'b0;
            stage <= S_EXP;
          end
        end
        S_EXP:  stage <= S_EXPW;   // exp_start pulsed this cycle
        S_EXPW: begin
          if (exp_done) begin
            u_reg <= exp_y;
            stage <= S_OP;
          end
        end
        S_OP: begin
          if (!pend) begin
            pend <= 1'b1;          // in_valid pulsed this cycle; wait
          end else if (op_done) begin
            pend   <= 1'b0;
            prev_y <= op_y;
            case (op)
              4'd0:  d_reg  <= op_y;
              4'd1:  v_reg  <= op_y;
              4'd2:  v2_reg <= op_y;
              4'd8:  h_reg  <= op_y;
              4'd13: begin
                y_reg <= op_y;
                done  <= 1'b1;
                stage <= S_IDLE;
              end
              default: ;
            endcase
            if (op != 4'd13)
              op <= op + 4'd1;
          end
        end
        default: stage <= S_IDLE;
      endcase
    end
  end

  assign y = y_reg;

endmodule
