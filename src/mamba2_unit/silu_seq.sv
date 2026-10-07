// Accurate streaming SiLU: y = x * sigmoid(x), binary32 result.
//
// sigmoid(x) = 1/(1+exp(-x)) is evaluated in binary32 with the fp_exp_seq
// polynomial exponential (the LUT-based silu_unit is only ~0.5% accurate,
// which is too coarse for the Mamba gate branch and the conv activation).
//
// fp_unit protocol: each operation is started by a 1-cycle in_valid pulse on
// its instance and latched on out_valid (MUL 4, ADD 3, DIV 17 cycles); the
// exponential is the sequential fp_exp_seq. Operation order/operands are
// unchanged, so the binary32 result is bit-identical.
//
// Handshake: `valid_i`/`data_i` (bf16) are accepted when `ready_o` is high
// (the unit is idle), `valid_o` pulses for one cycle with the binary32 result
// on `data_o`. ~90 cycles per element (the exponential and the iterative
// divider are sequential), so upstream sources must be backpressure-aware.

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off SYNCASYNCNET */

module silu_seq (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        valid_i,
  output logic        ready_o,
  input  logic [15:0] data_i,
  output logic        valid_o,
  output logic [31:0] data_o
);

  localparam logic [31:0] C_ONE = 32'h3F800000;

  /* verilator lint_off PINCONNECTEMPTY */
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0] unused_cmp;
  logic [4:0] unused_flags;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_add (
    .clk(clk), .rst_n(rst_n),
    .mode(fp_pkg::OP_ADD), .rm(fp_pkg::RM_RNE),
    .a(add_a), .b(add_b), .c('0),
    .y(add_y), .cmp(unused_cmp), .flags(unused_flags),
    .in_valid(add_start), .out_valid(add_ov)
  );

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_div (
    .clk(clk), .rst_n(rst_n),
    .mode(fp_pkg::OP_DIV), .rm(fp_pkg::RM_RNE),
    .a(div_a), .b(div_b), .c(32'h0),
    .y(div_y), .cmp(/*unused*/), .flags(/*unused*/),
    .in_valid(div_start), .out_valid(div_ov)
  );

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_mul (
    .clk(clk), .rst_n(rst_n),
    .mode(fp_pkg::OP_MUL), .rm(fp_pkg::RM_RNE),
    .a(mul_a), .b(mul_b), .c('0),
    .y(mul_y), .cmp(/*unused*/), .flags(/*unused*/),
    .in_valid(mul_start), .out_valid(mul_ov)
  );
  /* verilator lint_on PINCONNECTEMPTY */

  logic [31:0] exp_x, exp_y;
  logic        exp_start, exp_done;

  fp_exp_seq u_exp (
    .clk(clk), .rst_n(rst_n),
    .start(exp_start), .x(exp_x), .y(exp_y), .done(exp_done)
  );

  logic [31:0] add_a, add_b, add_y;
  logic        add_start, add_ov;
  logic [31:0] div_a, div_b, div_y;
  logic        div_start, div_ov;
  logic [31:0] mul_a, mul_b, mul_y;
  logic        mul_start, mul_ov;

  logic [31:0] x_reg, e_reg, sig_reg;

  typedef enum logic [3:0] {
    IDLE, EXP, EXP_W,
    ADD_REQ, ADD_W,
    DIV_REQ, DIV_W,
    MUL_REQ, MUL_W,
    OUT
  } state_t;

  state_t state;

  assign ready_o   = (state == IDLE);
  assign exp_start = (state == EXP);
  assign exp_x     = {~x_reg[31], x_reg[30:0]};   // -x

  always_comb begin
    add_a = '0; add_b = '0;
    div_a = '0; div_b = '0;
    mul_a = '0; mul_b = '0;

    case (state)
      ADD_REQ: begin add_a = C_ONE; add_b = exp_y; end
      DIV_REQ: begin div_a = C_ONE; div_b = e_reg; end
      MUL_REQ: begin mul_a = x_reg; mul_b = sig_reg; end
      default: ;
    endcase
  end

  assign add_start = (state == ADD_REQ);
  assign div_start = (state == DIV_REQ);
  assign mul_start = (state == MUL_REQ);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state   <= IDLE;
      valid_o <= 1'b0;
      x_reg   <= '0;
      e_reg   <= '0;
      sig_reg <= '0;
      data_o  <= '0;
    end else begin
      valid_o <= 1'b0;
      case (state)
        IDLE: begin
          if (valid_i) begin
            x_reg <= {data_i, 16'b0};
            state <= EXP;
          end
        end
        EXP:     state <= EXP_W;
        EXP_W:   if (exp_done) state <= ADD_REQ;
        ADD_REQ: state <= ADD_W;
        ADD_W: begin
          if (add_ov) begin e_reg <= add_y; state <= DIV_REQ; end
        end
        DIV_REQ: state <= DIV_W;
        DIV_W: begin
          if (div_ov) begin sig_reg <= div_y; state <= MUL_REQ; end
        end
        MUL_REQ: state <= MUL_W;
        MUL_W: begin
          if (mul_ov) begin
            data_o  <= mul_y;
            valid_o <= 1'b1;
            state   <= OUT;
          end
        end
        OUT:   state <= IDLE;
        default: state <= IDLE;
      endcase
    end
  end

endmodule
