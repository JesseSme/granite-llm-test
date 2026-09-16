// Scaled residual adder hardware unit.
//
// Implements the GraniteMoeHybridDecoderLayer residual connection
// (modeling_granitemoehybrid.py):
//
//   hidden_states = residual + hidden_states * residual_multiplier
//
// where `residual` is the layer input stream, `hidden_states` is the mixer
// (Mamba/Attention) or MLP output stream, and residual_multiplier = 0.246 is a
// fixed constant (config.residual_multiplier), not a learned parameter:
//
//   data_out = residual_in + data_in * 0.246
//
// Element-wise over (B x S x 768) bfloat16 tensors.
//
// Numerics: the Python model multiplies the bf16 tensor by the Python float
// 0.246 with the constant at full precision and rounds each arithmetic result
// once to bf16 (torch bf16 CPU semantics). To match that bit-exactly, the RTL
// widens the bf16 inputs to binary32 (exact: low mantissa bits are zero) and
// uses binary32 fp_units with the fp32 constant 0.246, rounding each operation
// result once back to bf16:
//
//   p     = bf16( data_in_fp32 * fp32(0.246) )
//   out   = bf16( residual_in_fp32 + p_fp32 )
//
// This replaces the previous bf16-constant datapath, which differed from the
// model by 1 bf16 ULP on ~4% of elements.
//
//   cycle N   : capture data_in / residual_in (MUL issue)
//   cycle N+1 : product valid, round to bf16, ADD issue
//   cycle N+2 : sum valid, round to bf16, register output
//   cycle N+3 : data_out / valid_out

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */

module residual_adder_unit #(
  parameter int W_DATA = 16
) (
  input  logic              clk,
  input  logic              rst_n,

  input  logic              valid_in,
  input  logic [W_DATA-1:0] data_in,      // branch output (mixer / MLP)
  input  logic [W_DATA-1:0] residual_in,  // residual stream (added unscaled)

  output logic [W_DATA-1:0] data_out,
  output logic              valid_out
);

  // ------------------------------------------------------------ constants
  // 0.246 in binary32.
  localparam logic [31:0] RESIDUAL_MULT_FP32 = 32'h3E7BE76D;

  // ------------------------------------------------------------ FPU: MUL
  fp_pkg::op_t       mul_mode;
  fp_pkg::rounding_t mul_rm;
  logic [31:0]       mul_a, mul_b, mul_y;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]        mul_cmp;
  logic [4:0]        mul_flags;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_fp_mul (
    .clk(clk), .rst_n(rst_n),
    .mode(mul_mode), .rm(mul_rm),
    .a(mul_a), .b(mul_b), .c('0),
    .y(mul_y), .cmp(mul_cmp), .flags(mul_flags)
  );

  // ------------------------------------------------------------ FPU: ADD
  fp_pkg::op_t       add_mode;
  fp_pkg::rounding_t add_rm;
  logic [31:0]       add_a, add_b, add_y;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]        add_cmp;
  logic [4:0]        add_flags;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_fp_add (
    .clk(clk), .rst_n(rst_n),
    .mode(add_mode), .rm(add_rm),
    .a(add_a), .b(add_b), .c('0),
    .y(add_y), .cmp(add_cmp), .flags(add_flags)
  );

  // ------------------------------------------------------------ datapath
  logic [15:0] prod_bf16;   // fp32 product rounded once to bf16
  logic [15:0] sum_bf16;    // fp32 sum rounded once to bf16
  logic [31:0] residual_d1;
  logic        valid_d1, valid_d2;

  assign mul_mode = fp_pkg::OP_MUL;
  assign mul_rm   = fp_pkg::RM_RNE;
  assign mul_a    = {data_in, {(32 - W_DATA) {1'b0}}};
  assign mul_b    = RESIDUAL_MULT_FP32;

  assign add_mode = fp_pkg::OP_ADD;
  assign add_rm   = fp_pkg::RM_RNE;
  assign add_a    = residual_d1;
  assign add_b    = {prod_bf16, {(32 - W_DATA) {1'b0}}};

  fp32_to_bf16_round u_round_mul (.x(mul_y), .y(prod_bf16));
  fp32_to_bf16_round u_round_add (.x(add_y), .y(sum_bf16));

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      residual_d1 <= '0;
      data_out    <= '0;
      valid_d1    <= 1'b0;
      valid_d2    <= 1'b0;
      valid_out   <= 1'b0;
    end else begin
      residual_d1 <= {residual_in, {(32 - W_DATA) {1'b0}}};
      valid_d1    <= valid_in;
      valid_d2    <= valid_d1;
      data_out    <= sum_bf16;
      valid_out   <= valid_d2;
    end
  end

endmodule
