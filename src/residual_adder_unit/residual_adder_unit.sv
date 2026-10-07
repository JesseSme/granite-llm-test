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
// fp_unit protocol (pipelined library): each fp_unit instance runs one
// operation at a time, started by a 1-cycle in_valid pulse, with y valid when
// out_valid pulses. fp_add_pipe/fp_mul_pipe accept a new start every cycle, so
// this unit keeps its one-element-per-cycle streaming interface by delaying
// the valid/residual pipeline to the measured binary32 latencies
// (result cycle - start cycle):
//   MUL 4 cycles (start during cycle t -> mul_y valid during cycle t+4)
//   ADD 3 cycles
// so an ADD is started exactly when the product of the matching element is on
// mul_y, and the rounded sum is captured three ADD-cycles later:
//
//   cycle t   : valid_in/data_in/residual_in sampled; MUL start
//   cycle t+4 : product on mul_y (bf16-rounded) + delayed residual; ADD start
//   cycle t+7 : sum on add_y (latched into data_out at the end of the cycle)
//   cycle t+8 : data_out / valid_out
//
// The operation order (round the product to bf16 first, then add) is
// unchanged, so the output is bit-identical to the previous 1-cycle-latency
// implementation.

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
  logic              mul_start;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]        mul_cmp;
  logic [4:0]        mul_flags;
  logic              mul_out_valid;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_fp_mul (
    .clk(clk), .rst_n(rst_n),
    .mode(mul_mode), .rm(mul_rm),
    .a(mul_a), .b(mul_b), .c('0),
    .y(mul_y), .cmp(mul_cmp), .flags(mul_flags),
    .in_valid(mul_start), .out_valid(mul_out_valid)
  );

  // ------------------------------------------------------------ FPU: ADD
  fp_pkg::op_t       add_mode;
  fp_pkg::rounding_t add_rm;
  logic [31:0]       add_a, add_b, add_y;
  logic              add_start;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]        add_cmp;
  logic [4:0]        add_flags;
  logic              add_out_valid;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(23)) u_fp_add (
    .clk(clk), .rst_n(rst_n),
    .mode(add_mode), .rm(add_rm),
    .a(add_a), .b(add_b), .c('0),
    .y(add_y), .cmp(add_cmp), .flags(add_flags),
    .in_valid(add_start), .out_valid(add_out_valid)
  );

  // ------------------------------------------------------------ datapath
  logic [15:0] prod_bf16;   // fp32 product rounded once to bf16
  logic [15:0] sum_bf16;    // fp32 sum rounded once to bf16

  // MUL start: one per input element (the multiply pipeline streams).
  assign mul_mode  = fp_pkg::OP_MUL;
  assign mul_rm    = fp_pkg::RM_RNE;
  assign mul_start = valid_in;
  assign mul_a     = {data_in, {(32 - W_DATA) {1'b0}}};
  assign mul_b     = RESIDUAL_MULT_FP32;

  // ADD start coincides with the product on mul_y (MUL latency 4 cycles), so
  // valid_in and residual_in are delayed by the same amount.
  logic        v_d0, v_d1, v_d2, v_d3;
  logic [31:0] res_d0, res_d1, res_d2, res_d3;
  assign add_start = v_d3;
  assign add_mode  = fp_pkg::OP_ADD;
  assign add_rm    = fp_pkg::RM_RNE;
  assign add_a     = res_d3;
  assign add_b     = {prod_bf16, {(32 - W_DATA) {1'b0}}};

  fp32_to_bf16_round u_round_mul (.x(mul_y), .y(prod_bf16));
  fp32_to_bf16_round u_round_add (.x(add_y), .y(sum_bf16));

  // Output stage: add_y is valid three cycles after add_start; latch the
  // rounded sum and raise valid_out at the same cycle as the registered data.
  logic av_d0, av_d1, av_d2;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      v_d0     <= 1'b0;
      v_d1     <= 1'b0;
      v_d2     <= 1'b0;
      v_d3     <= 1'b0;
      res_d0   <= '0;
      res_d1   <= '0;
      res_d2   <= '0;
      res_d3   <= '0;
      av_d0    <= 1'b0;
      av_d1    <= 1'b0;
      av_d2    <= 1'b0;
      data_out <= '0;
      valid_out <= 1'b0;
    end else begin
      v_d0   <= valid_in;
      v_d1   <= v_d0;
      v_d2   <= v_d1;
      v_d3   <= v_d2;
      res_d0 <= {residual_in, {(32 - W_DATA) {1'b0}}};
      res_d1 <= res_d0;
      res_d2 <= res_d1;
      res_d3 <= res_d2;
      av_d0  <= add_start;
      av_d1  <= av_d0;
      av_d2  <= av_d1;
      if (av_d2)
        data_out <= sum_bf16;
      valid_out <= av_d2;
    end
  end

endmodule
