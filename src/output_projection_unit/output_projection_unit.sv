// Output projection (LM head) of Granite 4.0-H-350M.
//
// Implements `GraniteMoeHybridForCausalLM.forward`'s final step:
//   logits = self.lm_head(hidden_states)          # nn.Linear(768, 100352, bias=False)
//   logits = logits / self.config.logits_scaling  # = 3
// The lm_head weight is tied to the input embedding matrix (same 100352 x 768
// bfloat16 table as embedding_lookup_unit).
//
// Numerics: the linear projection uses matrix_unit (bfloat16 weights,
// binary32 sequential accumulation, one bfloat16 rounding - matching torch
// F.linear up to ATen's blocked-GEMM accumulation order). The scaling is a
// binary32 division by 3.0 followed by one bfloat16 rounding, matching
// torch's bf16 `tensor / 3` (computed in fp32, rounded once).
//
// Architecture: matrix_unit (HIDDEN -> VOCAB) feeding a 1-entry scale stage
// built from an fp32 fp_unit (OP_DIV) and fp32_to_bf16_round. The scale stage
// holds one result under downstream backpressure and only accepts the next
// projected beat when free.
//
// AXI-Stream: input HIDDEN beats (tlast on the last), output VOCAB beats
// (tlast on the last). Weights are loaded through the matrix_unit load port
// (one element per cycle, no bias).

module output_projection_unit #(
  parameter int HIDDEN = 768,
  parameter int VOCAB  = 100352,
  parameter int W_DATA = 16,
  parameter int IN_W   = $clog2(HIDDEN),
  parameter int OUT_W  = $clog2(VOCAB)
) (
  input  logic                  clk,
  input  logic                  rst_n,

  // Weight load interface (pass-through to the projection, no bias)
  input  logic                  load_en,
  input  logic [OUT_W-1:0]      load_out_idx,
  input  logic [IN_W-1:0]       load_in_idx,
  input  logic [W_DATA-1:0]     load_wdata,

  // AXI-Stream input (hidden state vector: HIDDEN bf16 beats)
  input  logic                  s_axis_tvalid,
  output logic                  s_axis_tready,
  input  logic [W_DATA-1:0]     s_axis_tdata,
  input  logic                  s_axis_tlast,

  // AXI-Stream output (logits: VOCAB bf16 beats)
  output logic                  m_axis_tvalid,
  input  logic                  m_axis_tready,
  output logic [W_DATA-1:0]     m_axis_tdata,
  output logic                  m_axis_tlast,

  output logic                  busy
);
  localparam int W_MANT32 = 23;
  localparam int W_FP32   = 1 + 8 + W_MANT32;
  localparam logic [W_FP32-1:0] C_LOGITS_SCALING = 32'h40400000;  // 3.0

  logic              proj_m_tvalid, proj_m_tready, proj_m_tlast;
  logic [W_DATA-1:0] proj_m_tdata;
  logic              proj_busy;

  matrix_unit #(
    .IN_FEATURES (HIDDEN),
    .OUT_FEATURES(VOCAB)
  ) u_proj (
    .clk          (clk),
    .rst_n        (rst_n),
    .load_en      (load_en),
    .load_out_idx (load_out_idx),
    .load_in_idx  (load_in_idx),
    .load_wdata   (load_wdata),
    .load_is_bias (1'b0),
    .s_axis_tvalid(s_axis_tvalid),
    .s_axis_tready(s_axis_tready),
    .s_axis_tdata (s_axis_tdata),
    .s_axis_tlast (s_axis_tlast),
    .m_axis_tvalid(proj_m_tvalid),
    .m_axis_tready(proj_m_tready),
    .m_axis_tdata (proj_m_tdata),
    .m_axis_tlast (proj_m_tlast),
    .busy         (proj_busy)
  );

  // ----------------------------------------------------------------
  // Scale stage: logits / 3.0 in binary32, one bfloat16 rounding
  // ----------------------------------------------------------------
  logic              sc_valid;      // divide in flight
  logic              sc_out_valid;  // result ready, held for downstream
  logic [W_DATA-1:0] sc_tdata;
  logic              sc_tlast;
  logic [W_FP32-1:0] div_a, div_y;
  logic [W_DATA-1:0] div_bf;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]        div_cmp;
  logic [4:0]        div_flags;
  logic              unused_out_valid_div;
  /* verilator lint_on UNUSEDSIGNAL */

  fp_unit #(.W_EXP(8), .W_MANT(W_MANT32)) u_div (
    .clk      (clk),
    .rst_n    (rst_n),
    .in_valid (1'b1),
    .mode     (fp_pkg::OP_DIV),
    .rm       (fp_pkg::RM_RNE),
    .a        (div_a),
    .b        (C_LOGITS_SCALING),
    .c        (W_FP32'(0)),
    .y        (div_y),
    .cmp      (div_cmp),
    .flags    (div_flags),
    .out_valid(unused_out_valid_div)
  );

  fp32_to_bf16_round u_round (.x(div_y), .y(div_bf));

  function automatic logic [W_FP32-1:0] bf16_to_fp32(input logic [W_DATA-1:0] v);
    bf16_to_fp32 = {v, {(W_FP32 - W_DATA){1'b0}}};
  endfunction

  assign div_a         = bf16_to_fp32(proj_m_tdata);
  assign proj_m_tready = !sc_valid && !sc_out_valid;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      sc_valid     <= 1'b0;
      sc_out_valid <= 1'b0;
      sc_tdata     <= '0;
      sc_tlast     <= 1'b0;
    end else begin
      if (proj_m_tvalid && proj_m_tready) begin
        sc_valid <= 1'b1;
        sc_tlast <= proj_m_tlast;
      end else begin
        sc_valid <= 1'b0;
      end

      if (sc_valid) begin
        sc_out_valid <= 1'b1;
        sc_tdata     <= div_bf;
      end else if (m_axis_tready) begin
        sc_out_valid <= 1'b0;
      end
    end
  end

  assign m_axis_tvalid = sc_out_valid;
  assign m_axis_tdata  = sc_tdata;
  assign m_axis_tlast  = sc_tlast;
  assign busy          = proj_busy || sc_valid || sc_out_valid;

endmodule
