// MLP unit — the model-facing wrapper for the SwiGLU MLP.
//
// GraniteMoeHybridMLP (all 32 decoder layers) is exactly the SwiGLU block, so
// this unit is a thin wrapper around swiglu_unit: same ports, same weight load
// mapping (load_sel 0 = fused gate+up projection, 1 = down projection) and the
// same AXI-Stream framing (HIDDEN bf16 beats per frame in and out). It exists
// so the layer pipeline instantiates the model's class name while the datapath
// lives in SwiGLU_unit/.
//
// See swiglu_unit.sv for the numeric details (bf16 projections with fp32
// accumulation, silu(gate) * up with a single bf16 rounding, bf16 down
// projection).

/* verilator lint_off SYNCASYNCNET */

module mlp_unit #(
  parameter int HIDDEN = 768,
  parameter int INTER  = 2048,
  parameter int W_DATA = 16,
  parameter int LO_W   = $clog2(((2 * INTER) > HIDDEN) ? (2 * INTER) : HIDDEN),
  parameter int LI_W   = $clog2((HIDDEN > INTER) ? HIDDEN : INTER)
) (
  input  logic              clk,
  input  logic              rst_n,

  input  logic              load_en,
  input  logic              load_sel,
  input  logic [LO_W-1:0]   load_out_idx,
  input  logic [LI_W-1:0]   load_in_idx,
  input  logic [W_DATA-1:0] load_wdata,

  input  logic              s_axis_tvalid,
  output logic              s_axis_tready,
  input  logic [W_DATA-1:0] s_axis_tdata,
  input  logic              s_axis_tlast,

  output logic              m_axis_tvalid,
  input  logic              m_axis_tready,
  output logic [W_DATA-1:0] m_axis_tdata,
  output logic              m_axis_tlast,

  output logic              busy
);

  swiglu_unit #(
    .HIDDEN(HIDDEN), .INTER(INTER), .W_DATA(W_DATA)
  ) u_swiglu (
    .clk(clk), .rst_n(rst_n),
    .load_en(load_en), .load_sel(load_sel),
    .load_out_idx(load_out_idx), .load_in_idx(load_in_idx),
    .load_wdata(load_wdata),
    .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready),
    .s_axis_tdata(s_axis_tdata), .s_axis_tlast(s_axis_tlast),
    .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready),
    .m_axis_tdata(m_axis_tdata), .m_axis_tlast(m_axis_tlast),
    .busy(busy)
  );

endmodule
