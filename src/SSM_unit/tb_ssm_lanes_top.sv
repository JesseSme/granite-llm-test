// Lane-equivalence harness: the same SSM topology built with LANES=1 and
// LANES=4, driven from shared ports. The cocotb TB streams identical weights
// and input frames into both instances and asserts that every output beat is
// bit-identical (the lane optimization must not change any arithmetic order).
//
// Small odd geometry (NUM_HEADS=3, HEAD_DIM=5, D_STATE=6) exercises both a
// full 4-lane block and a masked partial block (HEAD_DIM % LANES != 0).

/* verilator lint_off SYNCASYNCNET */

module tb_ssm_lanes_top #(
  parameter int NUM_HEADS = 3,
  parameter int HEAD_DIM  = 5,
  parameter int D_STATE   = 6,
  parameter int H_W       = $clog2(NUM_HEADS)
) (
  input  logic              clk,
  input  logic              rst_n,
  input  logic              load_en,
  input  logic [1:0]        load_sel,
  input  logic [H_W-1:0]    load_idx,
  input  logic [15:0]       load_wdata,

  input  logic              s_axis_tvalid,
  input  logic [15:0]       s_axis_tdata,
  input  logic              s_axis_tlast,
  output logic              s0_axis_tready,
  output logic              s1_axis_tready,
  output logic              busy0,
  output logic              busy1,

  output logic              m0_axis_tvalid,
  input  logic              m0_axis_tready,
  output logic [31:0]       m0_axis_tdata,
  output logic              m0_axis_tlast,

  output logic              m1_axis_tvalid,
  input  logic              m1_axis_tready,
  output logic [31:0]       m1_axis_tdata,
  output logic              m1_axis_tlast
);

  ssm_unit #(
    .NUM_HEADS(NUM_HEADS), .HEAD_DIM(HEAD_DIM), .D_STATE(D_STATE), .LANES(1)
  ) u_lanes1 (
    .clk(clk), .rst_n(rst_n),
    .load_en(load_en), .load_sel(load_sel), .load_idx(load_idx),
    .load_wdata(load_wdata),
    .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s0_axis_tready),
    .s_axis_tdata(s_axis_tdata), .s_axis_tlast(s_axis_tlast),
    .m_axis_tvalid(m0_axis_tvalid), .m_axis_tready(m0_axis_tready),
    .m_axis_tdata(m0_axis_tdata), .m_axis_tlast(m0_axis_tlast),
    .busy(busy0)
  );

  ssm_unit #(
    .NUM_HEADS(NUM_HEADS), .HEAD_DIM(HEAD_DIM), .D_STATE(D_STATE), .LANES(4)
  ) u_lanes4 (
    .clk(clk), .rst_n(rst_n),
    .load_en(load_en), .load_sel(load_sel), .load_idx(load_idx),
    .load_wdata(load_wdata),
    .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s1_axis_tready),
    .s_axis_tdata(s_axis_tdata), .s_axis_tlast(s_axis_tlast),
    .m_axis_tvalid(m1_axis_tvalid), .m_axis_tready(m1_axis_tready),
    .m_axis_tdata(m1_axis_tdata), .m_axis_tlast(m1_axis_tlast),
    .busy(busy1)
  );

endmodule
