// Formal abstraction of ssm_unit: accepts one token frame (XN + 2*D_STATE +
// NUM_HEADS beats, tlast on the last), then streams XN binary32 output beats
// one per cycle with constant data. Same port list as the real unit.

module ssm_unit #(
  parameter int NUM_HEADS = 48,
  parameter int HEAD_DIM  = 32,
  parameter int D_STATE   = 128,
  parameter int W_DATA    = 16,
  parameter int H_W       = $clog2(NUM_HEADS),
  parameter int D_W       = $clog2(HEAD_DIM),
  parameter int S_W       = $clog2(D_STATE)
) (
  input  logic              clk,
  input  logic              rst_n,
  input  logic              load_en,
  input  logic [1:0]        load_sel,
  input  logic [H_W-1:0]    load_idx,
  input  logic [W_DATA-1:0] load_wdata,
  input  logic              s_axis_tvalid,
  output logic              s_axis_tready,
  input  logic [W_DATA-1:0] s_axis_tdata,
  input  logic              s_axis_tlast,
  output logic              m_axis_tvalid,
  input  logic              m_axis_tready,
  output logic [31:0]       m_axis_tdata,
  output logic              m_axis_tlast,
  output logic              busy
);

  localparam int XN = NUM_HEADS * HEAD_DIM;
  localparam int TOTAL = XN + 2 * D_STATE + NUM_HEADS;

  logic busy_r;
  logic [$clog2(TOTAL):0] icnt, ocnt;

  assign s_axis_tready = !busy_r;
  assign busy          = busy_r;
  assign m_axis_tvalid = busy_r;
  assign m_axis_tdata  = '0;
  assign m_axis_tlast  = (ocnt == XN - 1);

  always @(posedge clk) begin
    if (!rst_n) begin
      busy_r <= 1'b0;
      icnt   <= '0;
      ocnt   <= '0;
    end else if (!busy_r) begin
      if (s_axis_tvalid) begin
        if (s_axis_tlast || (icnt == TOTAL - 1)) begin
          icnt   <= '0;
          ocnt   <= '0;
          busy_r <= 1'b1;
        end else begin
          icnt <= icnt + 1'b1;
        end
      end
    end else begin
      if (m_axis_tready && (ocnt == XN - 1)) begin
        ocnt   <= '0;
        busy_r <= 1'b0;
      end else if (m_axis_tready) begin
        ocnt <= ocnt + 1'b1;
      end
    end
  end

endmodule
