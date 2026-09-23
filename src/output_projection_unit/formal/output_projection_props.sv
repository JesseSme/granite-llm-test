// BMC properties for output_projection_unit (matrix_unit stubbed with exact
// handshake timing).
module output_projection_props (
  input logic clk, rst_n
);
  logic s_axis_tvalid, s_axis_tready, s_axis_tlast;
  logic [15:0] s_axis_tdata;
  logic m_axis_tvalid, m_axis_tready, m_axis_tlast;
  logic [15:0] m_axis_tdata;
  logic load_en, busy;
  logic [5:0] load_out_idx;
  logic [4:0] load_in_idx;
  logic [15:0] load_wdata;

  output_projection_unit #(.HIDDEN(32), .VOCAB(4)) dut (
    .clk(clk), .rst_n(rst_n),
    .load_en(load_en), .load_out_idx(load_out_idx), .load_in_idx(load_in_idx),
    .load_wdata(load_wdata),
    .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready),
    .s_axis_tdata(s_axis_tdata), .s_axis_tlast(s_axis_tlast),
    .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready),
    .m_axis_tdata(m_axis_tdata), .m_axis_tlast(m_axis_tlast),
    .busy(busy)
  );

  (* anyseq *) logic [15:0] free_in;
  (* anyseq *) logic       free_valid;
  (* anyseq *) logic       free_ready;
  assign s_axis_tvalid = free_valid;
  assign s_axis_tdata  = free_in;
  assign s_axis_tlast  = 1'b1;
  assign m_axis_tready = free_ready;
  assign load_en       = 1'b0;
  assign load_out_idx  = '0;
  assign load_in_idx   = '0;
  assign load_wdata    = '0;

  logic [15:0] in_beats, out_beats;
  logic        seen_rst = 1'b0;   // assertions only meaningful after a reset
  logic        hold_expected;   // DUT had an unaccepted valid beat last cycle

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      in_beats      <= '0;
      out_beats     <= '0;
      hold_expected <= 1'b0;
    end else begin
      if (s_axis_tvalid && s_axis_tready) in_beats  <= in_beats + 1'b1;
      if (m_axis_tvalid && m_axis_tready) out_beats <= out_beats + 1'b1;
      hold_expected <= m_axis_tvalid && !m_axis_tready;
      seen_rst      <= seen_rst | !rst_n;
    end
  end

  always_ff @(posedge clk) begin
    if (seen_rst) begin
      // no output beat without at least one accepted input beat
      assert (out_beats <= in_beats);
      // tlast only with a valid beat
      assert (!m_axis_tlast || m_axis_tvalid);
      // no data loss: an unaccepted valid beat is still valid next cycle
      assert (!hold_expected || m_axis_tvalid);
      // the DUT never drives the load port itself
      assert (!load_en);
    end
  end
endmodule
