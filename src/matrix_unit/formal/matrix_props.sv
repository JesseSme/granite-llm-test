// Formal properties for matrix_unit (control/AXI-Stream invariants).
//
// The fp_unit datapaths are abstracted (anyseq stubs), so these properties
// verify the stream and FSM protocol independent of FP arithmetic:
//   - input beats accepted only while idle (s_axis_tready == !busy)
//   - output valid only while busy
//   - tlast only on the final output beat
//   - busy rises only after an accepted input beat
//   - exactly OUT_FEATURES output beats per input vector
//
// IN_FEATURES/OUT_FEATURES are reduced for BMC tractability; the FSM counters
// scale with the parameters and the properties are checked for the same
// parameterized logic.
//
// Run: sby -f bmc.sby

module matrix_props #(
  parameter int IN_FEATURES  = 4,
  parameter int OUT_FEATURES = 2
) (
  input logic                    clk,
  input logic                    rst_n,
  input logic                    s_axis_tvalid,
  input logic [15:0]             s_axis_tdata,
  input logic                    s_axis_tlast,
  input logic                    m_axis_tready,
  input logic                    load_en,
  input logic [$clog2(OUT_FEATURES)-1:0] load_out_idx,
  input logic [$clog2(IN_FEATURES)-1:0]  load_in_idx,
  input logic [15:0]             load_wdata,
  input logic                    load_is_bias
);

  logic        s_axis_tready;
  logic        m_axis_tvalid;
  logic [15:0] m_axis_tdata;
  logic        m_axis_tlast;
  logic        busy;

  matrix_unit #(.IN_FEATURES(IN_FEATURES), .OUT_FEATURES(OUT_FEATURES)) u_dut (
    .clk           (clk),
    .rst_n         (rst_n),
    .load_en       (load_en),
    .load_out_idx  (load_out_idx),
    .load_in_idx   (load_in_idx),
    .load_wdata    (load_wdata),
    .load_is_bias  (load_is_bias),
    .s_axis_tvalid (s_axis_tvalid),
    .s_axis_tready (s_axis_tready),
    .s_axis_tdata  (s_axis_tdata),
    .s_axis_tlast  (s_axis_tlast),
    .m_axis_tvalid (m_axis_tvalid),
    .m_axis_tready (m_axis_tready),
    .m_axis_tdata  (m_axis_tdata),
    .m_axis_tlast  (m_axis_tlast),
    .busy          (busy)
  );

  // Output beats completed in the current vector (cleared while idle).
  logic [15:0] out_count;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      out_count <= '0;
    else if (!busy)
      out_count <= '0;
    else if (m_axis_tvalid && m_axis_tready)
      out_count <= out_count + 1'b1;
  end

  // Start from reset so the DUT and the tracking counter are consistent.
  initial assume (!rst_n);

  // Suppresses assertions on the first clock so that $past is well-defined.
  reg f_past_valid = 1'b0;
  always @(posedge clk)
    f_past_valid <= 1'b1;

  always @(posedge clk) begin
    if (f_past_valid && rst_n && $past(rst_n)) begin
      assert (s_axis_tready == !busy);
      assert (!m_axis_tvalid || busy);
      assert (!m_axis_tlast || (out_count == OUT_FEATURES - 1));
      assert (!(busy && !$past(busy)) || $past(s_axis_tvalid && s_axis_tready));
      assert (!(!busy && $past(busy)) || out_count == OUT_FEATURES);
    end
  end

  always @(posedge clk) begin
    cover (s_axis_tvalid && s_axis_tready);
    cover (m_axis_tvalid && m_axis_tready);
    cover (m_axis_tvalid && !m_axis_tready);
    cover (m_axis_tlast && m_axis_tready);
  end

endmodule
