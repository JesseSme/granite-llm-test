// Formal properties for ssm_unit (AXI-Stream framing / control invariants).
//
// The FP datapaths (fp_unit, fp_exp_seq, fp_softplus_seq) are abstracted with
// anyseq stubs, so the properties below are independent of numeric results:
//   P1  s_axis_tready is high exactly while the unit is idle (!busy)
//   P2  output beats are only produced while busy
//   P3  tlast is asserted exactly on the last (NUM_HEADS*HEAD_DIM-th) beat of
//       the current activation
//   P4  at most XN output beats are accepted per activation
//   P5  a busy period ends only by accepting an output beat with tlast
//   P6  output data/tvalid are stable while stalled (tready low)
//   P7  no output beats before the first complete input frame is accepted
//
// Environment assumptions: load_en is idle, input frames are well formed
// (tlast on the final beat of a TOTAL_BEATS frame).
//
// Run: sby -f bmc.sby

module ssm_props (
  input logic        clk,
  input logic        rst_n,
  input logic        load_en,
  input logic [1:0]  load_sel,
  input logic [0:0]  load_idx,
  input logic [15:0] load_wdata,
  input logic        s_axis_tvalid,
  input logic [15:0] s_axis_tdata,
  input logic        s_axis_tlast,
  input logic        m_axis_tready
);

  localparam int NUM_HEADS = 2;
  localparam int HEAD_DIM  = 1;
  localparam int D_STATE   = 1;
  localparam int XN        = NUM_HEADS * HEAD_DIM;
  localparam int TOTAL_BEATS = XN + 2 * D_STATE + NUM_HEADS;

  logic        s_axis_tready;
  logic        m_axis_tvalid;
  logic [31:0] m_axis_tdata;
  logic        m_axis_tlast;
  logic        busy;

  ssm_unit #(
    .NUM_HEADS(NUM_HEADS),
    .HEAD_DIM (HEAD_DIM),
    .D_STATE  (D_STATE)
  ) u_dut (
    .clk(clk),
    .rst_n(rst_n),
    .load_en(load_en),
    .load_sel(load_sel),
    .load_idx(load_idx),
    .load_wdata(load_wdata),
    .s_axis_tvalid(s_axis_tvalid),
    .s_axis_tready(s_axis_tready),
    .s_axis_tdata(s_axis_tdata),
    .s_axis_tlast(s_axis_tlast),
    .m_axis_tvalid(m_axis_tvalid),
    .m_axis_tready(m_axis_tready),
    .m_axis_tdata(m_axis_tdata),
    .m_axis_tlast(m_axis_tlast),
    .busy(busy)
  );

  wire in_fire = s_axis_tvalid && s_axis_tready;
  wire out_fire = m_axis_tvalid && m_axis_tready;

  // Input frame model: the environment starts a frame at reset and marks its
  // final beat with tlast, so the beat counter below stays in sync with the
  // DUT's own in_cnt.
  logic [4:0] icnt;
  logic [15:0] in_frames;

  always @(posedge clk) begin
    if (!rst_n) begin
      icnt <= 5'd0;
    end else if (in_fire) begin
      if (s_axis_tlast)
        icnt <= 5'd0;
      else
        icnt <= icnt + 5'd1;
    end
  end

  always @(posedge clk) begin
    if (!rst_n)
      in_frames <= 16'd0;
    else if (in_fire && s_axis_tlast)
      in_frames <= in_frames + 16'd1;
  end

  // Set while an activation (accepted input frame) is being processed; the
  // initial state-clear busy period is not an activation.
  logic act;

  always @(posedge clk) begin
    if (!rst_n)
      act <= 1'b0;
    else if (in_fire && s_axis_tlast)
      act <= 1'b1;
    else if (!busy)
      act <= 1'b0;
  end

  // Output beat counter within the current activation.
  logic [4:0] ocnt;

  always @(posedge clk) begin
    if (!rst_n || !busy)
      ocnt <= 5'd0;
    else if (out_fire)
      ocnt <= ocnt + 5'd1;
  end

  // The design must start in reset.
  initial assume (!rst_n);

  // Well-formed-frame assumption.
  always @(posedge clk) begin
    if (rst_n) begin
      assume (!load_en);
      assume (!s_axis_tlast || s_axis_tvalid);
      // tlast on exactly the final beat of a TOTAL_BEATS frame
      assume (!s_axis_tlast || (icnt == TOTAL_BEATS - 1));
      assume ((icnt != TOTAL_BEATS - 1) || !s_axis_tvalid || s_axis_tlast);
    end
  end

  reg [2:0] f_past_valid = 3'b000;
  always @(posedge clk)
    f_past_valid <= {f_past_valid[1:0], 1'b1};

  always @(posedge clk) begin
    if (rst_n) begin
      assert (s_axis_tready == !busy);                       // P1
      assert (!m_axis_tvalid || busy);                       // P2
      assert (m_axis_tlast == (m_axis_tvalid &&
              (ocnt == XN - 1)));                            // P3
      assert (!busy || (ocnt < XN));                         // P4
      assert (!m_axis_tvalid || (in_frames != 0));           // P7
    end
    if (&f_past_valid && rst_n && $past(rst_n) && $past(busy) &&
        $past(act)) begin
      assert (busy || $past(m_axis_tvalid && m_axis_tready &&
                             m_axis_tlast));                 // P5
    end
    // P6: once a beat is offered and stalled, it stays valid and stable.
    if (&f_past_valid && rst_n && $past(rst_n) &&
        $past(m_axis_tvalid && !m_axis_tready)) begin
      assert (m_axis_tvalid && $stable(m_axis_tdata) &&
              $stable(m_axis_tlast));
    end
  end

  // Covers: a full frame is processed and the busy period ends.
  always @(posedge clk) begin
    if (rst_n) begin
      cover (busy);
      cover (!busy && $past(busy));
      cover (out_fire && m_axis_tlast);
    end
  end

endmodule
