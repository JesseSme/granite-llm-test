// Formal properties for attention_unit (AXI-Stream framing / sequencing).
//
// The FP datapaths are abstracted: matrix_unit is replaced by a stub with the
// same handshake timing but free data, and fp_unit/fp_exp_seq outputs are
// anyseq. So the properties below are independent of numeric results:
//   P1  s_axis_tready is high exactly while idle (!busy)
//   P2  m_axis_tvalid only while busy and tlast exactly on the HIDDEN-th beat
//   P3  at most HIDDEN output beats per activation, and a busy period ends
//       only after HIDDEN beats have been accepted
//   P4  no output beats before the first complete input frame
//
// Environment: load_en idle, well-formed input frames (HIDDEN beats with tlast
// on the final beat).
//
// Run: sby -f bmc.sby

module attention_props (
  input logic        clk,
  input logic        rst_n,
  input logic        load_en,
  input logic [1:0]  load_sel,
  input logic [9:0]  load_out_idx,
  input logic [9:0]  load_in_idx,
  input logic [15:0] load_wdata,
  input logic        s_axis_tvalid,
  input logic [15:0] s_axis_tdata,
  input logic        s_axis_tlast,
  input logic        m_axis_tready
);

  localparam int HIDDEN    = 8;
  localparam int NUM_HEADS = 2;
  localparam int KV_HEADS  = 1;
  localparam int HEAD_DIM  = 4;
  localparam int MAX_SEQ   = 2;

  logic        s_axis_tready;
  logic        m_axis_tvalid;
  logic [15:0] m_axis_tdata;
  logic        m_axis_tlast;
  logic        busy;

  attention_unit #(
    .HIDDEN(HIDDEN), .NUM_HEADS(NUM_HEADS), .NUM_KV_HEADS(KV_HEADS),
    .HEAD_DIM(HEAD_DIM), .MAX_SEQ(MAX_SEQ)
  ) u_dut (
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

  wire in_fire  = s_axis_tvalid && s_axis_tready;
  wire out_fire = m_axis_tvalid && m_axis_tready;

  // Input frame model (kept in sync with the DUT's own counter by the
  // well-formed-frame assumption below).
  logic [3:0]  icnt;
  logic [15:0] in_frames;

  always @(posedge clk) begin
    if (!rst_n) begin
      icnt <= 4'd0;
    end else if (in_fire) begin
      icnt <= s_axis_tlast ? 4'd0 : icnt + 4'd1;
    end
  end

  always @(posedge clk) begin
    if (!rst_n)
      in_frames <= 16'd0;
    else if (in_fire && s_axis_tlast)
      in_frames <= in_frames + 16'd1;
  end

  // Output beats accepted during the current activation.
  logic [3:0] ocnt;

  always @(posedge clk) begin
    if (!rst_n || !busy)
      ocnt <= 4'd0;
    else if (out_fire)
      ocnt <= ocnt + 4'd1;
  end

  initial assume (!rst_n);

  always @(posedge clk) begin
    if (rst_n) begin
      assume (!load_en);
      assume (!s_axis_tlast || s_axis_tvalid);
      assume (!s_axis_tlast || (icnt == HIDDEN - 1));
      assume ((icnt != HIDDEN - 1) || !s_axis_tvalid || s_axis_tlast);
    end
  end

  reg [2:0] f_past_valid = 3'b000;
  always @(posedge clk)
    f_past_valid <= {f_past_valid[1:0], 1'b1};

  always @(posedge clk) begin
    if (rst_n) begin
      assert (s_axis_tready == !busy);                       // P1
      assert (!m_axis_tvalid || busy);                       // P2
      assert (m_axis_tlast == (m_axis_tvalid && (ocnt == HIDDEN - 1)));
      assert (!busy || (ocnt < HIDDEN));                     // P3
      assert (!m_axis_tvalid || (in_frames != 0));           // P4
    end
    // P3: a busy period ends only after all HIDDEN output beats were accepted.
    if (&f_past_valid && rst_n && $past(rst_n) && $past(busy)) begin
      assert (busy || $past(out_fire));
    end
  end

  always @(posedge clk) begin
    if (rst_n) begin
      cover (busy);
      cover (!busy && $past(busy));
      cover (out_fire && m_axis_tlast);
    end
  end

endmodule
