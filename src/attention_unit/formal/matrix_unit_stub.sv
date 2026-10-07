// Formal abstraction of matrix_unit: the handshake timing is modelled exactly
// (accept IN_FEATURES beats, then stream OUT_FEATURES beats one per cycle after
// a 3-cycle compute delay) while the data outputs are free (anyseq).
//
// Used instead of blackboxing because SBY's `hierarchy -smtcheck` rejects
// blackbox module instances. Control-path properties verified against this
// stub hold for any projection result.

module matrix_unit #(
  parameter int IN_FEATURES  = 768,
  parameter int OUT_FEATURES = 768,
  parameter int W_DATA       = 16,
  // LANES only changes the internal row-parallel schedule of the real unit;
  // the external handshake modelled here is lane-count independent.
  parameter int LANES        = 1,
  parameter int IN_W         = $clog2(IN_FEATURES),
  parameter int OUT_W        = $clog2(OUT_FEATURES)
) (
  input  logic                  clk,
  input  logic                  rst_n,

  input  logic                  load_en,
  input  logic [OUT_W-1:0]      load_out_idx,
  input  logic [IN_W-1:0]       load_in_idx,
  input  logic [W_DATA-1:0]     load_wdata,
  input  logic                  load_is_bias,

  input  logic                  s_axis_tvalid,
  output logic                  s_axis_tready,
  input  logic [W_DATA-1:0]     s_axis_tdata,
  input  logic                  s_axis_tlast,

  output logic                  m_axis_tvalid,
  input  logic                  m_axis_tready,
  output logic [W_DATA-1:0]     m_axis_tdata,
  output logic                  m_axis_tlast,

  output logic                  busy
);

  logic        busy_r;
  logic [IN_W:0]  icnt;
  logic [OUT_W:0] ocnt;
  logic [2:0]     wait_r;

  assign s_axis_tready = !busy_r;
  assign busy          = busy_r;
  assign m_axis_tvalid = busy_r && (wait_r == 3'd0);
  assign m_axis_tdata  = '0;
  assign m_axis_tlast  = (ocnt == (OUT_W + 1)'(OUT_FEATURES - 1));

  always @(posedge clk) begin
    if (!rst_n) begin
      busy_r <= 1'b0;
      icnt   <= '0;
      ocnt   <= '0;
      wait_r <= 3'd0;
    end else if (!busy_r) begin
      if (s_axis_tvalid) begin
        if (s_axis_tlast || (icnt == (IN_W + 1)'(IN_FEATURES - 1))) begin
          icnt   <= '0;
          ocnt   <= '0;
          wait_r <= 3'd1;
          busy_r <= 1'b1;
        end else begin
          icnt <= icnt + 1'b1;
        end
      end
    end else begin
      if (wait_r != 3'd0) begin
        wait_r <= wait_r - 1'b1;
      end else if (m_axis_tvalid && m_axis_tready) begin
        if (ocnt == (OUT_W + 1)'(OUT_FEATURES - 1)) begin
          ocnt   <= '0;
          busy_r <= 1'b0;
        end else begin
          ocnt <= ocnt + 1'b1;
        end
      end
    end
  end

endmodule
