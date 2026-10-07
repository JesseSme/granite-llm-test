// Handshake-exact stub for formal verification: accepts IN_FEATURES beats
// while idle, then emits OUT_FEATURES beats with constant data.
module matrix_unit #(
  parameter int IN_FEATURES = 768,
  parameter int OUT_FEATURES = 768,
  parameter int W_DATA = 16,
  // LANES only changes the internal row-parallel schedule of the real unit;
  // the external handshake modelled here is lane-count independent.
  parameter int LANES = 1,
  parameter int IN_W = $clog2(IN_FEATURES),
  parameter int OUT_W = $clog2(OUT_FEATURES)
) (
  input  logic clk, rst_n,
  input  logic load_en, load_is_bias,
  input  logic [OUT_W-1:0] load_out_idx,
  input  logic [IN_W-1:0] load_in_idx,
  input  logic [W_DATA-1:0] load_wdata,
  input  logic s_axis_tvalid, output logic s_axis_tready,
  input  logic [W_DATA-1:0] s_axis_tdata, input logic s_axis_tlast,
  output logic m_axis_tvalid, input logic m_axis_tready,
  output logic [W_DATA-1:0] m_axis_tdata, output logic m_axis_tlast,
  output logic busy
);
  typedef enum logic [1:0] {IDLE, IN, OUT} st_t;
  st_t st;
  logic [IN_W-1:0] icnt;
  logic [OUT_W-1:0] ocnt;

  assign s_axis_tready = (st == IDLE) || (st == IN);
  assign m_axis_tvalid = (st == OUT);
  assign m_axis_tdata  = 16'h3C00;
  assign m_axis_tlast  = (st == OUT) && (ocnt == OUT_W'(OUT_FEATURES - 1));
  assign busy          = (st != IDLE);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st <= IDLE; icnt <= '0; ocnt <= '0;
    end else begin
      case (st)
        IDLE: if (s_axis_tvalid) begin icnt <= IN_W'(1); st <= IN; end
        IN:   if (s_axis_tvalid) begin
                if (icnt == IN_W'(IN_FEATURES - 1)) begin st <= OUT; ocnt <= '0; end
                else icnt <= icnt + 1'b1;
              end
        OUT:  if (m_axis_tready) begin
                if (ocnt == OUT_W'(OUT_FEATURES - 1)) st <= IDLE;
                else ocnt <= ocnt + 1'b1;
              end
        default: st <= IDLE;
      endcase
    end
  end
endmodule
