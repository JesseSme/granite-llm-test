// Formal abstraction of conv1d_unit: accepts CHANNELS input beats (valid_i
// while ready_o), then streams CHANNELS output beats one per cycle with
// constant data. Same port list as the real unit.

module conv1d_unit #(
  parameter int CHANNELS = 1536,
  parameter int KERNEL   = 4,
  parameter int PADDING  = 3,
  parameter int W_DATA   = 16
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        valid_i,
  input  logic [W_DATA-1:0] data_i,
  output logic        ready_o,
  output logic        valid_o,
  output logic [W_DATA-1:0] data_o,
  input  logic        load_en,
  input  logic [$clog2(CHANNELS)-1:0] load_ch,
  input  logic [$clog2(KERNEL)-1:0] load_tap,
  input  logic [W_DATA-1:0] load_wdata,
  input  logic        load_is_bias
);

  logic busy_r;
  logic [$clog2(CHANNELS):0] icnt, ocnt;

  assign ready_o = !busy_r;
  assign valid_o = busy_r;
  assign data_o  = '0;

  always @(posedge clk) begin
    if (!rst_n) begin
      busy_r <= 1'b0;
      icnt   <= '0;
      ocnt   <= '0;
    end else if (!busy_r) begin
      if (valid_i) begin
        if (icnt == CHANNELS - 1) begin
          icnt   <= '0;
          ocnt   <= '0;
          busy_r <= 1'b1;
        end else begin
          icnt <= icnt + 1'b1;
        end
      end
    end else begin
      if (ocnt == CHANNELS - 1) begin
        ocnt   <= '0;
        busy_r <= 1'b0;
      end else begin
        ocnt <= ocnt + 1'b1;
      end
    end
  end

endmodule
