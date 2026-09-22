// Formal abstraction of silu_seq: accepts an element (valid_i while ready_o)
// and pulses valid_o one cycle later with constant binary32 data.

module silu_seq (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        valid_i,
  output logic        ready_o,
  input  logic [15:0] data_i,
  output logic        valid_o,
  output logic [31:0] data_o
);

  assign ready_o = !valid_o;
  assign data_o  = '0;

  always @(posedge clk) begin
    if (!rst_n) valid_o <= 1'b0;
    else        valid_o <= valid_i && ready_o;
  end

endmodule
