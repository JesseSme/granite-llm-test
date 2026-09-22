// Formal properties for silu_unit (control/pipeline invariants).
//
// Output-valid pipeline: valid_out is exactly valid_in delayed by two clocks
// (one for the sigmoid LUT, one for the bfloat16 multiply). valid_in is gated
// with rst_n so the reset release is well-defined.
//
// Run: sby -f bmc.sby

module silu_props (
  input logic        clk,
  input logic        rst_n,
  input logic        valid_in,
  input logic [15:0] data_in
);

  logic [15:0] data_out;
  logic        valid_out;

  wire dut_valid_in = valid_in & rst_n;

  silu_unit u_dut (
    .clk       (clk),
    .rst_n     (rst_n),
    .valid_in  (dut_valid_in),
    .data_in   (data_in),
    .data_out  (data_out),
    .valid_out (valid_out)
  );

  // Suppresses assertions until $past(_, 2) is well-defined.
  reg [1:0] f_past_valid = 2'b00;
  always @(posedge clk)
    f_past_valid <= {f_past_valid[0], 1'b1};

  always @(posedge clk) begin
    if (&f_past_valid && rst_n && $past(rst_n) && $past(rst_n, 2))
      assert (valid_out == $past(dut_valid_in, 2));
  end

  always @(posedge clk) begin
    cover (valid_out);
    cover (!valid_out);
  end

endmodule
