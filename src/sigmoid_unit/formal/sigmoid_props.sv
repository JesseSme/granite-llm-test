// Formal properties for sigmoid_unit (control/pipeline invariants).
//
// Output-valid pipeline: valid_out is exactly valid_in delayed by one clock.
// valid_in is gated with rst_n so that the reset release is well-defined.
// The LUT contents themselves are covered by the cocotb unit test.
//
// Run: sby -f bmc.sby

module sigmoid_props (
  input logic        clk,
  input logic        rst_n,
  input logic        valid_in,
  input logic [15:0] data_in
);

  logic [15:0] data_out;
  logic        valid_out;

  wire dut_valid_in = valid_in & rst_n;

  sigmoid_unit u_dut (
    .clk       (clk),
    .rst_n     (rst_n),
    .valid_in  (dut_valid_in),
    .data_in   (data_in),
    .data_out  (data_out),
    .valid_out (valid_out)
  );

  // Suppresses assertions on the first clock so that $past is well-defined.
  reg f_past_valid = 1'b0;
  always @(posedge clk)
    f_past_valid <= 1'b1;

  always @(posedge clk) begin
    if (f_past_valid && rst_n && $past(rst_n))
      assert (valid_out == $past(dut_valid_in));
  end

  always @(posedge clk) begin
    cover (valid_out);
    cover (!valid_out);
  end

endmodule
