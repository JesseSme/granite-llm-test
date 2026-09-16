// Formal properties for residual_adder_unit (control/pipeline invariants).
//
// The fp_unit datapaths are abstracted (anyseq stubs), so this property
// verifies the pipeline timing independent of FP arithmetic:
//   - valid_out is exactly valid_in delayed by three clocks
//     (MUL register, ADD register, output register) with no stalls.
//
// Run: sby -f bmc.sby

module residual_adder_props (
  input logic        clk,
  input logic        rst_n,
  input logic        valid_in,
  input logic [15:0] data_in,
  input logic [15:0] residual_in
);

  logic [15:0] data_out;
  logic        valid_out;

  wire dut_valid_in = valid_in & rst_n;

  residual_adder_unit u_dut (
    .clk         (clk),
    .rst_n       (rst_n),
    .valid_in    (dut_valid_in),
    .data_in     (data_in),
    .residual_in (residual_in),
    .data_out    (data_out),
    .valid_out   (valid_out)
  );

  // Suppresses assertions until $past(_, 3) is well-defined.
  reg [2:0] f_past_valid = 3'b000;
  always @(posedge clk)
    f_past_valid <= {f_past_valid[1:0], 1'b1};

  always @(posedge clk) begin
    if (&f_past_valid && rst_n && $past(rst_n) && $past(rst_n, 2) && $past(rst_n, 3))
      assert (valid_out == $past(dut_valid_in, 3));
  end

  always @(posedge clk) begin
    cover (valid_out);
    cover (!valid_out);
  end

endmodule
