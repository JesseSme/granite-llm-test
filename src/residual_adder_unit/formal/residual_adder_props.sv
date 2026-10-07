// Formal properties for residual_adder_unit (control/pipeline invariants).
//
// The fp_unit datapaths are abstracted (anyseq stubs), so this property
// verifies the pipeline timing independent of FP arithmetic:
//   - valid_out is exactly valid_in delayed by eight clocks. The fp_unit
//     protocol makes the multiply result valid four cycles after its start
//     pulse and the add result three cycles after its start, and each output
//     is registered once more (4 + 3 + 1 = 8) with no stalls.
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

  // Suppresses assertions until $past(_, 8) is well-defined, and requires the
  // whole 8-cycle window to be out of reset (a mid-window reset would clear
  // the pipeline and make the delay comparison meaningless).
  reg [7:0] f_past_valid = 8'b0;
  reg [8:0] rst_hist = 9'b0;
  always @(posedge clk) begin
    f_past_valid <= {f_past_valid[6:0], 1'b1};
    rst_hist     <= {rst_hist[7:0], rst_n};
  end

  always @(posedge clk) begin
    if (&f_past_valid && (&rst_hist))
      assert (valid_out == $past(dut_valid_in, 8));
  end

  always @(posedge clk) begin
    cover (valid_out);
    cover (!valid_out);
  end

endmodule
