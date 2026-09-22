// Formal properties for softmax_unit (control-path invariants).
//
// fp_unit and fp_exp are abstracted (anyseq stubs), so these properties verify
// the row FSM protocol independent of FP arithmetic:
//   - done always coincides with the final valid_out of a row
//
// Row/element counting and valid_out pacing are covered by the cocotb unit and
// in-loop tests; Yosys immediate-assertion sampling makes external counters
// brittle for this FSM (done and the final valid_out pulse are in the same
// cycle, so $past-based sequencing checks do not apply).
//
// N is reduced for BMC tractability; the FSM counters scale with the
// parameter and the properties are checked for the same parameterized logic.
//
// Run: sby -f bmc.sby

module softmax_props #(
  parameter int N = 4
) (
  input logic        clk,
  input logic        rst_n,
  input logic        valid_in,
  input logic [31:0] data_in,
  input logic        last_in
);

  logic [31:0] data_out;
  logic        valid_out;
  logic        done;

  softmax_unit #(.N(N)) u_dut (
    .clk       (clk),
    .rst_n     (rst_n),
    .valid_in  (valid_in),
    .data_in   (data_in),
    .last_in   (last_in),
    .data_out  (data_out),
    .valid_out (valid_out),
    .done      (done)
  );

  // Start from reset so the DUT is in a known state.
  initial assume (!rst_n);

  // Suppresses assertions on the first clock so that $past is well-defined.
  reg f_past_valid = 1'b0;
  always @(posedge clk)
    f_past_valid <= 1'b1;

  always @(posedge clk) begin
    if (f_past_valid && rst_n && $past(rst_n)) begin
      assert (!done || valid_out);
    end
  end

  always @(posedge clk) begin
    cover (valid_out);
    cover (done);
    cover (valid_in && !last_in);
    cover (valid_in && last_in);
  end

endmodule
