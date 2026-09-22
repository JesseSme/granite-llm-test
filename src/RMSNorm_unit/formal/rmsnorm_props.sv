// Formal properties for rmsnorm_unit (control-path invariants).
//
// The fp_unit datapath is abstracted (anyseq stub), so these properties verify
// the FSM protocol independent of FP arithmetic:
//   - no output before the weight vector has been fully loaded
//   - output only while busy
//   - busy only rises after a valid input was accepted
//   - exactly WIDTH outputs are produced per accepted vector
//
// WIDTH is reduced for BMC tractability; the FSM counters scale with the
// parameter and the properties are checked for the same parameterized logic.
//
// Run: sby -f bmc.sby

module rmsnorm_props #(
  parameter int WIDTH = 4
) (
  input logic        clk,
  input logic        rst_n,
  input logic        valid_in,
  input logic [15:0] data_in,
  input logic        weight_valid,
  input logic [15:0] weight_in
);

  logic [15:0] data_out;
  logic        valid_out;
  logic        busy;

  rmsnorm_unit #(.WIDTH(WIDTH)) u_dut (
    .clk          (clk),
    .rst_n        (rst_n),
    .valid_in     (valid_in),
    .data_in      (data_in),
    .weight_valid (weight_valid),
    .weight_in    (weight_in),
    .data_out     (data_out),
    .valid_out    (valid_out),
    .busy         (busy)
  );

  // External weight-load tracker (>= WIDTH pulses required by the DUT).
  logic [15:0] weight_seen;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      weight_seen <= '0;
    else if (weight_valid)
      weight_seen <= weight_seen + 1'b1;
  end
  wire weight_done = (weight_seen >= WIDTH);

  // Outputs counted while the DUT is busy (cleared when idle between vectors).
  logic [15:0] out_count;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      out_count <= '0;
    else if (!busy)
      out_count <= '0;
    else if (valid_out)
      out_count <= out_count + 1'b1;
  end

  // Start from reset so the DUT and the tracking counters are consistent.
  initial assume (!rst_n);

  // Suppresses assertions on the first clock so that $past is well-defined.
  reg f_past_valid = 1'b0;
  always @(posedge clk)
    f_past_valid <= 1'b1;

  always @(posedge clk) begin
    if (f_past_valid && rst_n && $past(rst_n)) begin
      assert (!valid_out || weight_done);
      assert (!valid_out || busy);
      assert (!(busy && !$past(busy)) || $past(valid_in));
      assert (!(!busy && $past(busy)) || out_count == WIDTH);
    end
  end

  always @(posedge clk) begin
    cover (valid_out);
    cover (busy);
    cover (valid_out && busy);
    cover (!busy && $past(busy));
  end

endmodule
