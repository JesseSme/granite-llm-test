// Formal properties for conv1d_unit (control-path invariants).
//
// The fp_unit datapath is abstracted (anyseq stub), so these properties
// verify the stream protocol independent of FP arithmetic:
//   - an output requires at least one accepted input
//   - never more outputs than accepted inputs, and at most one input
//     outstanding at a time
//
// Note: valid_o is a registered pulse, so it remains high during the first
// S_IDLE cycle after S_OUTPUT and can overlap ready_o = 1. That is expected
// valid/ready behavior (valid does not wait for ready).
//
// CHANNELS is reduced for BMC tractability; the FSM counters scale with the
// parameter and the properties are checked for the same parameterized logic.
//
// Run: sby -f bmc.sby

module conv1d_props #(
  parameter int CHANNELS = 4,
  parameter int KERNEL   = 4
) (
  input logic        clk,
  input logic        rst_n,
  input logic        valid_i,
  input logic [15:0] data_i,
  input logic        load_en,
  input logic [$clog2(CHANNELS)-1:0] load_ch,
  input logic [$clog2(KERNEL)-1:0]   load_tap,
  input logic [15:0] load_wdata,
  input logic        load_is_bias
);

  logic        ready_o;
  logic        valid_o;
  logic [15:0] data_o;

  conv1d_unit #(.CHANNELS(CHANNELS), .KERNEL(KERNEL)) u_dut (
    .clk          (clk),
    .rst_n        (rst_n),
    .valid_i      (valid_i),
    .data_i       (data_i),
    .ready_o      (ready_o),
    .valid_o      (valid_o),
    .data_o       (data_o),
    .load_en      (load_en),
    .load_ch      (load_ch),
    .load_tap     (load_tap),
    .load_wdata   (load_wdata),
    .load_is_bias (load_is_bias)
  );

  logic [15:0] in_count;
  logic [15:0] out_count;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      in_count  <= '0;
      out_count <= '0;
    end else begin
      if (valid_i && ready_o)
        in_count <= in_count + 1'b1;
      if (valid_o)
        out_count <= out_count + 1'b1;
    end
  end

  // Start from reset so the DUT and the tracking counters are consistent.
  initial assume (!rst_n);

  // Suppresses assertions on the first clock so that $past is well-defined.
  reg f_past_valid = 1'b0;
  always @(posedge clk)
    f_past_valid <= 1'b1;

  always @(posedge clk) begin
    if (f_past_valid && rst_n && $past(rst_n)) begin
      assert (!valid_o || in_count > 0);
      assert (out_count <= in_count);
      assert ((in_count - out_count) <= 1);
    end
  end

  always @(posedge clk) begin
    cover (valid_o);
    cover (valid_i && ready_o);
    cover (valid_i && !ready_o);
  end

endmodule
