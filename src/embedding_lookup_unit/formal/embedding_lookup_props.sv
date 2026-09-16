// Formal properties for embedding_lookup_unit (control-path invariants).
//
// The fp_unit datapath is abstracted (anyseq stub), so these properties verify
// the lookup stream protocol independent of FP arithmetic:
//   - output only while busy
//   - at most DIM outputs per accepted token
//   - busy only rises after a valid input was accepted
//   - busy falls only after exactly DIM outputs
//
// VOCAB/DIM are reduced for BMC tractability; the FSM counters scale with the
// parameters and the properties are checked for the same parameterized logic.
//
// Run: sby -f bmc.sby

module embedding_lookup_props #(
  parameter int VOCAB = 4,
  parameter int DIM   = 4
) (
  input logic                    clk,
  input logic                    rst_n,
  input logic                    valid_in,
  input logic [$clog2(VOCAB)-1:0] token_id,
  input logic                    load_en,
  input logic [$clog2(VOCAB)-1:0] load_token,
  input logic [$clog2(DIM)-1:0]   load_idx,
  input logic [15:0]             load_wdata
);

  logic [15:0] data_out;
  logic        valid_out;
  logic        busy;

  embedding_lookup_unit #(.VOCAB(VOCAB), .DIM(DIM)) u_dut (
    .clk        (clk),
    .rst_n      (rst_n),
    .load_en    (load_en),
    .load_token (load_token),
    .load_idx   (load_idx),
    .load_wdata (load_wdata),
    .valid_in   (valid_in),
    .token_id   (token_id),
    .data_out   (data_out),
    .valid_out  (valid_out),
    .busy       (busy)
  );

  // Outputs counted while busy (cleared while idle between tokens).
  logic [15:0] out_count;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      out_count <= '0;
    else if (!busy)
      out_count <= '0;
    else if (valid_out)
      out_count <= out_count + 1'b1;
  end

  // Start from reset so the DUT and the tracking counter are consistent.
  initial assume (!rst_n);

  // Suppresses assertions on the first clock so that $past is well-defined.
  reg f_past_valid = 1'b0;
  always @(posedge clk)
    f_past_valid <= 1'b1;

  always @(posedge clk) begin
    if (f_past_valid && rst_n && $past(rst_n)) begin
      assert (!valid_out || busy);
      assert (!valid_out || out_count < DIM);
      assert (!(busy && !$past(busy)) || $past(valid_in));
      assert (!(!busy && $past(busy)) || out_count == DIM);
    end
  end

  always @(posedge clk) begin
    cover (valid_out);
    cover (busy);
    cover (valid_out && busy);
  end

endmodule
