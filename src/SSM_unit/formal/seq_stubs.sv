// Formal abstractions of the SSM unit's sequential FP helpers.
//
// fp_exp_seq / fp_softplus_seq results are free (anyseq); their `done` output
// is free too but the environment must respond within 4 cycles of `start`
// (bounded-latency assumption). Control-path properties verified against these
// stubs hold for any FP result and any (bounded) computation latency.

module fp_exp_seq (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  input  logic [31:0] x,
  output logic [31:0] y,
  output logic        done
);

  (* anyseq *) logic [31:0] y_free;
  (* anyseq *) logic        done_free;

  logic [2:0] cnt;

  always @(posedge clk) begin
    if (!rst_n) begin
      cnt <= 3'd0;
    end else if (start) begin
      cnt <= 3'd1;
    end else if (cnt != 3'd0) begin
      cnt <= cnt + 3'd1;
    end
  end

  assign y    = y_free;
  assign done = rst_n && done_free && (cnt != 3'd0);

  always @(posedge clk)
    if (rst_n && cnt == 3'd4)
      assume (done_free);

endmodule

module fp_softplus_seq (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  input  logic [31:0] z,
  output logic [31:0] y,
  output logic        done
);

  (* anyseq *) logic [31:0] y_free;
  (* anyseq *) logic        done_free;

  logic [2:0] cnt;

  always @(posedge clk) begin
    if (!rst_n) begin
      cnt <= 3'd0;
    end else if (start) begin
      cnt <= 3'd1;
    end else if (cnt != 3'd0) begin
      cnt <= cnt + 3'd1;
    end
  end

  assign y    = y_free;
  assign done = rst_n && done_free && (cnt != 3'd0);

  always @(posedge clk)
    if (rst_n && cnt == 3'd4)
      assume (done_free);

endmodule
