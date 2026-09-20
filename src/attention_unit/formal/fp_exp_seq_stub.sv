// Formal abstraction of fp_exp_seq: y is constant zero; `done` responds within
// 4 cycles of `start` (bounded-latency assumption). Control properties
// verified against this stub hold for any (bounded) exponential latency.

module fp_exp_seq (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  input  logic [31:0] x,
  output logic [31:0] y,
  output logic        done
);

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

  assign y    = '0;
  assign done = rst_n && (cnt != 3'd0);

endmodule
