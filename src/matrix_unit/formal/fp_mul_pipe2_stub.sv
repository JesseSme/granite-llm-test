// Formal abstraction of fp_mul_pipe2: the output is free (anyseq) each cycle.
//
// Used instead of blackboxing because SBY's `hierarchy -smtcheck` rejects
// blackbox module instances. Same port list and parameters as the real unit.
// Control-path properties verified against this stub hold for any FP result.

module fp_mul_pipe2 #(
  parameter int W_EXP  = 8,
  parameter int W_MANT = 23
) (
  input  logic                      clk,
  input  logic                      rst_n,
  input  logic [1+W_EXP+W_MANT-1:0] a,
  input  logic [1+W_EXP+W_MANT-1:0] b,
  output logic [1+W_EXP+W_MANT-1:0] y
);

  (* anyseq *) logic [1+W_EXP+W_MANT-1:0] y_free;

  assign y = y_free;

endmodule
