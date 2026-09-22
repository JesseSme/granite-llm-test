// Formal abstraction of fp_exp: output is free (anyseq) each cycle.
//
// Used instead of blackboxing because SBY's `hierarchy -smtcheck` rejects
// blackbox module instances. Same port list as the real unit. Control-path
// properties verified against this stub hold for any exp result.

module fp_exp (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [31:0] x,
  output logic [31:0] y
);

  (* anyseq *) logic [31:0] y_free;

  assign y = y_free;

endmodule
