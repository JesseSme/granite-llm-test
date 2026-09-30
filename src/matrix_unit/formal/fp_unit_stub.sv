// Formal abstraction of fp_unit: outputs are free (anyseq) each cycle.
//
// Used instead of blackboxing because SBY's `hierarchy -smtcheck` rejects
// blackbox module instances. Same port list and parameters as the real unit.
// Control-path properties verified against this stub hold for any FP result.

module fp_unit #(
  parameter int W_EXP  = 8,
  parameter int W_MANT = 23
) (
  input  logic                      clk,
  input  logic                      rst_n,
  input  logic                      in_valid,
  input  fp_pkg::op_t               mode,
  input  fp_pkg::rounding_t         rm,
  input  logic [1+W_EXP+W_MANT-1:0] a,
  input  logic [1+W_EXP+W_MANT-1:0] b,
  input  logic [1+W_EXP+W_MANT-1:0] c,
  output logic [1+W_EXP+W_MANT-1:0] y,
  output logic [1:0]                cmp,
  output logic [4:0]                flags,
  output logic                      out_valid
);

  (* anyseq *) logic [1+W_EXP+W_MANT-1:0] y_free;
  (* anyseq *) logic [1:0]                cmp_free;
  (* anyseq *) logic [4:0]                flags_free;
  (* anyseq *) logic                      out_valid_free;

  assign y     = y_free;
  assign cmp   = cmp_free;
  assign flags = flags_free;
  assign out_valid = out_valid_free;

endmodule
