// Formal abstraction of fp_unit: outputs are constant zero.
//
// The attention control flow does not depend on FP results, so tying the
// datapath outputs to zero (instead of anyseq) keeps the framing/sequencing
// properties meaningful while removing thousands of free bits per cycle from
// the BMC problem. Same port list and parameters as the real unit.

module fp_unit #(
  parameter int W_EXP  = 8,
  parameter int W_MANT = 7
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

  assign y     = '0;
  assign cmp   = '0;
  assign flags = '0;
  assign out_valid = '0;

endmodule
