# output_projection_unit - TODO

## Code Review Checklist

- [x] RTL passes Verilator lint with `-Wall` (no warnings).
- [x] cocotb testbench passes against golden sample for all test vectors
      (bit-exact 256/256, max abs/rel 0.0).
- [x] In-loop test passes - real Granite final-norm activations and tied
      lm_head weights (512 sampled vocab rows, bit-exact 1024/1024).
- [x] `description.yaml` is updated with interface, Fmax, and characteristics.
- [x] No hardcoded magic numbers - HIDDEN/VOCAB parameters, scaling constant
      named C_LOGITS_SCALING.
- [x] Follows existing code conventions (matrix_unit/conv1d_unit style).

## Implementation Steps

- [x] 1. Set `current_dut: output_projection_unit` in description.yaml
- [x] 2. Implement `output_projection_unit.sv` (matrix_unit + /logits_scaling)
- [x] 3. Lint with Verilator (`verilator --lint-only -Wall`)
- [x] 4. Generate golden sample (`gen_golden.py`, torch bf16 reference)
- [x] 5. Write cocotb testbench and runner
- [x] 6. Run RTL simulation against golden sample
- [x] 7. In-loop verification (real final-norm hidden states, sampled vocab)
- [x] 8. Formal verification (`formal/bmc.sby`, depth 60, PASS)
- [x] 9. Update `description.yaml`
- [x] 10. Clear `current_dut` field

## Notes / issues found

- The unit description said the logits are multiplied by logits_scaling (=3);
  the installed `modeling_granitemoehybrid.py` divides instead
  (`logits = logits / self.config.logits_scaling`), which is what this unit
  implements and what the golden/in-loop tests compare against.
- The full LM head has 100352 x 768 = 77M bfloat16 weights (~154 MB), which
  cannot be simulated in full. The unit test uses a small VOCAB; the in-loop
  test samples 512 vocabulary rows (each streamed position's argmax row plus
  seeded random rows) and loads only those rows. Both comparisons are
  bit-exact.
- The scale stage divides in binary32 and rounds once to bfloat16, matching
  torch's bf16 `tensor / 3` (fp32 internal precision, one rounding); dividing
  by 3.0 instead of multiplying by 1/3 avoids the ~1 ULP error of the
  rounded reciprocal.

- MATRIX_LANES = 4 rollout (branch opt/lanes-rollout): the LM-head matrix_unit
  now computes four output rows per pass (bit-identical). Verified: lint
  clean; unit test bit-exact 256/256; in-loop sampled-vocab run bit-exact
  1024/1024 (max abs/rel 0.0). The full 100352-row head drops from ~77.4M to
  ~19.4M projection cycles.
