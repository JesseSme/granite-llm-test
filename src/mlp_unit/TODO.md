# mlp_unit — TODO

## Code Review Checklist

- [x] RTL passes Verilator lint with `-Wall` (no warnings; checked at
      HIDDEN=16/INTER=8 and 768/2048).
- [x] cocotb testbench passes against golden sample for all test vectors
      (64/64 bit-exact, small config).
- [x] In-loop test passes — real Granite shared-MLP layer 0, 4 tokens:
      bit-exact vs the sequential-order emulation, 0/3072 outside 2e-2 vs the
      model.
- [x] `description.yaml` is updated with interface, Fmax, and characteristics.
- [x] No hardcoded magic numbers — use parameters or `define` constants.
- [x] Follows existing code conventions (see `systemverilog_fp_unit/rtl/`).

## Implementation Steps

- [x] 1. Set `current_dut: mlp_unit` in description.yaml (SwiGLU_unit
      description updated as its sub-unit)
- [x] 2. Implement `SwiGLU_unit/swiglu_unit.sv` (gate+up -> silu -> multiply -> down)
- [x] 3. Implement `mlp_unit.sv` (pass-through wrapper, model class name)
- [x] 4. Lint with Verilator (`verilator --lint-only -Wall`)
- [x] 5. Generate golden sample (`gen_golden.py`, fp32 reference)
- [x] 6. Write cocotb testbench (`tb_mlp_unit.py`) and runner
- [x] 7. Run RTL simulation against golden sample
- [x] 8. In-loop verification with real Granite MLP inputs
- [x] 9. Formal verification (`formal/bmc.sby`)
- [x] 10. Update `description.yaml` — interface, Fmax, characteristics
- [x] 11. Clear `current_dut` field (never left set; the unit is complete)

## Notes / issues found

- Lint caught a real bug: `I_W'(INTER)` truncates to 0 when INTER is a power of
  two, making `i_cnt < I_W'(INTER)` always false (the silu would never be fed).
  Fixed by comparing against the plain integer bound. The shared weight-load
  index ports also had to be widened to cover both projections
  (`clog2(max(2*INTER, HIDDEN))` / `clog2(max(HIDDEN, INTER))`).
- fp_unit has registered outputs, so operands must not fall back to 0 between
  issuing an op and reading its result: the first version held combinational
  operands across wait states, which clobbered the adder output before the
  divider read it (1/0 -> inf -> NaN in every output). The silu pipeline now
  latches each intermediate result into a register (e_reg, sig_reg, act_reg,
  gated_reg), which is robust and matches softmax_unit's style.
- Accuracy: the LUT-based silu_unit (sigmoid LUT < 0.5%, 1/32 input grid)
  compounds through the down projection to up to ~8% relative output error
  (measured), so SwiGLU uses an accurate inline silu built from fp_exp_seq
  instead. With it the unit is bit-exact vs the fp32 golden and vs the
  sequential-order emulation in-loop.
- The in-loop model comparison is looser (2e-2) because ATen's blocked-GEMM
  accumulation order differs from the RTL's sequential fp32 accumulation over
  the 2048-term down projection (measured 7.8e-3 absolute).

- MATRIX_LANES = 4 rollout (branch opt/lanes-rollout): both matrix_unit
  instances (gate+up and down) now compute four output rows per pass
  (bit-identical). Verified: lint clean; unit test 64/64 bit-exact; in-loop
  sequential-emulation comparison 0/3072 outside 1e-3 (max abs 0.0) and model
  comparison 0/3072 outside 2e-2 (max abs 7.8e-3), in ~38 min (was ~44 min).
  Measured ~1.25M cycles/token (was ~4.8M).
