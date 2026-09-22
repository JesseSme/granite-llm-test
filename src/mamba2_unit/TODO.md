# mamba2_unit — TODO

## Code Review Checklist

- [x] RTL passes Verilator lint with `-Wall` (no warnings; checked at
      8/4/2/2/2 and 768/1536/48/32/128).
- [x] cocotb testbench passes against golden sample for all test vectors
      (32/32 within 2e-2; post-conv chain bit-exact vs the DUT's own conv
      outputs).
- [ ] In-loop test passes — real Granite layer-0 Mamba mixer (running).
- [x] `description.yaml` is updated with interface, Fmax, and characteristics.
- [x] No hardcoded magic numbers — use parameters or `define` constants (the
      mean divisor is computed from INTER by an int->fp32 function).
- [x] Follows existing code conventions (see `systemverilog_fp_unit/rtl/`).

## Implementation Steps

- [x] 1. Set `current_dut: mamba2_unit` in description.yaml
- [x] 2. Implement `silu_seq.sv` (accurate streaming SiLU)
- [x] 3. Implement `mamba2_unit.sv` (in_proj -> conv -> SiLU -> SSM -> gated norm -> out_proj)
- [x] 4. Lint with Verilator (`verilator --lint-only -Wall`)
- [x] 5. Generate golden sample (`gen_golden.py`, fp32 reference)
- [x] 6. Write cocotb testbench (`tb_mamba2_unit.py`) and runner
- [x] 7. Run RTL simulation against golden sample
- [~] 8. In-loop verification FAILS at the full config (1408/3840 outside
      2e-2) - scale-specific issue, next step is a full-config unit test with a
      1-2 token sequence to isolate it
- [x] 9. Formal verification (`formal/bmc.sby`, depth 260, PASS)
- [x] 10. Update `description.yaml` — interface, Fmax, characteristics
- [ ] 11. Clear `current_dut` field (kept set: the unit is not complete)

## Notes / issues found

- The gated-norm output pass indexed `ssm_buf`/`gsilu` with the (reset)
  `sq_cnt` instead of `out_cnt`, so every element was normalized from element
  0 — found by dumping the pipeline stages against a Python reference.
- The mean used a hardcoded 1/1536 constant, which is wrong for any other
  INTER (the small unit test uses INTER=4). Replaced by dividing by the exact
  INTER converted to fp32 with an `int_to_fp32` function.
- The conv1d_unit matches ATen's bf16 conv only to ~1 bf16 ULP (its
  accumulation order differs at rounding boundaries, as documented in that
  unit); the SSM recurrence amplifies those ULP differences, so the unit test
  uses a 2e-2 tolerance. Decisive check: driving the SSM/gate/norm/out_proj
  reference with the DUT's own conv outputs matches the DUT bit-exactly.
- Formal used handshake-exact stubs for every sub-unit (matrix_unit,
  conv1d_unit, ssm_unit, silu_seq) because blackboxes are rejected by SBY's
  `hierarchy -smtcheck`; the silu stub keeps the 1-cycle valid_o latency so
  the sequencer's phases stay exercisable.
