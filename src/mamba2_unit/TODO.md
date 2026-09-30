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
- [x] 8. Root cause found and fixed: conv1d_unit rounded to bfloat16 at every
      MAC step (~1-2 bf16 ULP vs ATen), and the SSM recurrence amplified the
      drift (3% mismatch at seq=1, 37% at 5 tokens). conv1d_unit now
      accumulates in fp32 (fp_unit W_MANT=23) with a single bf16 rounding.
      Full-config unit test: 768/768, max abs/rel error 0.0 (bit-exact).
      In-loop re-run with the fix in progress.
      In-loop analysis (pure Python, no RTL): given the model's own SSM inputs,
      gen_golden.ssm_seq matches the model's chunk_scan to 3.7e-8, and the
      gated-norm formula matches the model's norm output to 7.6e-6. The
      remaining in-loop residual comes from the model layer's eps-dominated
      gated norm: at tokens where RMS(x*silu(gate)) << sqrt(eps) the norm
      applies a ~1/sqrt(eps) ~ 316x gain, amplifying sub-ULP differences
      between our fp32 GEMM order and ATen's blocked GEMM (~1e-3, <=1 bf16
      ULP in the in_proj) into O(10) output differences. A real bug was also
      found and fixed: the norm used eps=1e-6 instead of the model's
      rms_norm_eps=1e-5, i.e. a sqrt(10) = 3.16x gain error at exactly those
      eps-dominated tokens. C_EPS is now 1e-5 (0x3727C5AC) and gen_golden uses
      1e-5. In-loop re-run with both fixes: PASSES (max abs 3.906e-03, max rel
      7.692e-03, 0/3840 outside 2e-2, no non-finite values). Unit is complete.
- [x] 9. Formal verification (`formal/bmc.sby`, depth 260, PASS)
- [x] 10. Update `description.yaml` — interface, Fmax, characteristics
- [x] 11. Clear `current_dut` field

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

- MATRIX_LANES = 4 rollout (branch opt/lanes-rollout): the in_proj and
  out_proj matrix_unit instances now compute four output rows per pass
  (bit-identical). Verified: lint clean; small-config unit test 32/32 PASS
  (max abs 0.0); in-loop full-config 5-token run max_abs 3.906e-03, max_rel
  7.692e-03, 0/3840 outside 2e-2, non-finite 0, in ~66 min (measured
  ~2.3M cycles/token, was ~5.1M).
