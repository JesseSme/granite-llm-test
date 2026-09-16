# SSM_unit — TODO

## Code Review Checklist

- [x] RTL passes Verilator lint with `-Wall` (no warnings; checked at 2/2/4 and
      48/32/128).
- [x] cocotb testbench passes against golden sample for all test vectors
      (16/16 within tolerance, max abs/rel 1e-6; `tb_controlled.py` directed
      case within 1e-3).
- [x] In-loop test passes — real Granite Mamba2 SSM scan (0/13824 outside
      1e-3/1e-3; max rel 5.14e-6 vs the model's chunk_scan).
- [x] `description.yaml` is updated with interface, Fmax, and characteristics.
- [x] No hardcoded magic numbers — use parameters or `define` constants
      (fp32 constants are named localparams in the sequential FP units).
- [x] Follows existing code conventions (see `systemverilog_fp_unit/rtl/`).

## Implementation Steps

- [x] 1. Set `current_dut: ssm_unit` in description.yaml
- [x] 2. Implement `fp_softplus_seq.sv` (softplus via exp + atanh series)
- [x] 3. Implement `ssm_unit.sv` (fp32 recurrent S6 step per token)
- [x] 4. Lint with Verilator (`verilator --lint-only -Wall`)
- [x] 5. Generate golden sample (`gen_golden.py`, fp32 recurrence reference)
- [x] 6. Write cocotb testbench (`tb_ssm_unit.py`) and runner
- [x] 7. Run RTL simulation against golden sample
- [x] 8. In-loop verification with real Granite Mamba2 inputs
- [x] 9. Formal verification (`formal/bmc.sby`)
- [x] 10. Update `description.yaml` — interface, Fmax, characteristics
- [x] 11. Clear `current_dut` field

## Notes / issues found

- The shared LUT-based `fp_exp` (softmax_unit) has ~20% relative error — far
  too coarse for the SSM recurrence. Replaced by `fp_exp_seq.sv`: range
  scaling (x/32), degree-6 Horner polynomial, 5 squarings, |x| clamped to 16.
  Measured max relative error 7.6e-5 (worst case |x|=16), ~2e-5 typical.
- `fp_softplus_seq.sv` was free-running out of idle (the step increment was
  outside the `step == 0` guard) and always computed softplus(0). Fixed; now
  the unit stays at step 0 until `start`.
- A parallel reset loop over `h_state` (196k words at full size) makes Yosys
  unroll 196k assignments and exhaust memory (~6 GB, `bad_alloc`) during
  `read_verilog`, before formal even starts. Replaced with a sequential clear
  state (one word per cycle); `s_axis_tready` stays low until it completes.
  State is now a flat 1-D array with a computed index.
- SBY has no memory-limit option: cap it externally (`ulimit -v`, cgroup) and
  set `timeout` in `[options]` to bound runaway proofs.
- Formal specifics: Yosys rejects `->` implications (use `!a || b`), needs
  `initial assume(!rst_n)` for a defined start state, and the framing
  assumptions must force `tlast` exactly on the final frame beat (otherwise
  the RTL's accept-without-tlast path breaks the input counter model).
