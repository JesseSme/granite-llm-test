# TODO — conv1d_unit

## Code Review Checklist

- [x] RTL passes Verilator lint with `-Wall` (no warnings).
- [x] cocotb testbench passes against golden sample for all test vectors.
- [x] In-loop test passes — 13824/13824 outputs (9 timesteps × 1536 channels) within
      0.1 abs of the recomputed causal conv on real layer-0 Mamba projected states;
      max abs error 0.03125.
- [x] `description.yaml` is updated with interface, Fmax, and characteristics.
- [x] No hardcoded magic numbers — use parameters or `define` constants.
- [x] Follows existing code conventions (see `systemverilog_fp_unit/rtl/`).

## Implementation Steps

- [x] 1. Set `current_dut: conv1d_unit` in description.yaml
- [x] 2. Implement `conv1d_unit.sv`
- [x] 3. Lint with Verilator (`verilator --lint-only -Wall`)
- [x] 4. Generate golden sample (`gen_golden.py`)
- [x] 5. Write cocotb testbench (`tb_conv1d_unit.py`) and Makefile
- [x] 6. Run RTL simulation — 49152/49152 passed, max error 0.0625
- [x] 7. Write in-loop test (`tb_conv1d_unit_inloop.py`).
- [x] 8. Re-verified after the fp32 accumulation fix: unit test bit-exact
      (49152/49152, max abs error 0.0), formal BMC depth 120 PASS, in-loop
      test against the real layer-0 projected states PASSES with the same
      13824/13824 elements (now with fp32 accumulation and one bf16 round).
      Run: `.venv/bin/python run_test.py tb_conv1d_unit_inloop`.
- [x] 8. Update `description.yaml` with interface, Fmax, characteristics
- [x] 9. Clear `current_dut` and update TODO.md
- [x] 10. Formal verification — formal/bmc.sby (BMC depth 120, fp_unit
      abstracted): output requires an accepted input, out_count <= in_count,
      at most one input outstanding; PASS
