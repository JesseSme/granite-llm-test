# residual_adder_unit — TODO

## Code Review Checklist

- [x] RTL passes Verilator lint with `-Wall` (no warnings).
- [x] cocotb testbench passes against golden sample for all test vectors
      (6144/6144 bit-exact vs torch model semantics, including an idle gap).
- [x] In-loop test passes — 52224/52224 bit-exact vs the model tensors
      (100.0%, max abs/rel error 0.0) on real mixer/MLP residual pairs.
- [x] `description.yaml` is updated with interface, Fmax, and characteristics.
- [x] No hardcoded magic numbers — named constants (RESIDUAL_MULT_FP32).
- [x] Follows existing code conventions (see `systemverilog_fp_unit/rtl/`).

## Implementation Steps

- [x] 1. Set `current_dut: residual_adder_unit` in description.yaml
- [x] 2. Implement `residual_adder_unit.sv`
- [x] 3. Lint with Verilator (`verilator --lint-only -Wall`)
- [x] 4. Generate golden sample (`gen_golden.py`, model semantics)
- [x] 5. Write cocotb testbench (`tb_residual_adder_unit.py`) and runner
- [x] 6. Run RTL simulation — 6144/6144 bit-exact
- [x] 7. In-loop verification with real Granite activations — 52224/52224 exact
      Run: `.venv/bin/python run_test.py tb_residual_adder_unit_inloop`
- [x] 8. Formal verification — `formal/bmc.sby` PASS (BMC depth 10)
- [x] 9. Update `description.yaml` — interface, Fmax, characteristics
- [x] 10. Clear `current_dut` field
- [x] 11. Upgrade datapath to binary32 + single bf16 rounding per op to match the
      model bit-exactly (replaces the bf16(0.246)-constant version, which had
      1-ULP deviations on ~4% of elements); new `fp32_to_bf16_round.sv` helper
