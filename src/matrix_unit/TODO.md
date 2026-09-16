# matrix_unit — TODO

## Code Review Checklist

- [x] RTL passes Verilator lint with `-Wall` (no warnings).
- [x] cocotb testbench passes against golden sample for all test vectors
      (128/128 bit-exact with random AXI-Stream bubbles/stalls).
- [x] In-loop test passes — q_proj 768x768: 6144/6144 bit-exact; k_proj
      768x256: 2048/2048 bit-exact (max abs error 0.0 both).
- [x] `description.yaml` is updated with interface, Fmax, and characteristics.
- [x] No hardcoded magic numbers — all sizes/format parameters.
- [x] Follows existing code conventions (see `systemverilog_fp_unit/rtl/`).

## Implementation Steps

- [x] 1. Set `current_dut: matrix_unit` in description.yaml
- [x] 2. Implement `matrix_unit.sv` (fp32 MAC pipeline, AXI-Stream in/out, weight load)
- [x] 3. Lint with Verilator (`verilator --lint-only -Wall`)
- [x] 4. Generate golden sample (`gen_golden.py`, sequential fp32 + bias last)
- [x] 5. Write cocotb testbench (`tb_matrix_unit.py`) and parameterized runner
- [x] 6. Run RTL simulation — 128/128 bit-exact
- [x] 7. In-loop verification with real Granite Linear layers — 100% bit-exact
      Run: `.venv/bin/python run_test.py inloop`
- [x] 8. Formal verification — `formal/bmc.sby` PASS (BMC depth 40)
- [x] 9. Update `description.yaml` — interface, Fmax, characteristics
- [x] 10. Clear `current_dut` field
