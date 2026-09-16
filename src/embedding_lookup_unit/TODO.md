# embedding_lookup_unit — TODO

## Code Review Checklist

- [x] RTL passes Verilator lint with `-Wall` (no warnings).
- [x] cocotb testbench passes against golden sample for all test vectors
      (12288/12288 bit-exact).
- [x] In-loop test passes — 16896/16896 bit-exact on real Granite token IDs and
      embedding rows (max abs error 0.0).
- [x] `description.yaml` is updated with interface, Fmax, and characteristics.
- [x] No hardcoded magic numbers — parameters plus named constants (SCALE_12).
- [x] Follows existing code conventions (see `systemverilog_fp_unit/rtl/`).

## Implementation Steps

- [x] 1. Set `current_dut: embedding_lookup_unit` in description.yaml
- [x] 2. Implement `embedding_lookup_unit.sv` (table RAM + load port + bf16 ×12)
- [x] 3. Lint with Verilator (`verilator --lint-only -Wall`)
- [x] 4. Generate golden sample (`gen_golden.py`)
- [x] 5. Write cocotb testbench (`tb_embedding_lookup_unit.py`) and runner
- [x] 6. Run RTL simulation — 12288/12288 bit-exact
- [x] 7. In-loop verification with real Granite token IDs — 16896/16896 bit-exact
      Run: `.venv/bin/python run_test.py tb_embedding_lookup_unit_inloop`
- [x] 8. Formal verification — `formal/bmc.sby` PASS (BMC depth 32)
- [x] 9. Update `description.yaml` — interface, Fmax, characteristics
- [x] 10. Clear `current_dut` field
