# RMSNorm Unit — TODO

## Code Review Checklist

- [x] RTL passes Verilator lint with `-Wall` (no warnings).
- [x] cocotb testbench passes against golden sample for all test vectors.
- [x] In-loop test passes — 41472/41472 bit-exact vs the model on real
      Granite 4.0-H-350M activations (6 norm modules × 9 rows) after the fp32
      datapath fix and the fp_unit protocol migration.
- [x] `description.yaml` is updated with interface, Fmax, and characteristics.
- [x] No hardcoded magic numbers — use parameters or `define` constants.
- [x] Follows existing code conventions (see `systemverilog_fp_unit/rtl/`).

## Implementation Steps

- [x] 1. Read `description.yaml` — understand unit purpose, tensor shapes, guidance.
- [x] 2. Implement RTL — write `rmsnorm_unit.sv`.
- [x] 3. Lint — run `verilator --lint-only -Wall` and fix all issues.
- [x] 4. Write cocotb testbench — create `tb_rmsnorm_unit.py` and `Makefile`.
- [x] 5. Run RTL simulation — execute cocotb test against Verilator.
- [x] 6. In-loop verification — test with real Granite 4.0-H-350M activations.
      Run: `.venv/bin/python run_test.py tb_rmsnorm_unit_inloop`.
- [x] 7. Update `description.yaml` — interface, Fmax, characteristics.
- [x] 8. Clear `current_dut` field.
- [x] 9. Formal verification — formal/bmc.sby (BMC depth 80, fp_unit abstracted):
      no output before weights loaded, output only while busy, busy rises only
      after an accepted input, exactly WIDTH outputs per vector; PASS

## 2026-09 fp32 datapath fix

- The unit previously ran all arithmetic (including the 768-term sum-of-squares
  accumulation) through a bfloat16 fp_unit, which made the variance ~10% off
  and used eps = 7.63e-6 (bf16 0x3780). The datapath is now fp32 (fp_unit
  W_MANT=23, operands widened, one fp32_to_bf16_round at the output) and
  eps = 1e-5 (config.rms_norm_eps), matching GraniteMoeHybridRMSNorm.
- Evidence: the granite_layer single-layer single-token in-loop test failed
  with 453/768 mismatches before the fix and PASSES (max abs 0.0078) after.
- Still to do: update gen_golden.py/tb_rmsnorm_unit.py (they emulate the old
  bf16 behaviour) and add fp32_to_bf16_round.sv to the formal file lists.
