# softmax_unit — TODO

- [x] Set `current_dut: softmax_unit` in description.yaml
- [x] Implement softmax_unit.sv
- [x] Lint with Verilator (verilator --lint-only -Wall)
- [x] Generate golden sample (Python reference)
- [x] Write cocotb testbench tb_softmax_unit.py
- [x] Run RTL simulation and verify against golden sample
- [x] Write in-loop test tb_softmax_unit_inloop.py
- [x] Run in-loop test with real model activations — rewritten to capture actual
      attention softmax operands (eager attn + patched F.softmax); 3888/3888 rows
      passed, max abs error 0.005668, max row-sum error 0.000000. The previous
      version replayed the unit-test golden vectors.
      Run: `.venv/bin/python run_test.py tb_softmax_unit_inloop`.
- [x] Update description.yaml with interface, Fmax, characteristics
- [x] Clear current_dut from description.yaml
- [x] Formal verification — formal/bmc.sby (BMC depth 120, fp_unit/fp_exp
      abstracted): done always coincides with the final valid_out; PASS
