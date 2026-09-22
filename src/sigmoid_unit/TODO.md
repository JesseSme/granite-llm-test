# sigmoid_unit — TODO

- [x] Set `current_dut: sigmoid_unit` in description.yaml
- [x] Implement sigmoid_unit.sv
- [x] Lint with Verilator (verilator --lint-only -Wall)
- [x] Generate golden sample (Python reference)
- [x] Write cocotb testbench tb_sigmoid_unit.py
- [x] Run RTL simulation and verify against golden sample
- [x] Write in-loop test tb_sigmoid_unit_inloop.py
- [x] Run in-loop test with real model activations
- [x] Update description.yaml with interface, Fmax, characteristics
- [x] Clear current_dut from description.yaml
- [x] Formal verification — formal/bmc.sby (BMC depth 10): valid_out is
      valid_in delayed by exactly one clock; PASS
