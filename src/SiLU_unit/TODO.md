# SiLU_unit — TODO

- [x] Set `current_dut: silu_unit` in description.yaml
- [x] Implement silu_unit.sv (instantiates sigmoid_unit)
- [x] Lint with Verilator (verilator --lint-only -Wall)
- [x] Generate golden sample (Python reference)
- [x] Write cocotb testbench tb_silu_unit.py
- [x] Run RTL simulation and verify against golden sample (341/341 passed)
- [x] Write in-loop test tb_silu_unit_inloop.py
- [x] Run in-loop test with real model activations (200/200 passed)
- [x] Update description.yaml with interface, Fmax, characteristics
- [x] Clear current_dut from description.yaml
- [x] Formal verification — formal/bmc.sby (BMC depth 10): valid_out is
      valid_in delayed by exactly two clocks; PASS
- [x] Fmax estimate filled in (150 MHz, Yosys logic-depth ratio vs sigmoid_unit)
