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

## Optimization task: branch `opt/matrix-unit`

- Goal: raise clock frequency (critical path is the binary32 MAC/fp_unit loop)
  and/or throughput (currently ~1 MAC-cycle per output with IN_FEATURES MAC
  cycles per output element), and reduce area where free.
- Hard constraint: the projection result must remain **bit-identical** to the
  current implementation (bf16 inputs/weights widened to fp32, sequential fp32
  accumulation in declaration order, bias last, one final bf16 rounding), so
  any parallel/reordered accumulation must be proven order-equivalent or is
  out of scope.
- Status: investigation complete (sub-agent report); implementation not started.

### Findings

- Baseline: description claims 200 MHz (hand estimate); the objective proxy on
  the pinned library is fp_add = 218 logic levels, fp_mul = 150, flattened
  matrix_unit = 224 (Yosys ltp, same recipe as the library metrics).
- True critical path: the fp32 accumulator loop
  add_y -> fp_add datapath -> add_y (not the weight read / multiplier, which is
  ~146 levels but becomes co-critical at full memory depth).
- Throughput today: 1 MAC/cycle, IN+3 cycles per output element, one row at a
  time. Per-token matrix cost is ~87.5 M cycles, of which the LM head
  (100352x768) alone is 77.4 M.

### Prioritized opportunities (all analysed for bit-accuracy)

| # | Change | Effect | Bit-exact | Risk |
|---|---|---|---|---|
| P1 | L parallel output lanes (L full MAC datapaths, different output rows) | throughput xL | yes (independent rows, same per-row order) | medium |
| P2 | Pipelined fp_add (P stages) + P accumulator contexts per lane | Fmax x1.5-2.5 | yes (registers only; same function) | high |
| P3 | Register multiplier operands + m_axis_tdata (RAM off the arith path) | removes RAM->mul path; enables SRAM | yes | low |
| P4 | Balanced leading-one detectors in fp_add/fp_mul | 5-15% Fmax | yes (same function) | medium |
| P5 | Direct fp_add/fp_mul instances (no fp_unit op mux); specialised exact bf16 multiplier | area + sim model | yes with caveats | medium |
| P6 | Overlap OUT with next row MAC; tie constant mode/rm | ~1-3% latency | yes (except 0+0 removal) | low |

Out of scope (violates bit-accuracy): any reordered/tree/K-split accumulation,
fp_fma fusion (differs on product underflow/overflow), wide exact accumulator,
skipping the 0+p0 seed (differs for -0).

### Chosen order

1. P3 (low risk, unblocks Fmax and a realistic weight memory)
2. P1 with L=4 (biggest throughput win; dimensions all divide by 4)
3. P2 (pipeline the adder + interleave P contexts) once P1/P3 are stable
4. P4 inside the fp submodule, with the library's exhaustive suites re-run

### Progress

- [x] P3 registered multiplier operands (MAC_FILL) — +1 cycle per output row,
      arithmetic schedule unchanged. Evidence: lint clean; unit 128/128
      bit-exact; in-loop q_proj 6144/6144 and k_proj 2048/2048 bit-exact
      (max abs 0.0); formal/bmc.sby depth 40 PASS. Busy-cycle count for the
      32x16 unit test (8 vectors): 4627 cycles (576/vector compute + stalls).
- [ ] P1 LANES parallel output lanes
