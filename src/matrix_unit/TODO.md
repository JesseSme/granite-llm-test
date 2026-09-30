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
- [x] P1 LANES parallel output lanes (default 1, bit-exact) — LANES complete
      fp32 MAC datapaths (own MUL/ADD fp_unit + accumulator) on different output
      rows, packed weight words (one read serves all lanes), per-row bias, and
      an out_buf STORE stage (data stable under backpressure). Rows keep the
      sequential fp32 add order of LANES=1 by construction; blocks stream lanes
      0..LANES-1 in row order and the final block masks OUT_FEATURES % LANES.
      Evidence: lint clean (default, LANES=4, tail, odd LANES); forced-rebuild
      unit tests: 32x16 LANES=1 128/128, 32x16 LANES=4 128/128, 32x6 LANES=4
      (tail) 48/48, all bit-exact with tlast checked; in-loop q_proj 6144/6144
      and k_proj 2048/2048 bit-exact (max abs 0.0); formal/bmc.sby and
      formal/bmc_lanes.sby (LANES=4, OUT=2<LANES) depth 40 PASS.
      Busy cycles (8 vectors, 32x16): LANES=1 4755, LANES=4 1299 (~3.7x);
      tail 32x6 LANES=4 643. Cycle model: per LANES-block
      IN+4+LANES cycles (MAC_FILL + IN MAC + ADD_LAST + BIAS + STORE + LANES
      output beats), i.e. IN+5 per row at LANES=1.
- [x] P2 pipelined fp_add (2 stages) + 2 interleaved accumulator contexts per
      lane — new fp-library module `fp_add_pipe2.sv`: register bank after the
      exponent alignment plus output register, documented 2-cycle latency,
      bit-identical to `fp_add` with OP_ADD/RM_RNE. `matrix_unit` now computes
      ROWSPW = 2*LANES rows per pass: each lane keeps two interleaved contexts
      (rows word*ROWSPW+lane and word*ROWSPW+LANES+lane) that submit on
      alternating MAC slots, so the MAC issue rate stays 1/cycle/lane; the
      adder output register itself is each context's accumulator (it is updated
      every other cycle), and one MUL per lane alternates products between the
      contexts. Weight words are packed ROWSPW wide, rows still stream in
      order, and the tail mask is generalized to OUT_FEATURES % ROWSPW.
      Evidence: fp_add_pipe2 vs fp_add equivalence harness (Verilator C++):
      20,000,000 biased-random vectors + zeros, 0 mismatches; lint clean
      (default, LANES=4, tail, odd LANES); unit tests bit-exact 32x16 LANES=1
      128/128, 32x16 LANES=4 128/128, 32x6 LANES=4 (tail) 48/48; busy cycles
      4755/1299/643, identical to P1 (same cycles per row by construction);
      in-loop q_proj 6144/6144 and k_proj 2048/2048 bit-exact (max abs 0.0);
      formal/bmc.sby and formal/bmc_lanes.sby depth 40 PASS (new anyseq stub
      `formal/fp_add_pipe2_stub.sv`); Yosys ltp -noff on the flattened
      matrix_unit (32x16, LANES=1, flatten before synth so the constant op
      mux prunes unused datapaths): 211 -> 137 levels (fp_add 208 -> fp_add_pipe2
      137), ~1.54x Fmax proxy.
- [x] P4 balanced leading-one detectors in fp_add/fp_mul (fp library, branch
      `opt/lzc-balanced` commit 2c013aa) — one-hot prefix reduction + constant
      mask encoder in bitlen/lpos/pbitlen, function-identical, no latency
      change. Evidence: 50M-vector old-vs-new equivalence (0 mismatches); library
      make smoke/regress PASS at all 6 widths; outer residual_adder 6144/6144,
      matrix_unit 128/128, RMSNorm PASS; fp_add ltp 218 -> 208.
- [x] P5 pipelined multiplier (fp library, branch `opt/fp-mul-pipe2` commit
      83f470e) — new `fp_mul_pipe2.sv`: same equations as `fp_mul` specialized
      to RM_RNE, register bank after the significand product + leading-one
      detection plus an output register (2-cycle latency, no rm/flags), same
      pattern as `fp_add_pipe2`. `matrix_unit` keeps the registered q operands
      (P3) and shifts the prefetch one slot earlier: operands loaded at slot d
      appear at mul_y in slot d+3, so even slots load context B and odd slots
      context A with element index ceil(d/2); the MAC slot count, output order
      and busy-cycle model are unchanged. Evidence: library lint clean; 20M
      biased-random-vector equivalence vs `fp_mul`(RNE) with the 2-cycle delay
      modelled (0 mismatches, includes zeros/inf/NaN/subnormal/bf16-widened
      patterns); unit tests bit-exact 128/128 LANES=1, 128/128 LANES=4, 48/48
      tail; in-loop q_proj 6144/6144 and k_proj 2048/2048 bit-exact (max abs
      0.0); formal/bmc.sby + formal/bmc_lanes.sby depth 40 PASS (new
      `formal/fp_mul_pipe2_stub.sv`); Yosys ltp -noff (32x16 LANES=1, flattened
      before synth): fp_mul 148 -> fp_mul_pipe2 74 levels, flattened
      matrix_unit stays 137 (the fp_add_pipe2 accumulator is now the sole
      critical path).

## Next optimization target: realize the matrix gains in the dependent units

Branch to create: `opt/lanes-rollout` (from main).

Why: matrix_unit now offers LANES (parallel output rows, bit-exact) and a
2-stage pipelined adder (P2), but every consumer still uses the default
LANES=1, so the end-to-end gains are dormant. The linear layers dominate a
token; pre-optimization matrix cycles per token: LM head 77.4M, gate+up
3.16M, in_proj 2.60M, down 1.58M, q/o 1.19M, out_proj 1.18M, k/v 0.40M.

Steps:
1. Set LANES=4 at every consumer: attention_unit (Q/K/V/O), SwiGLU_unit
   (gate+up and down), mamba2_unit (in_proj/out_proj), output_projection_unit.
   Prefer a parent-level localparam so the parameter is visible per unit.
   Do NOT change any interface; the matrix AXI handshake already hides the
   extra internal latency (its outputs are bit-identical).
2. Expected: matrix cycles per instance divide by ~4. Evidence from the
   matrix unit test (8 vectors, busy cycles): 4755 -> 1299 at LANES=4.
   LM head 77.4M -> ~19.4M; gate+up 3.16M -> ~0.8M; in_proj 2.60M -> ~0.65M.
3. Verification per unit: `verilator --lint-only -Wall` clean; the unit
   golden test bit-exact (`run_test.py`); one in-loop per unit (recommended;
   the unit goldens are the hard bit-exact gate, the in-loop proves it
   against the real model). Those in-loop runs use real layer weights and
   take a while - run ONE at a time (16 GB RAM, one build/sim at a time).
4. Keep LANES=1 as the matrix_unit default and keep the small-config unit
   tests unchanged; only the DUT configurations should use LANES=4.
5. If some unit is not bit-exact at LANES=4, stop and debug the lane mapping
   there (packed weight words store LANES consecutive rows per word; lane j
   must compute row word*LANES + j with its own accumulator).

Follow-up after the rollout: pipeline the multiplier. DONE (see the P5 entry
above): `fp_mul_pipe2` landed in the fp library (commit 83f470e), cuts the
multiplier from 148 to 74 ltp levels and leaves the flattened matrix_unit at
137 (adder-limited, busy cycles unchanged).

Constraints: bit-exactness is absolute; no interface changes; update
description.yaml/TODO per unit as they land; run one heavy build at a time.
