# Agent Instructions - tinyllm

## Overview

SystemVerilog implementation of the Granite 4.0-H-350M LLM layers, unit by unit,
each verified against the Python model. **All units in
`LLM_LAYER_DESCRIPTION.md` are implemented, verified, and committed**, plus a
single-layer integration wrapper and a hybrid end-to-end test.

Model architecture: 32 decoder layers with `config.layer_types` =
['mamba' x10, 'attention', 'mamba' x2, 'attention', 'mamba' x3, 'attention',
'mamba' x7, 'attention', 'mamba' x4] (attention at indices 10, 13, 17, 27;
28 mamba layers). hidden_size 768, attention/intermediate 1536, shared MLP
intermediate 2048, GQA 12/4 heads x 64, mamba num_heads 48, head_dim 32,
d_state 128, conv kernel 4, chunk 256, vocab 100352 (tied LM head).

Key config constants (must appear exactly in the RTL):
`rms_norm_eps` 1e-5, `residual_multiplier` 0.246, `logits_scaling` 3
(the model **divides** logits by it), `embedding_multiplier` 12,
`attention_multiplier` 0.015625, `mamba_proj_bias`/`attention_bias` false,
`mamba_conv_bias` true, `time_step_limit` none.

## Unit inventory and status

| Unit (`src/...`) | Status |
|---|---|
| `embedding_lookup_unit` | verified: unit + in-loop 16896/16896 bit-exact; formal PASS |
| `residual_adder_unit` | verified: unit 6144/6144, in-loop 52224/52224 bit-exact; fp32 datapath with 0.246 built in; formal PASS |
| `matrix_unit` | verified: in-loop q/k_proj bit-exact; formal PASS; rare (<0.1%) 1-ULP ATen blocked-GEMM order differences documented. Optimized bit-exactly: LANES parallel output rows (8-vector busy cycles 4755 -> 1299 at LANES=4), P2 2-stage pipelined adder + 2 interleaved accumulator contexts, operand-register pipeline; flattened ltp 221 -> 137 (~1.54x Fmax proxy) |
| `SSM_unit` | verified: unit, in-loop max abs 4.5e-8; formal depth 140 PASS |
| `attention_unit` | verified: unit 1536/1536 bit-exact, in-loop bit-exact vs layer-10 eager attention; formal depth 220 PASS |
| `mlp_unit` (`SwiGLU_unit/`) | verified: unit bit-exact, in-loop bit-exact + 0/3072 outside 2e-2 vs model; formal depth 260 PASS |
| `mamba2_unit` | verified: full-config unit 768/768 bit-exact, in-loop 0/3840 outside 2e-2; formal depth 260 PASS |
| `conv1d_unit` | verified: unit 49152/49152 bit-exact, in-loop 13824/13824 bit-exact; formal depth 120 PASS |
| `RMSNorm_unit` | verified: unit + in-loop PASS after the fp32 datapath fix; its formal files still need `fp32_to_bf16_round.sv` added |
| `softmax_unit` | verified: in-loop 3888/3888 rows, formal depth 120 PASS; LUT exp is coarse (max rel 0.209) - attention uses `attn_softmax_seq` (accurate) instead |
| `sigmoid_unit`, `SiLU_unit` | verified: formal depth 10 PASS; LUT-based approximations (<0.5% sigmoid error), not used on fp32-critical paths |
| `output_projection_unit` | verified: unit bit-exact 256/256, in-loop bit-exact 1024/1024 on 512 sampled vocab rows (the 77M-param table cannot be simulated in full), formal depth 60 PASS |
| `granite_layer` | verified: single-token real-weight in-loop PASS (max abs 0.0078); end-to-end hybrid PASS |
| `fp_unit` | spec only - implementation is the external `systemverilog_fp_unit/` git repository, which also provides `fp_add_pipe2` (the 2-stage pipelined fp32 adder used by matrix_unit) |

## Numerics rules (learned the hard way)

1. **Match the model's compute precision, not just its formula.** Where the
   model computes in fp32 (dot products, variance, convolution, gate
   statistics), instantiate `fp_unit` with fp32 dimensions
   (`W_EXP=8, W_MANT=23`), widen bfloat16 operands exactly with the 16 bits in
   the HIGH half (`{v, 16'h00}`), and round to bfloat16 ONCE at the output
   (`fp32_to_bf16_round`). Instantiating the bf16 `fp_unit` (`W_MANT=7`) for
   fp32 math caused the two biggest bugs in this project: `conv1d_unit`
   (per-MAC bf16 rounding) and `rmsnorm_unit` (768-term bf16 sum-of-squares,
   ~10% variance error) - both surfaced only in full-config in-loop tests and
   were invisible to unit tests whose goldens emulated the RTL instead of the
   model.
2. **Golden samples must come from the model's torch semantics**, never from a
   re-implementation of what the RTL currently does.
3. **eps and scale constants must equal the config values** (1e-5 for
   rms_norm_eps; 0.246 for the residual multiplier; division by 3 for
   logits_scaling). Wrong eps only shows up where a statistic is
   eps-dominated; it was a 3.16x gain error in `mamba2_unit`.
4. **Widen bf16 to fp32 in the high half** or NaN/inf become denormals and
   silently pass tolerance checks.
5. **Tolerance checks must be NaN-aware**: `abs_err > tol and rel_err > tol`
   plus an explicit non-finite count; Python `max(x, nan)` hides NaN.
6. Registered `fp_unit` outputs mean operands must be held through wait
   states, or latch each intermediate in a register.

## Simulation feasibility

Measured cost (Verilator, ~3-6k cycles/s wall):

| Path | Cycles |
|---|---|
| weight load | 1 beat/cycle (mamba2 layer ~3.8M, full decoder layer ~8.5M) |
| SSM per token | ~1.19M |
| attention per token | ~1.6M |
| MLP per token | ~4.8M |
| mamba2 layer per token | ~5.1M |
| full 32-layer token | ~169M (~8-15 h) + ~350M weight-load beats |

A full 32-layer run is therefore **not** simulatable. Use `granite_layer`
(one layer, one token, real weights) and `tb_granite_layer_e2e.py` (one RTL
layer + the other 31 layers in software, comparing final logits), which is the
accepted end-to-end verification. Latest result: same argmax and top-5
prediction as the software baseline. Note: the matrix_unit optimizations (LANES=4,
P2 pipelined adder) reduce the ~87.5M matrix cycles per token to roughly a
quarter; the per-path cycle estimates above predate them.

## Open-Source Toolchain

The `oss-cad-suite/` directory at the project root provides all required
tools (Verilator, Yosys, SymbiYosys/SBY, cocotb support). It is not tracked;
the build system is vendored as the `oss-cad-suite-build/` submodule
(https://github.com/yosyshq/oss-cad-suite-build) - unpack a release tarball
there or build the suite from it. Activate before running any tool:

```bash
source /path/to/tinyllm/oss-cad-suite/environment
```

Use `.venv/bin/python` for all Python (zsh does not
word-split unquoted variables - use `$=VAR` or explicit arguments).

| Tool | Purpose | Command |
|------|---------|---------|
| **Verilator** | RTL simulation, linting | `verilator --lint-only -Wall ...` |
| **Yosys** | Synthesis, linting | `yosys -p "read_verilog -lint *.sv"` |
| **SymbiYosys (sby)** | Formal property verification | `sby -f config.sby` |
| **cocotb** | Python-driven RTL testbenches | per-unit `run_test.py` (see below) |

### Running tests

Plain `make` for cocotb is broken (`No GPI_USERS`). Every unit has its own
`run_test.py` (cocotb_tools runner + a `_sim_env`/`_run_attempt` retry helper,
`COCOTB_TEST_MODULES`, temp log in `/tmp/tmp*.log`):

```bash
cd src/<unit>
python run_test.py                 # unit test
python run_test.py inloop          # in-loop test (real model activations)
python run_test.py tb_<unit>_inloop  # conv1d/RMSNorm style: module argument
```

Launch long runs with `nohup ... > <name>.log 2>&1 &`
(the repo root, not /tmp) and poll. `granite_layer` also has an `e2e` mode.

### Linting

```bash
verilator --lint-only -Wall src/<unit>/<unit>.sv
```

Fix all warnings; a clean lint pass is required before formal or simulation.
Note `HW+1'(x)` parses as `HW + 1'(x)` - parenthesize widths
(`localparam int CW = HW + 1; CW'(x)`).

### Formal Verification (SymbiYosys)

Distilled rules from all units' `formal/bmc.sby` runs:

- `blackbox` is rejected by `hierarchy -smtcheck`; use `(* anyseq *)` inputs or
  constant/handshake-exact stubs in `formal/<unit>_stub.sv`.
- `smtbmc boolector` is much faster than the default z3; set
  `[engines] smtbmc boolector`.
- No `assert property`, no `->` implication (use `!a || b`).
- Add `initial assume(!rst_n)`.
- Gate assertions on an observed reset (`logic seen_rst = 1'b0;` updated with
  `seen_rst | !rst_n`) so free initial register values cannot fail them.
- Tie off unused harness inputs (an undriven input is free and trips
  "signal never driven by DUT" assertions).
- Set `timeout` in `[options]`; no memory-limit option, wrap with
  `bash -c 'ulimit -v 6000000; sby -f bmc.sby'`.
- `abc` engine needs an `[abc]` section; `[files]` paths are relative to the
  formal dir (`../<unit>.sv`), while `[script]` reads the copied basenames.
- Yosys explodes on parallel constant reset loops over big arrays; use a
  sequential clear state.

## Working on a Unit - Step by Step

1. Set `current_dut: <unit>` in that unit's `description.yaml` (clear it when
   the unit is done).
2. Read `description.yaml` (`llm_guidance`) and the model source in
   `.venv/.../transformers/models/granitemoehybrid/modeling_granitemoehybrid.py`
   - the model is the source of truth.
3. Implement the RTL; lint clean.
4. Generate the golden sample from the model's torch semantics
   (`gen_golden.py`), not from an RTL emulation.
5. Write/run the cocotb unit test.
6. In-loop: load the real checkpoint
   (`AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16,
   attn_implementation="eager")`), capture activations with hooks
   (`with_kwargs=True`; never `.float()` the captured bf16) and compare.
7. Update `description.yaml` (interface, max_clock_speed, characteristics,
   verification results) and `TODO.md`.
8. Commit the unit (one commit per unit, message includes the verification
   evidence).

## Golden Sample Generation

The Python model is the source of truth. Capture activations with
`torch.no_grad()` and hooks; save bf16 values as 4-hex-digit words
(`t.view(torch.uint16)`), and read them back placing the bits in the high half.

## Code Review Checklist

- [ ] RTL passes Verilator lint with `-Wall` (no warnings).
- [ ] cocotb unit test passes against a model-derived golden sample.
- [ ] In-loop test passes with real model activations/weights.
- [ ] Formal verification passes (or the gap is documented in the description).
- [ ] `description.yaml` updated (interface, Fmax, characteristics, results).
- [ ] No hardcoded magic numbers (use parameters/localparams).
- [ ] Compute precision matches the model (see Numerics rules).
- [ ] Follows existing code conventions (see `systemverilog_fp_unit/rtl/`).

## Remaining Work

- **Migrate the outer units to upstream's pipelined `fp_unit`** (new per-op
  latencies + in_valid handshake; `ITER_DIVSQRT` default 1). Until this is
  done the outer repo stays pinned at library commit `6907045`; see
  `src/fp_unit/description.yaml` (`migration_note`). The merged library works
  on the library side (conflict resolution committed as `0a75e4c`).
- Next optimization target (recorded in `src/matrix_unit/description.yaml` as
  `current_dut` + `src/matrix_unit/TODO.md`): roll LANES=4 out to every
  matrix_unit consumer, then pipeline the multiplier (`fp_mul_pipe2`).
- Attention-type variant of `granite_layer` (GraniteMoeHybridAttention layers
  at indices 10, 13, 17, 27).
- Wrapper-level formal properties for `granite_layer`.
- Audit remaining units for bf16-vs-fp32 datapath gaps (`softmax_unit` LUT exp
  is the known coarse one; attention already uses the accurate variant).
- Fix the stale "mamba2 unit" PASSED label printed by
  `granite_layer/run_test.py`.
- Optional: multi-token runs of `granite_layer` and the hybrid e2e test.

## Notes

- `systemverilog_fp_unit/` is a git submodule
  (`git@github.com:JesseSme/systemverilog_fp_unit.git`); changes inside it must
  be committed and pushed in that repository first, then the updated pointer
  committed here (the outer repo only records the pinned commit).
- `.gitignore` excludes `oss-cad-suite/`, `granite-4.0-h-350m/`, `sim_build*/`,
  `results.xml`, `**/formal/bmc*/`, `__pycache__/`, `.venv`.
