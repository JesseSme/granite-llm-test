# tinyllm — Granite 4.0-H-350M decoder layers in SystemVerilog

This project implements the layers of the Granite 4.0-H-350M LLM in
SystemVerilog, unit by unit, and verifies every unit against the PyTorch
reference model (`transformers`). The goal is a register-based hardware
implementation with the model weights baked into the design.

Model: Granite 4.0-H-350M is a hybrid decoder with 32 layers —
28 Mamba2 layers and 4 attention layers (indices 10, 13, 17, 27).
`hidden_size` 768, attention/MLP intermediate 1536/2048, GQA 12/4 heads x 64,
Mamba num_heads 48 / head_dim 32 / d_state 128 / conv kernel 4,
vocab 100352 (LM head tied to the embedding table).

All units from `LLM_LAYER_DESCRIPTION.md` are implemented, verified, and
committed, plus a single-decoder-layer wrapper (`granite_layer`) and a hybrid
end-to-end test (one RTL layer + the remaining 31 layers in software).

## Repository layout

```
src/<unit>/                 one directory per verified unit, containing
  <unit>.sv                   the RTL
  gen_golden.py               golden-sample generator (torch reference)
  tb_<unit>.py                cocotb unit testbench
  tb_<unit>_inloop.py         cocotb in-loop test (real model activations)
  run_test.py                 build + run helper for this unit
  golden_*.hex                golden samples (bf16 hex words)
  formal/                     SymbiYosys BMC config, properties, stubs
  description.yaml            interface, Fmax, characteristics, results
  TODO.md                     implementation/verification checklist
src/granite_layer/          one mamba-type decoder layer + tests
  granite_layer.sv            input_layernorm -> mamba2 -> residual (+0.246)
                              -> post_attention_layernorm -> MLP -> residual
  tb_granite_layer_inloop.py  single layer, single token, real weights
  tb_granite_layer_e2e.py     one RTL layer + 31 layers in software
scripts/                    experiment / exploration scripts
results/                    JSON outputs of those experiments
tests/                      pytest suite (software model checks)
systemverilog_fp_unit/      floating-point unit library (git submodule)
oss-cad-suite-build/        OSS CAD Suite build system (git submodule,
                            https://github.com/yosyshq/oss-cad-suite-build)
oss-cad-suite/              open-source toolchain (Verilator, Yosys, SBY;
                            not tracked - install from the submodule above)
granite-4.0-h-350m/        model checkpoint (weights + tokenizer)
AGENTS.md                   development guide, numerics rules, SBY gotchas
LLM_LAYER_DESCRIPTION.md    the original unit list / specification
```

The per-unit Python files (`gen_golden.py`, `tb_*.py`, `run_test.py`) are kept
next to their RTL on purpose: they reference the RTL and golden files with
paths relative to the unit directory.

## Requirements

- **Toolchain** — the OSS CAD Suite build system is vendored as the
  `oss-cad-suite-build/` submodule
  (https://github.com/yosyshq/oss-cad-suite-build); grab a release tarball
  from its releases page (or build the suite from it) and unpack it as
  `oss-cad-suite/` in the project root. That directory (Verilator, Yosys,
  SymbiYosys/SBY, cocotb support) is not tracked. Activate before running:
  ```bash
  source oss-cad-suite/environment
  ```
- **Python** — the project venv (created from `pyproject.toml` with `uv`),
  providing `cocotb`, `cocotb-tools`, and for the in-loop tests
  `transformers`, `torch`, `safetensors`:
  ```bash
  .venv/bin/python --version      # Python 3.13
  uv sync --extra test                               # (re)install deps
  ```
  zsh does not word-split unquoted variables: use explicit paths.
- **Model checkpoint** — `granite-4.0-h-350m/` (Hugging Face
  `ibm-granite/granite-4.0-h-350m`), needed only for the in-loop tests.
- **FP library** — `systemverilog_fp_unit/` is a git submodule
  (`git@github.com:JesseSme/systemverilog_fp_unit.git`). Clone with
  `git clone --recurse-submodules ...` or run
  `git submodule update --init` after cloning.

## How to run

### Lint a unit (required to be warning-free)

```bash
source oss-cad-suite/environment
verilator --lint-only -Wall src/<unit>/<unit>.sv
```

### Unit test (small synthetic config, golden sample from torch)

```bash
cd src/matrix_unit   && .venv/bin/python run_test.py
cd src/conv1d_unit   && .venv/bin/python run_test.py tb_conv1d_unit_inloop
```

Conventions: most units default to the unit test; `SSM_unit`, `mamba2_unit`
and `granite_layer` accept a mode argument (`unit`, `inloop`, and `e2e` for
`granite_layer`); `conv1d_unit`/`RMSNorm_unit` take the test module name.

### In-loop test (real weights + activations from the model)

Runs the actual checkpoint forward pass, captures activations with hooks and
compares the DUT output. These are long (the weight-load and MAC costs below);
launch them in the background and poll.

```bash
cd src/mamba2_unit    && .venv/bin/python run_test.py inloop
cd src/SSM_unit       && .venv/bin/python run_test.py inloop
```

Parameter overrides for larger/smaller configs use per-unit environment
variables (e.g. `MAMBA_HIDDEN`, `MAMBA_INTER`, `MAMBA_HEADS`, `MAMBA_HEAD_DIM`,
`MAMBA_D_STATE` for `mamba2_unit`; `OUTPROJ_HIDDEN`, `OUTPROJ_VOCAB` for
`output_projection_unit`). `output_projection_unit` verifies a sampled subset
of the 100352 vocabulary rows (the full 77M-parameter table cannot be
simulated).

### Single-layer and hybrid end-to-end

```bash
cd src/granite_layer
.venv/bin/python run_test.py inloop   # layer 0, one token
.venv/bin/python run_test.py e2e     # RTL layer + 31 SW layers
```

The `e2e` mode streams layer 0 through the RTL, substitutes its output into a
software forward pass for the remaining 31 layers + LM head, and compares the
final logits (and top-5 prediction) against the software baseline.

### Formal verification (SymbiYosys)

```bash
cd src/<unit>/formal
bash -c 'ulimit -v 6000000; sby -f bmc.sby'
```

### Software model / experiment checks

```bash
.venv/bin/python -m pytest tests/
.venv/bin/python scripts/<experiment>.py   # writes results/*.json
```

## Verification status (summary)

| Unit | Result |
|---|---|
| `embedding_lookup_unit` | unit + in-loop 16896/16896 bit-exact; formal PASS |
| `residual_adder_unit` | unit 6144/6144, in-loop 52224/52224; formal PASS (fp32 + 0.246) |
| `matrix_unit` | in-loop q/k proj bit-exact; formal PASS (rare 1-ULP ATen order) |
| `SSM_unit` | unit, in-loop max abs 4.5e-8; formal depth 140 PASS |
| `attention_unit` | unit 1536/1536 bit-exact, in-loop bit-exact; formal depth 220 |
| `mlp_unit` / `SwiGLU_unit` | unit bit-exact, in-loop 0/3072 outside 2e-2; formal 260 |
| `mamba2_unit` | full-config unit 768/768 bit-exact, in-loop 0/3840 outside 2e-2; formal 260 |
| `conv1d_unit` | unit + in-loop bit-exact (fp32 accumulation); formal depth 120 |
| `RMSNorm_unit` | unit + in-loop PASS (fp32 datapath, eps 1e-5) |
| `softmax_unit` | in-loop 3888/3888 rows; formal 120 (coarse LUT exp, documented) |
| `sigmoid_unit`, `SiLU_unit` | formal depth 10; LUT-based approximations |
| `output_projection_unit` | unit 256/256, in-loop 1024/1024 bit-exact (512 sampled rows); formal 60 |
| `granite_layer` | single-token in-loop PASS (max abs 0.0078); e2e hybrid PASS (same argmax/top-5) |

## Simulation cost (why the full 32-layer run is not simulated)

Measured with Verilator at ~3–6k cycles/s: weight load 1 beat/cycle
(Mamba2 layer ~3.8M, full decoder layer ~8.5M), SSM ~1.19M cycles/token,
attention ~1.6M, MLP ~4.8M, Mamba2 layer ~5.1M, full 32-layer token ~169M
(~8–15 h) plus ~350M weight-load beats. Hence the accepted end-to-end
verification is `granite_layer` + `tb_granite_layer_e2e.py`.

## Numerics rules

The model is the source of truth; goldens must be generated from torch
semantics, never from an emulation of the RTL. Where the model computes in
fp32, instantiate the FP unit with fp32 dimensions (`W_EXP=8`, `W_MANT=23`),
widen bfloat16 operands in the high half and round to bfloat16 once at the
output. `eps`/scale constants must match the config (1e-5, 0.246, /3 ...).
See `AGENTS.md` for the full list and the SymbiYosys gotchas.

## Known gaps

- Attention-type variant of `granite_layer` (attention layers 10/13/17/27):
  needs an `attention_unit` wrapper (same residual/norm structure, plus the
  position input) and its own in-loop verification.
- `softmax_unit`'s LUT exp is coarse (max rel ~0.21); the attention unit uses
  the accurate `attn_softmax_seq` instead. Making `softmax_unit` accurate means
  replacing the row pipeline's LUT exp with the serial `fp_exp_seq`.
- `fp_unit` is spec-only here; the implementation lives in the submodule.
  (`RMSNorm_unit/formal` now includes `fp32_to_bf16_round.sv`; BMC re-run.)
