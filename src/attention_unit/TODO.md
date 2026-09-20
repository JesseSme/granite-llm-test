# attention_unit — TODO

## Code Review Checklist

- [x] RTL passes Verilator lint with `-Wall` (no warnings; checked at
      16/2/1/8/4, 192/6/2/32/8 and 768/12/4/64/64).
- [x] cocotb testbench passes against golden sample for all test vectors
      (small config 64/64 and full-dim config 1536/1536, both bit-exact vs the
      fp32 golden).
- [x] In-loop test passes — real Granite attention layer 10, 9 tokens:
      bit-exact vs the model's eager attention output (max abs/rel 0.0).
- [x] `description.yaml` is updated with interface, Fmax, and characteristics.
- [x] No hardcoded magic numbers — use parameters or `define` constants
      (the attention multiplier is a named localparam).
- [x] Follows existing code conventions (see `systemverilog_fp_unit/rtl/`).

## Implementation Steps

- [x] 1. Set `current_dut: attention_unit` in description.yaml
- [x] 2. Implement `attn_softmax_seq.sv` (row softmax with accurate fp_exp_seq)
- [x] 3. Implement `attention_unit.sv` (GQA: Q/K/V proj + scores + softmax + context + O proj)
- [x] 4. Lint with Verilator (`verilator --lint-only -Wall`)
- [x] 5. Generate golden sample (`gen_golden.py`, fp32 reference)
- [x] 6. Write cocotb testbench (`tb_attention_unit.py`) and runner
- [x] 7. Run RTL simulation against golden sample
- [x] 8. In-loop verification with real Granite attention inputs
- [x] 9. Formal verification (`formal/bmc.sby`)
- [x] 10. Update `description.yaml` — interface, Fmax, characteristics
- [x] 11. Clear `current_dut` field

## Notes / issues found

- Architecture: the four projections reuse `matrix_unit` instances (Q 768->768,
  K/V 768->256, O 768->768); the input vector is broadcast to Q/K/V in
  parallel, then drained, then the per-head score/softmax/context loop runs,
  then the context vector is fed to the O projection. K/V are cached on chip
  (bf16), so one frame = one sequence position and causality is implicit.
- `attn_softmax_seq.sv` is an internal streaming row-softmax with the accurate
  `fp_exp_seq` (borrowed from SSM_unit): the LUT-based `fp_exp` in
  softmax_unit (~20% relative error) is too coarse. Its fp_unit operands must
  be held through the wait states, because fp_unit has registered outputs —
  the first version let the operands fall back to 0 during the wait state,
  which made the exp always see 0 (uniform probabilities) and latched a 0/0
  NaN from the divider.
- Both TB comparison bugs found during bring-up: NaN outputs pass a
  `abs > tol and rel > tol` check, and a bf16->fp32 conversion that places the
  bits in the low half turns NaN into tiny denormals (a silent false pass).
  The unit TB now asserts no non-finite outputs and widens bf16 correctly; the
  in-loop TB keeps the captured hidden states in bf16 (calling `.float()` on
  them made `view(torch.uint16)` reinterpret fp32 words as uint16).
- Verified bit-exactly against the model because the model rounds the softmax
  probabilities to bf16; the ~1e-5 poly-exp error is far below the bf16
  rounding step and does not change the rounded probabilities.
