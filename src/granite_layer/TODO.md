# granite_layer - TODO

- [x] Single-layer (mamba type) RTL wrapper
- [x] Verilator -Wall clean
- [x] Single-token in-loop test with real layer-0 weights: PASSED
      (max abs 0.0078, 0 mismatches)
- [ ] Update rmsnorm_unit's own golden/unit test to the fp32 datapath
- [x] End-to-end hybrid test (RTL layer 0 + 31 software layers): PASSED,
      same argmax and top-5 as the baseline; re-run after the LANES=4 /
      fp_mul_pipe2 optimizations with identical results
- [ ] Attention-type layer variant
- [ ] Formal properties for the wrapper
- [x] Fix the stale "mamba2 unit" label printed by run_test.py (the
      docstring/prints now say granite_layer; e2e and in-loop have their own
      labels)
