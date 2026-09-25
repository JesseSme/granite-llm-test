# granite_layer - TODO

- [x] Single-layer (mamba type) RTL wrapper
- [x] Verilator -Wall clean
- [x] Single-token in-loop test with real layer-0 weights: PASSED
      (max abs 0.0078, 0 mismatches)
- [ ] Update rmsnorm_unit's own golden/unit test to the fp32 datapath
- [ ] Attention-type layer variant
- [ ] Formal properties for the wrapper
- [ ] Fix the stale "mamba2 unit" label printed by run_test.py
