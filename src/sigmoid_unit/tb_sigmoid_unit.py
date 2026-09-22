"""cocotb testbench for sigmoid_unit."""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import struct
import os

def bfloat16_to_float(val):
    """Convert 16-bit bfloat16 pattern to Python float."""
    val = val & 0xFFFF
    float32_val = val << 16
    return struct.unpack('f', struct.pack('I', float32_val))[0]

def float_to_bfloat16(val):
    """Convert Python float to 16-bit bfloat16 pattern."""
    f32 = struct.pack('f', val)
    i32 = struct.unpack('I', f32)[0]
    rounding_bias = (1 << 15) + ((i32 >> 16) & 1)
    i32 += rounding_bias
    bf16 = i32 >> 16
    return bf16 & 0xFFFF

@cocotb.test()
async def test_sigmoid_unit(dut):
    """Test the sigmoid_unit against golden sample."""

    # Start clock
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.data_in.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Load golden sample
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inputs_path = os.path.join(script_dir, "golden_inputs.hex")
    outputs_path = os.path.join(script_dir, "golden_outputs.hex")

    with open(inputs_path, 'r') as f:
        golden_inputs = [int(line.strip(), 16) for line in f if line.strip()]
    with open(outputs_path, 'r') as f:
        golden_outputs = [int(line.strip(), 16) for line in f if line.strip()]

    assert len(golden_inputs) == len(golden_outputs), \
        f"Input/output count mismatch: {len(golden_inputs)} vs {len(golden_outputs)}"

    num_vectors = len(golden_inputs)
    dut._log.info(f"Running {num_vectors} test vectors")

    mismatches = 0
    max_abs_error = 0.0
    max_rel_error = 0.0

    for i in range(num_vectors):
        # Drive input
        dut.data_in.value = golden_inputs[i]
        dut.valid_in.value = 1
        await RisingEdge(dut.clk)
        dut.valid_in.value = 0

        # Wait for result (1 cycle latency)
        await RisingEdge(dut.clk)

        # Check output
        actual = dut.data_out.value.integer & 0xFFFF
        expected = golden_outputs[i]

        if actual != expected:
            # Allow for rounding differences - compare as float
            actual_float = bfloat16_to_float(actual)
            expected_float = bfloat16_to_float(expected)
            abs_err = abs(actual_float - expected_float)
            if expected_float != 0.0:
                rel_err = abs_err / abs(expected_float)
            else:
                rel_err = abs_err

            if abs_err > 0.02:  # More than 2% absolute error
                mismatches += 1
                if mismatches <= 20:  # Print first 20 mismatches
                    dut._log.warning(
                        f"Vector {i}: input=0x{golden_inputs[i]:04X} "
                        f"({bfloat16_to_float(golden_inputs[i]):.6f}), "
                        f"expected=0x{expected:04X} ({expected_float:.6f}), "
                        f"actual=0x{actual:04X} ({actual_float:.6f}), "
                        f"abs_err={abs_err:.6f}"
                    )

            if abs_err > max_abs_error:
                max_abs_error = abs_err
            if rel_err > max_rel_error:
                max_rel_error = rel_err

    dut._log.info(f"Results: {num_vectors - mismatches}/{num_vectors} passed")
    dut._log.info(f"Max absolute error: {max_abs_error:.6f}")
    dut._log.info(f"Max relative error: {max_rel_error:.6f}")

    if mismatches > 0:
        dut._log.error(f"FAILED: {mismatches} mismatches out of {num_vectors}")
        assert False, f"Test failed with {mismatches} mismatches"
    else:
        dut._log.info("PASSED: All test vectors matched")
