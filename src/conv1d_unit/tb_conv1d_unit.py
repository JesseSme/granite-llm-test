"""cocotb testbench for conv1d_unit."""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import struct
import os


def bfloat16_to_float(val):
    val = val & 0xFFFF
    float32_val = val << 16
    return struct.unpack('f', struct.pack('I', float32_val))[0]


@cocotb.test()
async def test_conv1d_unit(dut):
    """Test conv1d_unit against golden sample."""

    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst_n.value = 0
    dut.valid_i.value = 0
    dut.data_i.value = 0
    dut.load_en.value = 0
    dut.load_ch.value = 0
    dut.load_tap.value = 0
    dut.load_wdata.value = 0
    dut.load_is_bias.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Load golden sample
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "golden_weights.hex")
    biases_path = os.path.join(script_dir, "golden_biases.hex")
    inputs_path = os.path.join(script_dir, "golden_inputs.hex")
    outputs_path = os.path.join(script_dir, "golden_outputs.hex")

    with open(weights_path, 'r') as f:
        golden_weights = [int(line.strip(), 16) for line in f if line.strip()]
    with open(biases_path, 'r') as f:
        golden_biases = [int(line.strip(), 16) for line in f if line.strip()]
    with open(inputs_path, 'r') as f:
        golden_inputs = [int(line.strip(), 16) for line in f if line.strip()]
    with open(outputs_path, 'r') as f:
        golden_outputs = [int(line.strip(), 16) for line in f if line.strip()]

    CHANNELS = 1536
    KERNEL = 4

    # Load weights
    dut._log.info("Loading weights...")
    for c in range(CHANNELS):
        for k in range(KERNEL):
            idx = c * KERNEL + k
            dut.load_en.value = 1
            dut.load_ch.value = c
            dut.load_tap.value = k
            dut.load_wdata.value = golden_weights[idx]
            dut.load_is_bias.value = 0
            await RisingEdge(dut.clk)
    dut.load_en.value = 0
    await RisingEdge(dut.clk)

    # Load biases
    dut._log.info("Loading biases...")
    for c in range(CHANNELS):
        dut.load_en.value = 1
        dut.load_ch.value = c
        dut.load_wdata.value = golden_biases[c]
        dut.load_is_bias.value = 1
        await RisingEdge(dut.clk)
    dut.load_en.value = 0
    await RisingEdge(dut.clk)

    dut._log.info("Running convolution test...")

    SEQ_LEN = len(golden_inputs) // CHANNELS
    mismatches = 0
    max_abs_error = 0.0

    for t in range(SEQ_LEN):
        for c in range(CHANNELS):
            idx = t * CHANNELS + c

            # Wait for ready
            while int(dut.ready_o.value) == 0:
                await RisingEdge(dut.clk)

            # Drive input
            dut.data_i.value = golden_inputs[idx]
            dut.valid_i.value = 1
            await RisingEdge(dut.clk)
            dut.valid_i.value = 0

            # Wait for output (8 cycles pipeline)
            while int(dut.valid_o.value) == 0:
                await RisingEdge(dut.clk)

            # Check output
            actual = int(dut.data_o.value) & 0xFFFF
            expected = golden_outputs[idx]

            if actual != expected:
                actual_float = bfloat16_to_float(actual)
                expected_float = bfloat16_to_float(expected)
                abs_err = abs(actual_float - expected_float)

                if abs_err > 0.1:
                    mismatches += 1
                    if mismatches <= 10:
                        dut._log.warning(
                            f"t={t} c={c}: expected=0x{expected:04X} "
                            f"({expected_float:.6f}), actual=0x{actual:04X} "
                            f"({actual_float:.6f}), err={abs_err:.6f}"
                        )

                if abs_err > max_abs_error:
                    max_abs_error = abs_err

    total = SEQ_LEN * CHANNELS
    dut._log.info(f"Results: {total - mismatches}/{total} passed")
    dut._log.info(f"Max absolute error: {max_abs_error:.6f}")

    if mismatches > 0:
        dut._log.error(f"FAILED: {mismatches} mismatches")
        assert False, f"Test failed with {mismatches} mismatches"
    else:
        dut._log.info("PASSED: All test vectors matched")
