"""cocotb testbench for softmax_unit."""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import struct
import os

def float_to_hex(val):
    """Convert Python float to float32 hex pattern."""
    return struct.unpack('I', struct.pack('f', val))[0]

def hex_to_float(hex_val):
    """Convert float32 hex pattern to Python float."""
    hex_val = hex_val & 0xFFFFFFFF
    return struct.unpack('f', struct.pack('I', hex_val))[0]

@cocotb.test()
async def test_softmax_unit(dut):
    """Test the softmax_unit against golden sample."""

    N = 8  # elements per row

    # Start clock
    clock = Clock(dut.clk, 10, unit='ns')
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.data_in.value = 0
    dut.last_in.value = 0
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

    num_values = len(golden_inputs)
    num_rows = num_values // N
    dut._log.info(f"Running {num_rows} rows, {num_values} total values")

    mismatches = 0
    max_abs_error = 0.0

    for row_idx in range(num_rows):
        # --- Feed row elements ---
        for i in range(N):
            val_idx = row_idx * N + i
            dut.data_in.value = golden_inputs[val_idx]
            dut.valid_in.value = 1
            dut.last_in.value = 1 if (i == N - 1) else 0
            await RisingEdge(dut.clk)
        dut.valid_in.value = 0
        dut.last_in.value = 0

        # --- Collect outputs as they appear during NORM phase ---
        # Don't wait for done first - collect valid_out pulses
        row_outputs = []
        row_mismatches = 0
        timeout = N * 20  # generous timeout per row

        for _ in range(timeout):
            await RisingEdge(dut.clk)

            if dut.valid_out.value == 1:
                actual = dut.data_out.value.to_unsigned() & 0xFFFFFFFF
                row_outputs.append(actual)

            if dut.done.value == 1:
                break

        # --- Verify outputs ---
        row_sum = 0.0
        for i, actual_hex in enumerate(row_outputs):
            if i >= N:
                break
            val_idx = row_idx * N + i
            expected = golden_outputs[val_idx]
            actual_float = hex_to_float(actual_hex)
            expected_float = hex_to_float(expected)
            abs_err = abs(actual_float - expected_float)
            row_sum += actual_float

            if abs_err > 0.05:
                row_mismatches += 1
                mismatches += 1
                if mismatches <= 20:
                    dut._log.warning(
                        f"Row {row_idx}, elem {i}: "
                        f"expected=0x{expected:08X} ({expected_float:.6f}), "
                        f"actual=0x{actual_hex:08X} ({actual_float:.6f}), "
                        f"abs_err={abs_err:.6f}"
                    )

            if abs_err > max_abs_error:
                max_abs_error = abs_err

        # Check if we got all outputs
        if len(row_outputs) < N:
            dut._log.warning(f"Row {row_idx}: only {len(row_outputs)}/{N} outputs")
            mismatches += (N - len(row_outputs))

        # Verify row sums to ~1.0
        if len(row_outputs) >= N:
            sum_err = abs(row_sum - 1.0)
            if sum_err > 0.1:
                dut._log.warning(f"Row {row_idx}: sum={row_sum:.6f} (err={sum_err:.6f})")

        if row_mismatches > 0:
            dut._log.warning(f"Row {row_idx}: {row_mismatches}/{N} mismatches")

    total = num_values
    passed = total - mismatches
    dut._log.info(f"Results: {passed}/{total} passed ({mismatches} mismatches)")
    dut._log.info(f"Max absolute error: {max_abs_error:.6f}")

    if mismatches > 0:
        dut._log.error(f"FAILED: {mismatches} mismatches out of {total}")
        assert False, f"Test failed with {mismatches} mismatches"
    else:
        dut._log.info("PASSED: All test vectors matched")
