"""Debug testbench for conv1d_unit - traces first element."""
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
async def test_conv1d_debug(dut):
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())

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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Load just ch0 weight/bias
    with open(os.path.join(script_dir, 'golden_weights.hex')) as f:
        all_weights = [int(l.strip(), 16) for l in f if l.strip()]
    with open(os.path.join(script_dir, 'golden_biases.hex')) as f:
        all_biases = [int(l.strip(), 16) for l in f if l.strip()]
    with open(os.path.join(script_dir, 'golden_inputs.hex')) as f:
        all_inputs = [int(l.strip(), 16) for l in f if l.strip()]
    with open(os.path.join(script_dir, 'golden_outputs.hex')) as f:
        all_outputs = [int(l.strip(), 16) for l in f if l.strip()]

    CHANNELS = 1536
    KERNEL = 4

    # Load weight for ch0 only
    for k in range(KERNEL):
        dut.load_en.value = 1
        dut.load_ch.value = 0
        dut.load_tap.value = k
        dut.load_wdata.value = all_weights[k]
        dut.load_is_bias.value = 0
        await RisingEdge(dut.clk)
    dut.load_en.value = 0
    await RisingEdge(dut.clk)

    # Load bias for ch0
    dut.load_en.value = 1
    dut.load_ch.value = 0
    dut.load_wdata.value = all_biases[0]
    dut.load_is_bias.value = 1
    await RisingEdge(dut.clk)
    dut.load_en.value = 0
    await RisingEdge(dut.clk)

    # Feed first element (t=0, c=0)
    inp = all_inputs[0]
    expected = all_outputs[0]
    dut._log.info(f"Input: 0x{inp:04X} ({bfloat16_to_float(inp):.4f})")
    dut._log.info(f"Expected: 0x{expected:04X} ({bfloat16_to_float(expected):.4f})")

    # Wait for ready
    while int(dut.ready_o.value) == 0:
        await RisingEdge(dut.clk)

    dut.data_i.value = inp
    dut.valid_i.value = 1
    await RisingEdge(dut.clk)
    dut.valid_i.value = 0

    # Trace for 20 cycles
    for cycle in range(20):
        await RisingEdge(dut.clk)
        state = dut.state.value
        valid_o = int(dut.valid_o.value)
        data_o = int(dut.data_o.value) & 0xFFFF
        ready_o = int(dut.ready_o.value)
        fpu_y_val = int(dut.fpu_y.value) & 0xFFFF
        acc_val = int(dut.acc.value) & 0xFFFF
        
        dut._log.info(
            f"cycle={cycle}: state={state} ready={ready_o} valid_o={valid_o} "
            f"data_o=0x{data_o:04X}({bfloat16_to_float(data_o):.4f}) "
            f"fpu_y=0x{fpu_y_val:04X}({bfloat16_to_float(fpu_y_val):.4f}) "
            f"acc=0x{acc_val:04X}({bfloat16_to_float(acc_val):.4f})"
        )
        
        if valid_o:
            dut._log.info(f"OUTPUT: 0x{data_o:04X} ({bfloat16_to_float(data_o):.4f}), expected=0x{expected:04X} ({bfloat16_to_float(expected):.4f})")
            break
